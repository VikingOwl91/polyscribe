// SPDX-License-Identifier: MIT

//! Cache-first YouTube audio acquisition and local transcription.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use polyscribe_core::backend::{
    BackendKind, ModelSession, select_backend, transcribe_with_decoder_timeout,
};
use polyscribe_core::find_model_file;
use polyscribe_core::ui::progress::TranscriptProgress;
use polyscribe_core::youtube::{
    CaptionCue, NormalizationStats, VideoMetadata, YOUTUBE_NORMALIZATION_VERSION, render_srt,
    render_txt, validate_youtube_url,
};
use serde::Serialize;
use tempfile::TempDir;

use crate::youtube_cache::{
    CACHE_SCHEMA_VERSION, CacheIdentity, CacheRecord, ExecutableIdentity, Lookup, RequestKey,
    SourceIdentity, VideoSummary, YoutubeCache, model_stat,
};
use crate::youtube_process::{self, ProcessFailure, ProcessOutput, ProcessPolicy};

const YTDLP_TIMEOUT: Duration = Duration::from_secs(120);
const AUDIO_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const AUDIO_DECODER_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const MAX_METADATA_BYTES: usize = 50 * 1024 * 1024;
const MAX_AUDIO_BYTES: u64 = 250 * 1024 * 1024;
const MAX_STDERR_BYTES: usize = 32 * 1024;
const AUDIO_POLICY_VERSION: u32 = 2;

struct DownloadContext<'a> {
    ytdlp: &'a Path,
    environment: &'a [(OsString, OsString)],
    directory: &'a Path,
    cache: Option<&'a YoutubeCache>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VoxtralConfigSnapshot {
    executable: Option<OsString>,
    path_bytes: Option<Vec<u8>>,
    executable_stat: Option<ExecutableIdentity>,
    cache_eligible: bool,
}

impl VoxtralConfigSnapshot {
    fn from_environment() -> Self {
        let executable = std::env::var_os("POLYSCRIBE_VOXTRAL");
        let path_bytes = executable
            .as_deref()
            .map(|value| value.as_encoded_bytes().to_vec());
        let executable_stat = executable.as_deref().and_then(|value| {
            let path = Path::new(value);
            path.is_absolute()
                .then(|| crate::youtube_cache::executable_identity(path).ok())
                .flatten()
        });
        let cache_eligible = executable.is_none() || executable_stat.is_some();
        Self {
            executable,
            path_bytes,
            executable_stat,
            cache_eligible,
        }
    }

    fn is_current(&self) -> bool {
        self == &Self::from_environment()
    }
}

struct FileLimitPolicy<'a> {
    directory: &'a Path,
    prefix: &'static str,
    max_bytes: u64,
    overflow_kind: YouTubeErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum YouTubeErrorKind {
    InvalidUrl,
    InvalidLanguage,
    UnsupportedFormat,
    UnsupportedMode,
    ToolUnavailable,
    RateLimited,
    Forbidden,
    Network,
    Downloader,
    DownloaderTimeout,
    DownloaderOutputTooLarge,
    MalformedMetadata,
    AudioDownload,
    AudioDecode,
    Whisper,
}

#[derive(Debug)]
struct YouTubeError {
    kind: YouTubeErrorKind,
    message: String,
}

impl std::fmt::Display for YouTubeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.kind, self.message)
    }
}

impl std::fmt::Display for YouTubeErrorKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::InvalidUrl => "invalid_url",
            Self::InvalidLanguage => "invalid_language",
            Self::UnsupportedFormat => "unsupported_format",
            Self::UnsupportedMode => "unsupported_mode",
            Self::ToolUnavailable => "tool_unavailable",
            Self::RateLimited => "rate_limited",
            Self::Forbidden => "forbidden",
            Self::Network => "network",
            Self::Downloader => "downloader",
            Self::DownloaderTimeout => "downloader_timeout",
            Self::DownloaderOutputTooLarge => "downloader_output_too_large",
            Self::MalformedMetadata => "malformed_metadata",
            Self::AudioDownload => "audio_download",
            Self::AudioDecode => "audio_decode",
            Self::Whisper => "whisper",
        };
        formatter.write_str(name)
    }
}

impl std::error::Error for YouTubeError {}

fn youtube_error(kind: YouTubeErrorKind, message: impl Into<String>) -> anyhow::Error {
    YouTubeError {
        kind,
        message: message.into(),
    }
    .into()
}

fn failure_kind(error: &anyhow::Error) -> Option<YouTubeErrorKind> {
    error.downcast_ref::<YouTubeError>().map(|error| error.kind)
}

#[derive(Serialize)]
struct JsonTranscript<'a> {
    source: &'a str,
    source_url: &'a str,
    video_id: &'a str,
    title: &'a Option<String>,
    uploader: &'a Option<String>,
    duration_seconds: Option<f64>,
    requested_language: Option<&'a str>,
    language: Option<&'a str>,
    engine: &'a str,
    model: &'a str,
    caption_language: Option<&'a str>,
    is_automatic: Option<bool>,
    is_translated: Option<bool>,
    cue_count: usize,
    normalization: &'a NormalizationStats,
    cues: &'a [CaptionCue],
}

/// Run the `youtube` subcommand.
#[allow(clippy::too_many_arguments)]
pub fn run(
    url: String,
    language: Option<String>,
    format: String,
    output: Option<PathBuf>,
    _prefer_manual: bool,
    refresh: bool,
    captions_only: bool,
    _audio_only: bool,
) -> Result<()> {
    let yt_url = validate_youtube_url(&url)
        .map_err(|error| youtube_error(YouTubeErrorKind::InvalidUrl, error.to_string()))?;
    let source_url = url.trim().to_string();
    if !matches!(format.as_str(), "txt" | "srt" | "json") {
        return Err(youtube_error(
            YouTubeErrorKind::UnsupportedFormat,
            "unsupported YouTube output format",
        ));
    }
    if let Some(path) = &output {
        validate_output_path(path)?;
    }
    let language = validate_language(language)?;
    if captions_only {
        return Err(youtube_error(
            YouTubeErrorKind::UnsupportedMode,
            "--captions-only is unsupported; YouTube transcription uses audio only",
        ));
    }
    let voxtral_config = VoxtralConfigSnapshot::from_environment();
    let request = RequestKey {
        cache_schema_version: CACHE_SCHEMA_VERSION,
        normalization_version: YOUTUBE_NORMALIZATION_VERSION,
        selection_policy_version: AUDIO_POLICY_VERSION,
        video_id: yt_url.video_id.clone(),
        requested_language: language.clone(),
        voxtral_executable: voxtral_config.path_bytes.clone(),
        voxtral_executable_identity: voxtral_config.executable_stat.clone(),
    };

    let cache = open_cache();
    let cache = if voxtral_config.cache_eligible {
        cache
    } else {
        polyscribe_core::dlog!(
            1,
            "Voxtral executable identity could not be inspected; bypassing the transcript cache."
        );
        None
    };
    if !refresh && let Some(cache) = &cache {
        match cache.lookup(&request) {
            Ok(Lookup::Hit(record)) if cache_record_is_current(&record) => {
                return write_transcript(
                    &source_url,
                    &yt_url.video_id,
                    &format,
                    output.as_deref(),
                    &record,
                );
            }
            Ok(Lookup::Hit(_)) | Ok(Lookup::Invalid) => {
                polyscribe_core::dlog!(1, "Ignoring a stale or invalid YouTube cache entry.");
            }
            Ok(Lookup::Miss) => {}
            Err(error) => {
                polyscribe_core::dlog!(1, "Ignoring unreadable YouTube cache: {error:#}");
            }
        }
    }

    let ytdlp = find_ytdlp()?;
    let scratch = TempDir::new().context("create private temporary directory for yt-dlp")?;
    let environment = youtube_process::safe_environment(scratch.path());
    let download_context = DownloadContext {
        ytdlp: &ytdlp,
        environment: &environment,
        directory: scratch.path(),
        cache: cache.as_ref(),
    };
    polyscribe_core::ilog!("Fetching YouTube metadata...");
    let metadata = fetch_metadata(
        &ytdlp,
        &environment,
        &yt_url.video_id,
        &yt_url.canonical_url,
    )?;
    let record = transcribe_audio(
        &download_context,
        &yt_url.canonical_url,
        summarize_metadata(&metadata),
        language.as_deref(),
        &request,
        &voxtral_config,
    )?;
    write_transcript(
        &source_url,
        &yt_url.video_id,
        &format,
        output.as_deref(),
        &record,
    )
}

fn validate_language(language: Option<String>) -> Result<Option<String>> {
    let Some(language) = language else {
        return Ok(None);
    };
    let language = language.trim();
    if language.is_empty()
        || language.len() > 64
        || language.eq_ignore_ascii_case("auto")
        || !language
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(youtube_error(
            YouTubeErrorKind::InvalidLanguage,
            "language must be a code containing only ASCII letters, digits, '-' or '_' and be at most 64 bytes; omit --language for automatic selection",
        ));
    }
    Ok(Some(language.to_ascii_lowercase()))
}

fn open_cache() -> Option<YoutubeCache> {
    match YoutubeCache::open_default(true) {
        Ok(cache) => cache,
        Err(error) => {
            polyscribe_core::dlog!(1, "YouTube transcript caching is unavailable: {error:#}");
            None
        }
    }
}

fn cache_record_is_current(record: &CacheRecord) -> bool {
    let SourceIdentity::LocalWhisper {
        model_stat: cached, ..
    } = &record.identity.source
    else {
        return true;
    };
    match find_model_file() {
        Ok(path) => cache_model_stat_matches(cached, model_stat(&path)),
        Err(_) => true,
    }
}

fn cache_model_stat_matches(
    cached: &crate::youtube_cache::ModelStat,
    current: Result<crate::youtube_cache::ModelStat>,
) -> bool {
    current.is_ok_and(|current| &current == cached)
}

fn store_cache(cache: Option<&YoutubeCache>, record: &CacheRecord) {
    if let Some(cache) = cache
        && let Err(error) = cache.store(record)
    {
        polyscribe_core::dlog!(1, "Could not save YouTube transcript cache: {error:#}");
    }
}

fn summarize_metadata(metadata: &VideoMetadata) -> VideoSummary {
    VideoSummary {
        title: metadata.title.clone(),
        uploader: metadata.uploader.clone(),
        duration_seconds: metadata.duration,
    }
}

fn find_ytdlp() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("POLYSCRIBE_YTDLP") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(youtube_error(
            YouTubeErrorKind::ToolUnavailable,
            format!(
                "POLYSCRIBE_YTDLP does not name a file: {}",
                polyscribe_core::ui::terminal_safe_path(&path)
            ),
        ));
    }
    if let Ok(path) = which::which("yt-dlp") {
        return Ok(path);
    }
    Err(youtube_error(
        YouTubeErrorKind::ToolUnavailable,
        "yt-dlp not found on PATH. Install it with: pip install yt-dlp",
    ))
}

fn common_args() -> Vec<OsString> {
    [
        "--ignore-config",
        "--no-cache-dir",
        "--no-playlist",
        "--no-write-subs",
        "--no-write-auto-subs",
        "--retries",
        "0",
        "--fragment-retries",
        "0",
        "--extractor-retries",
        "0",
    ]
    .into_iter()
    .map(OsString::from)
    .collect()
}

fn fetch_metadata(
    ytdlp: &Path,
    environment: &[(OsString, OsString)],
    expected_video_id: &str,
    url: &str,
) -> Result<VideoMetadata> {
    let mut args = common_args();
    args.push("--skip-download".into());
    args.push("--dump-single-json".into());
    args.push(url.into());
    let output = run_ytdlp(
        ytdlp,
        &args,
        environment,
        ProcessPolicy {
            timeout: YTDLP_TIMEOUT,
            stdout_limit: MAX_METADATA_BYTES,
            stderr_limit: MAX_STDERR_BYTES,
        },
    )?;
    ensure_success(&output, "yt-dlp metadata request")?;
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).map_err(|error| {
        youtube_error(
            YouTubeErrorKind::Downloader,
            format!("yt-dlp returned invalid metadata JSON: {error}"),
        )
    })?;
    if !value.is_object() {
        return Err(youtube_error(
            YouTubeErrorKind::MalformedMetadata,
            "yt-dlp metadata root must be a JSON object",
        ));
    }
    let id = value.get("id").and_then(|v| v.as_str()).unwrap_or_default();
    if id.is_empty() {
        return Err(youtube_error(
            YouTubeErrorKind::Downloader,
            "yt-dlp metadata did not include a video id",
        ));
    }
    if id != expected_video_id {
        return Err(youtube_error(
            YouTubeErrorKind::Downloader,
            format!(
                "yt-dlp returned a different video id for requested video '{expected_video_id}'"
            ),
        ));
    }
    Ok(VideoMetadata {
        id: id.to_string(),
        title: non_empty_string(value.get("title")),
        uploader: non_empty_string(value.get("uploader")),
        duration: value.get("duration").and_then(serde_json::Value::as_f64),
    })
}

fn non_empty_string(value: Option<&serde_json::Value>) -> Option<String> {
    value
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty() && *value != "NA")
        .map(ToString::to_string)
}

fn download_audio(context: &DownloadContext<'_>, url: &str) -> Result<PathBuf> {
    polyscribe_core::ilog!("Downloading YouTube audio...");
    let template = context.directory.join("audio.%(ext)s");
    let mut args = common_args();
    args.extend([
        "--format".into(),
        "bestaudio/best".into(),
        "--max-filesize".into(),
        format!("{}M", MAX_AUDIO_BYTES / (1024 * 1024)).into(),
        "--no-progress".into(),
        "--output".into(),
        template.as_os_str().to_os_string(),
        url.into(),
    ]);
    let output = run_ytdlp_with_file_limit(
        context.ytdlp,
        &args,
        context.environment,
        FileLimitPolicy {
            directory: context.directory,
            prefix: "audio.",
            max_bytes: MAX_AUDIO_BYTES,
            overflow_kind: YouTubeErrorKind::AudioDownload,
        },
        ProcessPolicy {
            timeout: AUDIO_DOWNLOAD_TIMEOUT,
            stdout_limit: 1024 * 1024,
            stderr_limit: MAX_STDERR_BYTES,
        },
    )
    .map_err(wrap_audio_download_error)?;
    ensure_success(&output, "yt-dlp audio download").map_err(wrap_audio_download_error)?;
    find_single_audio_artifact(context.directory, MAX_AUDIO_BYTES)?.ok_or_else(|| {
        youtube_error(
            YouTubeErrorKind::AudioDownload,
            "yt-dlp produced no audio file",
        )
    })
}

fn wrap_audio_download_error(error: anyhow::Error) -> anyhow::Error {
    if failure_kind(&error).is_some_and(|kind| {
        matches!(
            kind,
            YouTubeErrorKind::RateLimited
                | YouTubeErrorKind::Forbidden
                | YouTubeErrorKind::Network
                | YouTubeErrorKind::ToolUnavailable
                | YouTubeErrorKind::DownloaderTimeout
                | YouTubeErrorKind::DownloaderOutputTooLarge
                | YouTubeErrorKind::AudioDownload
        )
    }) {
        error
    } else {
        youtube_error(
            YouTubeErrorKind::AudioDownload,
            format!("could not download YouTube audio: {error:#}"),
        )
    }
}

fn transcribe_audio(
    context: &DownloadContext<'_>,
    url: &str,
    metadata: VideoSummary,
    language: Option<&str>,
    request: &RequestKey,
    voxtral_config: &VoxtralConfigSnapshot,
) -> Result<CacheRecord> {
    let audio_path = download_audio(context, url)?;
    polyscribe_core::ilog!("YouTube audio downloaded; starting transcription...");
    let (source, cues) = transcribe_with_whisper_fallback(
        language,
        || {
            crate::youtube_voxtral::transcribe(
                voxtral_config.executable.clone(),
                &audio_path,
                language,
                context.directory,
                &polyscribe_core::stream::CancellationToken::new(),
            )
        },
        || transcribe_whisper(&audio_path, language),
    )?;
    let record = CacheRecord {
        schema_version: CACHE_SCHEMA_VERSION,
        normalization_version: YOUTUBE_NORMALIZATION_VERSION,
        identity: CacheIdentity {
            request: request.clone(),
            source,
        },
        acquired_at: chrono::Utc::now().to_rfc3339(),
        metadata,
        original_cues: cues.clone(),
        cues,
        normalization: NormalizationStats::default(),
    };
    if context.cache.is_some() && !voxtral_config.is_current() {
        polyscribe_core::dlog!(
            1,
            "Voxtral configuration changed during transcription; skipping the cache write."
        );
    } else {
        store_cache(context.cache, &record);
    }
    Ok(record)
}

fn transcribe_with_whisper_fallback(
    language: Option<&str>,
    voxtral: impl FnOnce() -> Result<
        crate::youtube_voxtral::VoxtralTranscript,
        crate::youtube_voxtral::VoxtralError,
    >,
    whisper: impl FnOnce() -> Result<(SourceIdentity, Vec<CaptionCue>)>,
) -> Result<(SourceIdentity, Vec<CaptionCue>)> {
    match voxtral() {
        Ok(transcript) => Ok((
            SourceIdentity::LocalVoxtral {
                model: transcript.model,
                language: transcript.language,
                language_hint: language.map(ToString::to_string),
                protocol_version: 1,
            },
            transcript.cues,
        )),
        Err(voxtral_error) => {
            let reason = if matches!(
                voxtral_error,
                crate::youtube_voxtral::VoxtralError::NotConfigured
            ) {
                "Voxtral is not configured".to_string()
            } else {
                format!("Voxtral failed ({voxtral_error})")
            };
            polyscribe_core::ilog!("{reason}; trying local Whisper once.");
            whisper().map_err(|whisper_error| {
                let kind = failure_kind(&whisper_error).unwrap_or(YouTubeErrorKind::Whisper);
                youtube_error(
                    kind,
                    format!(
                        "Voxtral failed ({voxtral_error}); local Whisper fallback failed: {whisper_error:#}"
                    ),
                )
            })
        }
    }
}

fn transcribe_whisper(
    audio_path: &Path,
    language: Option<&str>,
) -> Result<(SourceIdentity, Vec<CaptionCue>)> {
    polyscribe_core::ilog!("Locating Whisper model...");
    let model_path = find_model_file().map_err(|error| {
        youtube_error(
            YouTubeErrorKind::Whisper,
            format!("no local Whisper model is available: {error:#}"),
        )
    })?;
    let stat = model_stat(&model_path).map_err(|error| {
        youtube_error(
            YouTubeErrorKind::Whisper,
            format!("could not inspect the local Whisper model: {error:#}"),
        )
    })?;
    let selection = select_backend(BackendKind::Auto, polyscribe_core::verbose_level() > 0)
        .map_err(|error| {
            youtube_error(
                YouTubeErrorKind::Whisper,
                format!("select local Whisper backend: {error:#}"),
            )
        })?;
    let backend = selection.chosen;
    polyscribe_core::ilog!("Using {:?} backend.", backend);
    if backend == BackendKind::Cpu && selection.compiled.is_empty() {
        polyscribe_core::ilog!(
            "No GPU backend is compiled into this binary; using the CPU backend. Build with --features gpu-hip for AMD ROCm or --features gpu-cuda for NVIDIA."
        );
    }
    drop(selection);

    let model_name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("whisper-model")
        .to_string();
    let speaker = "speaker1";
    let transcript_progress = TranscriptProgress::start("Transcribing YouTube audio");
    let progress = |percent| {
        transcript_progress.update(percent);
        if !transcript_progress.is_enabled() {
            polyscribe_core::dlog!(1, "local YouTube transcription: {percent}%");
        }
    };
    let mut session = ModelSession::with_model(model_path);
    polyscribe_core::ilog!(
        "Loading Whisper model and transcribing audio (this may take a while)..."
    );
    let result = transcribe_with_decoder_timeout(
        &mut session,
        backend,
        audio_path,
        speaker,
        language_for_whisper(language).as_deref(),
        Some(&progress),
        AUDIO_DECODER_TIMEOUT,
    );
    transcript_progress.finish(if result.is_ok() {
        "Transcript ready"
    } else {
        "Transcription failed"
    });
    let entries = result.map_err(|error| {
        let text = format!("{error:#}");
        let kind = if text.contains("exceeded its timeout") {
            YouTubeErrorKind::AudioDecode
        } else {
            YouTubeErrorKind::Whisper
        };
        youtube_error(kind, format!("local Whisper transcription failed: {text}"))
    })?;
    if entries.is_empty() {
        return Err(youtube_error(
            YouTubeErrorKind::Whisper,
            "local Whisper produced no transcript cues",
        ));
    }
    let mut cues: Vec<CaptionCue> = entries
        .into_iter()
        .map(|entry| CaptionCue {
            start_ms: seconds_to_milliseconds(entry.start),
            end_ms: seconds_to_milliseconds(entry.end).max(seconds_to_milliseconds(entry.start)),
            text: entry.text,
            speaker: (!entry.speaker.is_empty()).then_some(entry.speaker),
            speaker_marker: false,
        })
        .collect();
    cues.sort_by_key(|cue| cue.start_ms);
    crate::youtube_transcript::normalize_and_validate_cues(&mut cues).map_err(|reason| {
        youtube_error(
            YouTubeErrorKind::Whisper,
            format!("local Whisper produced invalid transcript cues: {reason}"),
        )
    })?;
    Ok((
        SourceIdentity::LocalWhisper {
            model_name,
            model_stat: stat,
            backend: format!("{backend:?}").to_ascii_lowercase(),
            language_hint: language.map(ToString::to_string),
        },
        cues,
    ))
}

fn language_for_whisper(language: Option<&str>) -> Option<String> {
    language.map(|language| {
        polyscribe_core::normalize_lang_code(language).unwrap_or_else(|| {
            language
                .split(['-', '_'])
                .next()
                .unwrap_or(language)
                .to_ascii_lowercase()
        })
    })
}

fn seconds_to_milliseconds(seconds: f64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * 1000.0).round().min(u64::MAX as f64) as u64
}

fn find_single_audio_artifact(directory: &Path, max_bytes: u64) -> Result<Option<PathBuf>> {
    let mut result = None;
    for entry in std::fs::read_dir(directory).context("list temporary YouTube audio directory")? {
        let entry = entry.context("read temporary YouTube audio directory entry")?;
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("audio.") {
            continue;
        }
        let metadata = std::fs::symlink_metadata(entry.path())
            .context("inspect downloaded YouTube audio file")?;
        if !metadata.file_type().is_file() || metadata.len() == 0 || metadata.len() > max_bytes {
            return Err(youtube_error(
                YouTubeErrorKind::AudioDownload,
                "downloaded audio is not a regular file within the configured size limit",
            ));
        }
        if result.replace(entry.path()).is_some() {
            return Err(youtube_error(
                YouTubeErrorKind::AudioDownload,
                "yt-dlp produced more than one audio file for a single video",
            ));
        }
    }
    Ok(result)
}

fn run_ytdlp(
    ytdlp: &Path,
    args: &[OsString],
    environment: &[(OsString, OsString)],
    policy: ProcessPolicy,
) -> Result<ProcessOutput> {
    let cancellation = polyscribe_core::stream::CancellationToken::new();
    run_ytdlp_cancellable(ytdlp, args, environment, policy, &cancellation)
}

fn run_ytdlp_cancellable(
    ytdlp: &Path,
    args: &[OsString],
    environment: &[(OsString, OsString)],
    policy: ProcessPolicy,
    cancellation: &polyscribe_core::stream::CancellationToken,
) -> Result<ProcessOutput> {
    youtube_process::run(ytdlp, args, environment, policy, cancellation).map_err(|failure| {
        match failure {
            ProcessFailure::Spawn(error) => youtube_error(
                YouTubeErrorKind::ToolUnavailable,
                format!("failed to start yt-dlp: {error}"),
            ),
            ProcessFailure::Timeout => youtube_error(
                YouTubeErrorKind::DownloaderTimeout,
                format!(
                    "yt-dlp exceeded the {} second time limit",
                    policy.timeout.as_secs()
                ),
            ),
            ProcessFailure::Cancelled => {
                youtube_error(YouTubeErrorKind::Downloader, "yt-dlp was cancelled")
            }
            ProcessFailure::StdoutLimit => youtube_error(
                YouTubeErrorKind::DownloaderOutputTooLarge,
                "yt-dlp stdout exceeded the configured size limit",
            ),
            ProcessFailure::Wait(error) | ProcessFailure::Read(error) => youtube_error(
                YouTubeErrorKind::Downloader,
                format!("failed while running yt-dlp: {error}"),
            ),
            ProcessFailure::ReaderPanicked => youtube_error(
                YouTubeErrorKind::Downloader,
                "yt-dlp output reader stopped unexpectedly",
            ),
        }
    })
}

fn run_ytdlp_with_file_limit(
    ytdlp: &Path,
    args: &[OsString],
    environment: &[(OsString, OsString)],
    file_limit: FileLimitPolicy<'_>,
    policy: ProcessPolicy,
) -> Result<ProcessOutput> {
    let cancellation = polyscribe_core::stream::CancellationToken::new();
    let stop = Arc::new(AtomicBool::new(false));
    let exceeded = Arc::new(AtomicBool::new(false));
    let watcher = watch_download_size(
        file_limit.directory.to_path_buf(),
        file_limit.prefix,
        file_limit.max_bytes,
        cancellation.clone(),
        Arc::clone(&stop),
        Arc::clone(&exceeded),
    );
    let result = run_ytdlp_cancellable(ytdlp, args, environment, policy, &cancellation);
    stop.store(true, Ordering::Relaxed);
    let _ = watcher.join();
    if exceeded.load(Ordering::Relaxed) {
        return Err(youtube_error(
            file_limit.overflow_kind,
            format!(
                "yt-dlp download exceeded the {}-byte file limit",
                file_limit.max_bytes
            ),
        ));
    }
    result
}

fn watch_download_size(
    directory: PathBuf,
    prefix: &'static str,
    max_bytes: u64,
    cancellation: polyscribe_core::stream::CancellationToken,
    stop: Arc<AtomicBool>,
    exceeded: Arc<AtomicBool>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            let over_limit = std::fs::read_dir(&directory).ok().is_some_and(|entries| {
                entries.flatten().any(|entry| {
                    if !entry.file_name().to_string_lossy().starts_with(prefix) {
                        return false;
                    }
                    std::fs::symlink_metadata(entry.path()).is_ok_and(|metadata| {
                        metadata.file_type().is_file() && metadata.len() > max_bytes
                    })
                })
            });
            if over_limit {
                exceeded.store(true, Ordering::Relaxed);
                cancellation.cancel();
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    })
}

fn ensure_success(output: &ProcessOutput, action: &str) -> Result<()> {
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr_tail);
    let kind = classify_downloader_youtube_error(&stderr);
    let message = match kind {
        YouTubeErrorKind::RateLimited => {
            format!("{action} failed: YouTube rate-limited the request")
        }
        YouTubeErrorKind::Forbidden => format!("{action} failed: YouTube denied access"),
        YouTubeErrorKind::Network => {
            format!("{action} failed: network error while contacting YouTube")
        }
        _ => format!("{action} failed"),
    };
    Err(youtube_error(kind, message))
}

fn classify_downloader_youtube_error(stderr: &str) -> YouTubeErrorKind {
    let lowercase = stderr.to_ascii_lowercase();
    if lowercase.contains("http error 429")
        || lowercase.contains("too many requests")
        || lowercase.contains("status code 429")
    {
        YouTubeErrorKind::RateLimited
    } else if lowercase.contains("http error 403")
        || lowercase.contains("forbidden")
        || lowercase.contains("status code 403")
    {
        YouTubeErrorKind::Forbidden
    } else if [
        "timed out",
        "timeout",
        "network is unreachable",
        "temporary failure in name resolution",
        "name or service not known",
        "connection reset",
        "connection refused",
        "unable to download webpage",
        "urlopen error",
        "tls handshake",
    ]
    .iter()
    .any(|needle| lowercase.contains(needle))
    {
        YouTubeErrorKind::Network
    } else {
        YouTubeErrorKind::Downloader
    }
}

fn write_transcript(
    source_url: &str,
    video_id: &str,
    format: &str,
    output_path: Option<&Path>,
    record: &CacheRecord,
) -> Result<()> {
    let request = &record.identity.request;
    let metadata = &record.metadata;
    let source = &record.identity.source;
    let cues = &record.cues;
    let normalization = &record.normalization;
    let (source_name, engine, model, language, caption_language, is_automatic, is_translated) =
        match source {
            SourceIdentity::LocalVoxtral {
                model, language, ..
            } => (
                "local-voxtral",
                "voxtral",
                model.as_str(),
                language.as_deref(),
                None,
                None,
                None,
            ),
            SourceIdentity::LocalWhisper {
                model_name,
                language_hint,
                ..
            } => (
                "local-whisper",
                "whisper",
                model_name.as_str(),
                language_hint.as_deref(),
                None,
                None,
                None,
            ),
        };
    let rendered = match format {
        "txt" => render_txt(cues),
        "srt" => render_srt(cues),
        "json" => serde_json::to_string_pretty(&JsonTranscript {
            source: source_name,
            source_url,
            video_id,
            title: &metadata.title,
            uploader: &metadata.uploader,
            duration_seconds: metadata.duration_seconds,
            requested_language: request.requested_language.as_deref(),
            language,
            engine,
            model,
            caption_language,
            is_automatic,
            is_translated,
            cue_count: cues.len(),
            normalization,
            cues,
        })
        .context("serialize YouTube transcript JSON")?,
        _ => {
            return Err(youtube_error(
                YouTubeErrorKind::UnsupportedFormat,
                "unsupported output format",
            ));
        }
    };
    if let Some(path) = output_path {
        std::fs::write(path, &rendered).with_context(|| {
            format!(
                "failed to write transcript to {}",
                polyscribe_core::ui::terminal_safe_path(path)
            )
        })?;
        eprintln!(
            "Wrote {} cues to {}",
            cues.len(),
            polyscribe_core::ui::terminal_safe_path(path)
        );
    } else {
        let mut stdout = std::io::stdout();
        stdout.write_all(rendered.as_bytes())?;
        if !rendered.ends_with('\n') {
            stdout.write_all(b"\n")?;
        }
        stdout.flush()?;
    }
    match source {
        SourceIdentity::LocalVoxtral {
            model, language, ..
        } => eprintln!(
            "source: local-voxtral, video: {video_id}, model: {model}, language: {}, cues: {}",
            language.as_deref().unwrap_or("unknown"),
            cues.len()
        ),
        SourceIdentity::LocalWhisper { backend, .. } => eprintln!(
            "source: local-whisper, video: {video_id}, backend: {backend}, cues: {}",
            cues.len()
        ),
    }
    Ok(())
}

fn validate_output_path(path: &Path) -> Result<()> {
    let path_text = path.to_string_lossy();
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(anyhow!("output path must not contain '..'"));
    }
    if path_text.chars().any(char::is_control) {
        return Err(anyhow!("output path must not contain control characters"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn voxtral_timeout_invokes_the_whisper_fallback_once() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("slow-voxtral");
        std::fs::write(&executable, "#!/bin/sh\nsleep 10\n").unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        let audio = directory.path().join("audio.wav");
        std::fs::write(&audio, b"offline audio fixture").unwrap();
        let mut whisper_attempts = 0;

        let error = transcribe_with_whisper_fallback(
            None,
            || {
                crate::youtube_voxtral::transcribe_with_timeout(
                    Some(executable.as_os_str().to_os_string()),
                    &audio,
                    None,
                    directory.path(),
                    &polyscribe_core::stream::CancellationToken::new(),
                    Duration::from_millis(80),
                )
            },
            || {
                whisper_attempts += 1;
                Err(youtube_error(
                    YouTubeErrorKind::Whisper,
                    "controlled offline fallback failure",
                ))
            },
        )
        .unwrap_err();

        assert_eq!(whisper_attempts, 1);
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("local Voxtral transcription exceeded its process deadline"));
        assert!(diagnostic.contains("local Whisper fallback failed"));
        assert!(diagnostic.contains("controlled offline fallback failure"));
    }

    #[test]
    fn downloader_statuses_are_classified_without_fallback_aliasing() {
        assert_eq!(
            classify_downloader_youtube_error("ERROR: HTTP Error 429: Too Many Requests"),
            YouTubeErrorKind::RateLimited
        );
        assert_eq!(
            classify_downloader_youtube_error("ERROR: HTTP Error 403: Forbidden"),
            YouTubeErrorKind::Forbidden
        );
        assert_eq!(
            classify_downloader_youtube_error("Unable to download webpage: timed out"),
            YouTubeErrorKind::Network
        );
        assert_eq!(
            classify_downloader_youtube_error("Unsupported URL"),
            YouTubeErrorKind::Downloader
        );
    }

    #[test]
    fn language_validation_never_silently_rewrites_invalid_values() {
        assert_eq!(
            validate_language(Some("DE-de".into())).unwrap(),
            Some("de-de".into())
        );
        assert!(validate_language(Some("en;echo bad".into())).is_err());
        assert!(validate_language(Some("  ".into())).is_err());
    }

    #[test]
    fn downloader_disables_caption_writes_and_retries() {
        let args: Vec<String> = common_args()
            .into_iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert!(args.contains(&"--no-write-subs".to_string()));
        assert!(args.contains(&"--no-write-auto-subs".to_string()));
        assert!(args.contains(&"--retries".to_string()));
        assert!(args.contains(&"--fragment-retries".to_string()));
        assert!(args.contains(&"--extractor-retries".to_string()));
        assert!(!args.iter().any(|arg| {
            matches!(
                arg.as_str(),
                "--write-subs" | "--write-auto-subs" | "--sub-langs"
            )
        }));
    }

    #[test]
    fn cached_whisper_model_is_current_only_when_its_stat_matches() {
        let cached = crate::youtube_cache::ModelStat {
            device: 1,
            inode: 2,
            size: 3,
            modified_seconds: 4,
            modified_nanoseconds: 5,
            changed_seconds: 6,
            changed_nanoseconds: 7,
        };
        assert!(cache_model_stat_matches(&cached, Ok(cached.clone())));
        let mut changed = cached.clone();
        changed.inode += 1;
        assert!(!cache_model_stat_matches(&cached, Ok(changed)));
        assert!(!cache_model_stat_matches(
            &cached,
            Err(anyhow!("stat failed"))
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn moving_a_voxtral_file_away_and_back_changes_its_cache_stat() {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("voxtral");
        let saved = directory.path().join("voxtral.saved");
        std::fs::write(&executable, b"executable").unwrap();
        let initial = crate::youtube_cache::executable_identity(&executable).unwrap();
        std::fs::rename(&executable, &saved).unwrap();
        std::fs::rename(&saved, &executable).unwrap();
        let restored = crate::youtube_cache::executable_identity(&executable).unwrap();
        assert_ne!(initial, restored);
    }

    #[test]
    fn output_path_rejects_parent_components_not_dot_substrings() {
        for path in ["notes..txt", "...", "dir/notes..txt"] {
            assert!(validate_output_path(Path::new(path)).is_ok(), "{path}");
        }
        for path in ["../notes.txt", "dir/../notes.txt", "dir/.."] {
            assert!(validate_output_path(Path::new(path)).is_err(), "{path}");
        }
    }

    #[test]
    fn unsupported_format_diagnostic_does_not_echo_control_characters() {
        let error = run(
            "https://youtu.be/dQw4w9WgXcQ".to_string(),
            None,
            "\u{1b}]52;c;aGVsbG8=\u{7}".to_string(),
            None,
            true,
            false,
            false,
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("unsupported_format"));
        assert!(!error.chars().any(char::is_control));
    }

    #[test]
    fn file_size_monitor_cancels_downloads_before_the_process_exits() {
        let directory = tempfile::tempdir().unwrap();
        let cancellation = polyscribe_core::stream::CancellationToken::new();
        let stop = Arc::new(AtomicBool::new(false));
        let exceeded = Arc::new(AtomicBool::new(false));
        let watcher = watch_download_size(
            directory.path().to_path_buf(),
            "audio.",
            4,
            cancellation.clone(),
            Arc::clone(&stop),
            Arc::clone(&exceeded),
        );
        std::fs::write(directory.path().join("audio.part"), b"12345").unwrap();
        for _ in 0..20 {
            if exceeded.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        stop.store(true, Ordering::Relaxed);
        watcher.join().unwrap();
        assert!(exceeded.load(Ordering::Relaxed));
        assert!(cancellation.is_cancelled());
    }
}

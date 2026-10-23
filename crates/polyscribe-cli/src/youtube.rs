// SPDX-License-Identifier: MIT

//! YouTube caption retrieval and local-audio fallback.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use polyscribe_core::backend::{
    BackendKind, ModelSession, select_backend, transcribe_with_decoder_timeout,
};
use polyscribe_core::find_model_file;
use polyscribe_core::youtube::{
    CaptionCue, CaptionTrack, NormalizationStats, NormalizedTranscript, VideoMetadata,
    YOUTUBE_NORMALIZATION_VERSION, normalize_rolling_cues, parse_vtt, render_srt, render_txt,
    select_caption_track, validate_youtube_url,
};
use serde::Serialize;
use tempfile::TempDir;

use crate::youtube_cache::{
    CACHE_SCHEMA_VERSION, CacheIdentity, CacheRecord, Lookup, RequestKey, RequestMode,
    SourceIdentity, VideoSummary, YoutubeCache, model_sha256, model_stat,
};
use crate::youtube_process::{self, ProcessFailure, ProcessOutput, ProcessPolicy};

const YTDLP_TIMEOUT: Duration = Duration::from_secs(120);
const AUDIO_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const AUDIO_DECODER_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const MAX_METADATA_BYTES: usize = 50 * 1024 * 1024;
const MAX_CAPTION_BYTES: usize = 50 * 1024 * 1024;
const MAX_AUDIO_BYTES: u64 = 250 * 1024 * 1024;
const MAX_STDERR_BYTES: usize = 32 * 1024;
const SELECTION_POLICY_VERSION: u32 = 1;

struct DownloadContext<'a> {
    ytdlp: &'a Path,
    environment: &'a [(OsString, OsString)],
    directory: &'a Path,
    cache: Option<&'a YoutubeCache>,
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
    ToolUnavailable,
    RateLimited,
    Forbidden,
    Network,
    Downloader,
    DownloaderTimeout,
    DownloaderOutputTooLarge,
    MalformedMetadata,
    NoCaptions,
    NoUsableCaptionTrack,
    UnsupportedCaptionFormat,
    RequestedLanguageUnavailable,
    MissingCaptionArtifact,
    MalformedSubtitle,
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
            Self::ToolUnavailable => "tool_unavailable",
            Self::RateLimited => "rate_limited",
            Self::Forbidden => "forbidden",
            Self::Network => "network",
            Self::Downloader => "downloader",
            Self::DownloaderTimeout => "downloader_timeout",
            Self::DownloaderOutputTooLarge => "downloader_output_too_large",
            Self::MalformedMetadata => "malformed_metadata",
            Self::NoCaptions => "no_captions",
            Self::NoUsableCaptionTrack => "no_usable_caption_track",
            Self::UnsupportedCaptionFormat => "unsupported_caption_format",
            Self::RequestedLanguageUnavailable => "requested_language_unavailable",
            Self::MissingCaptionArtifact => "missing_caption_artifact",
            Self::MalformedSubtitle => "malformed_subtitle",
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
    video_id: &'a str,
    title: &'a Option<String>,
    uploader: &'a Option<String>,
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
    prefer_manual: bool,
    refresh: bool,
    captions_only: bool,
    audio_only: bool,
) -> Result<()> {
    let yt_url = validate_youtube_url(&url)
        .map_err(|error| youtube_error(YouTubeErrorKind::InvalidUrl, error.to_string()))?;
    if !matches!(format.as_str(), "txt" | "srt" | "json") {
        return Err(youtube_error(
            YouTubeErrorKind::UnsupportedFormat,
            format!("unsupported format: {format}"),
        ));
    }
    if let Some(path) = &output {
        validate_output_path(path)?;
    }
    let language = validate_language(language)?;
    let mode = if captions_only {
        RequestMode::CaptionsOnly
    } else if audio_only {
        RequestMode::AudioOnly
    } else {
        RequestMode::Default
    };
    let request = RequestKey {
        cache_schema_version: CACHE_SCHEMA_VERSION,
        normalization_version: YOUTUBE_NORMALIZATION_VERSION,
        selection_policy_version: SELECTION_POLICY_VERSION,
        video_id: yt_url.video_id.clone(),
        requested_language: language.clone(),
        prefer_manual,
        mode,
    };

    let cache = open_cache();
    if !refresh && let Some(cache) = &cache {
        match cache.lookup(&request) {
            Ok(Lookup::Hit(record)) if cache_record_is_current(&record) => {
                return write_transcript(
                    &yt_url.video_id,
                    &format,
                    output.as_deref(),
                    &record.metadata,
                    &record.identity.source,
                    &record.cues,
                    &record.normalization,
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

    if audio_only {
        let record = transcribe_audio(
            &download_context,
            &yt_url.canonical_url,
            VideoSummary::default(),
            language.as_deref(),
            &request,
        )?;
        return write_transcript(
            &yt_url.video_id,
            &format,
            output.as_deref(),
            &record.metadata,
            &record.identity.source,
            &record.cues,
            &record.normalization,
        );
    }

    let (metadata, manual_inventory, auto_inventory) = fetch_metadata(
        &ytdlp,
        &environment,
        &yt_url.video_id,
        &yt_url.canonical_url,
    )?;
    let has_caption_formats = manual_inventory.has_formats || auto_inventory.has_formats;
    let manual_tracks = manual_inventory.tracks;
    let auto_tracks = auto_inventory.tracks;
    if has_caption_formats && manual_tracks.is_empty() && auto_tracks.is_empty() {
        return Err(youtube_error(
            YouTubeErrorKind::UnsupportedCaptionFormat,
            "yt-dlp reports captions, but no WebVTT caption formats are available",
        ));
    }
    let track = select_caption_track(
        &manual_tracks,
        &auto_tracks,
        language.as_deref(),
        prefer_manual,
    );

    let caption_result = match track {
        Some(track) => match download_captions(
            &ytdlp,
            &environment,
            scratch.path(),
            &track,
            &yt_url.canonical_url,
        ) {
            Ok(vtt) => Some((track, vtt)),
            Err(error)
                if failure_kind(&error) == Some(YouTubeErrorKind::MissingCaptionArtifact) =>
            {
                if captions_only {
                    return Err(error);
                }
                polyscribe_core::dlog!(
                    1,
                    "The selected caption artifact disappeared; trying local audio transcription."
                );
                None
            }
            Err(error) => return Err(error),
        },
        None if has_caption_formats && language.is_some() => {
            return Err(youtube_error(
                YouTubeErrorKind::RequestedLanguageUnavailable,
                format!(
                    "no caption track is available for the requested language '{}'",
                    language.as_deref().unwrap_or_default()
                ),
            ));
        }
        None if has_caption_formats => {
            return Err(youtube_error(
                YouTubeErrorKind::NoUsableCaptionTrack,
                "caption tracks are present, but none can be selected under the current preferences",
            ));
        }
        None if captions_only => {
            return Err(youtube_error(
                YouTubeErrorKind::NoCaptions,
                "no usable caption tracks are available for this video",
            ));
        }
        None => None,
    };

    if let Some((track, vtt)) = caption_result {
        let parsed = parse_vtt(&vtt).map_err(|error| {
            youtube_error(
                YouTubeErrorKind::MalformedSubtitle,
                format!("failed to parse WebVTT captions: {error}"),
            )
        })?;
        if parsed.cues.is_empty() || parsed.malformed_cues > 0 {
            return Err(youtube_error(
                YouTubeErrorKind::MalformedSubtitle,
                format!(
                    "caption data contained {} valid cues, {} malformed cues, and {} empty cues",
                    parsed.cues.len(),
                    parsed.malformed_cues,
                    parsed.empty_cues
                ),
            ));
        }
        let NormalizedTranscript { cues, stats } = normalize_rolling_cues(&parsed.cues);
        if cues.is_empty() {
            return Err(youtube_error(
                YouTubeErrorKind::MalformedSubtitle,
                "caption normalization produced no transcript cues",
            ));
        }
        let record = CacheRecord {
            schema_version: CACHE_SCHEMA_VERSION,
            normalization_version: YOUTUBE_NORMALIZATION_VERSION,
            identity: CacheIdentity {
                request,
                source: SourceIdentity::YoutubeCaptions {
                    selected_language: track.language,
                    automatic: track.is_automatic,
                    translated: track.is_translated,
                },
            },
            acquired_at: chrono::Utc::now().to_rfc3339(),
            metadata: summarize_metadata(&metadata),
            original_cues: parsed.cues,
            cues,
            normalization: stats,
        };
        store_cache(cache.as_ref(), &record);
        return write_transcript(
            &yt_url.video_id,
            &format,
            output.as_deref(),
            &record.metadata,
            &record.identity.source,
            &record.cues,
            &record.normalization,
        );
    }

    let record = transcribe_audio(
        &download_context,
        &yt_url.canonical_url,
        summarize_metadata(&metadata),
        language.as_deref(),
        &request,
    )?;
    write_transcript(
        &yt_url.video_id,
        &format,
        output.as_deref(),
        &record.metadata,
        &record.identity.source,
        &record.cues,
        &record.normalization,
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
        Ok(path) => model_stat(&path).is_ok_and(|current| &current == cached),
        Err(_) => true,
    }
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
            format!("POLYSCRIBE_YTDLP does not name a file: {}", path.display()),
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
    ["--ignore-config", "--no-cache-dir", "--no-playlist"]
        .into_iter()
        .map(OsString::from)
        .collect()
}

fn fetch_metadata(
    ytdlp: &Path,
    environment: &[(OsString, OsString)],
    expected_video_id: &str,
    url: &str,
) -> Result<(VideoMetadata, CaptionInventory, CaptionInventory)> {
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
            format!("yt-dlp returned video id '{id}' for requested video '{expected_video_id}'"),
        ));
    }
    let manual_inventory = parse_tracks(value.get("subtitles"), false, "subtitles")?;
    let auto_inventory = parse_tracks(value.get("automatic_captions"), true, "automatic_captions")?;
    let metadata = VideoMetadata {
        id: id.to_string(),
        title: non_empty_string(value.get("title")),
        uploader: non_empty_string(value.get("uploader")),
        duration: value.get("duration").and_then(serde_json::Value::as_f64),
        subtitles: value
            .get("subtitles")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({})),
        automatic_captions: value
            .get("automatic_captions")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({})),
    };
    Ok((metadata, manual_inventory, auto_inventory))
}

fn non_empty_string(value: Option<&serde_json::Value>) -> Option<String> {
    value
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty() && *value != "NA")
        .map(ToString::to_string)
}

struct CaptionInventory {
    tracks: Vec<CaptionTrack>,
    has_formats: bool,
}

fn parse_tracks(
    json: Option<&serde_json::Value>,
    is_automatic: bool,
    field: &str,
) -> Result<CaptionInventory> {
    let Some(json) = json else {
        return Ok(CaptionInventory {
            tracks: Vec::new(),
            has_formats: false,
        });
    };
    let Some(object) = json.as_object() else {
        return Err(youtube_error(
            YouTubeErrorKind::MalformedMetadata,
            format!("yt-dlp metadata field '{field}' must be an object"),
        ));
    };
    let mut inventory = CaptionInventory {
        tracks: Vec::new(),
        has_formats: false,
    };
    for (language, formats) in object {
        let Some(formats) = formats.as_array() else {
            return Err(youtube_error(
                YouTubeErrorKind::MalformedMetadata,
                format!("caption track '{language}' in metadata field '{field}' must be an array"),
            ));
        };
        for format in formats {
            let Some(format_object) = format.as_object() else {
                return Err(youtube_error(
                    YouTubeErrorKind::MalformedMetadata,
                    format!(
                        "caption format for '{language}' in metadata field '{field}' must be an object"
                    ),
                ));
            };
            inventory.has_formats = true;
            let Some(ext) = format_object.get("ext").and_then(serde_json::Value::as_str) else {
                return Err(youtube_error(
                    YouTubeErrorKind::MalformedMetadata,
                    format!(
                        "caption format for '{language}' in metadata field '{field}' has no string extension"
                    ),
                ));
            };
            if !ext.eq_ignore_ascii_case("vtt") {
                continue;
            }
            let url = format_object
                .get("url")
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string);
            let translated = url
                .as_ref()
                .is_some_and(|url| url.split(['?', '&']).any(|part| part.starts_with("tlang=")));
            inventory.tracks.push(CaptionTrack {
                language: language.clone(),
                name: format_object
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .map(ToString::to_string),
                url,
                ext: Some(ext.to_string()),
                is_automatic,
                is_translated: translated,
            });
        }
    }
    Ok(inventory)
}

fn download_captions(
    ytdlp: &Path,
    environment: &[(OsString, OsString)],
    directory: &Path,
    track: &CaptionTrack,
    url: &str,
) -> Result<String> {
    let template = directory.join("caption");
    let mut args = common_args();
    args.extend([
        "--skip-download".into(),
        "--sub-format".into(),
        "vtt".into(),
        "--sub-langs".into(),
        track.language.clone().into(),
        "--output".into(),
        template.as_os_str().to_os_string(),
    ]);
    args.push(if track.is_automatic {
        "--write-auto-subs".into()
    } else {
        "--write-subs".into()
    });
    args.push(url.into());
    let output = run_ytdlp_with_file_limit(
        ytdlp,
        &args,
        environment,
        FileLimitPolicy {
            directory,
            prefix: "caption",
            max_bytes: MAX_CAPTION_BYTES as u64,
            overflow_kind: YouTubeErrorKind::DownloaderOutputTooLarge,
        },
        ProcessPolicy {
            timeout: YTDLP_TIMEOUT,
            stdout_limit: 1024 * 1024,
            stderr_limit: MAX_STDERR_BYTES,
        },
    )?;
    ensure_success(&output, "yt-dlp caption download")?;
    let path = find_single_artifact(directory, "caption", "vtt", MAX_CAPTION_BYTES as u64)?
        .ok_or_else(|| {
            youtube_error(
                YouTubeErrorKind::MissingCaptionArtifact,
                "yt-dlp did not produce the selected WebVTT caption file",
            )
        })?;
    std::fs::read_to_string(&path)
        .with_context(|| format!("read downloaded WebVTT file {}", path.display()))
        .map_err(|error| {
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == std::io::ErrorKind::InvalidData)
            {
                youtube_error(
                    YouTubeErrorKind::MalformedSubtitle,
                    "downloaded WebVTT captions are not valid UTF-8",
                )
            } else if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound)
            {
                youtube_error(
                    YouTubeErrorKind::MissingCaptionArtifact,
                    "the selected caption artifact disappeared before it could be read",
                )
            } else {
                youtube_error(
                    YouTubeErrorKind::Downloader,
                    format!("failed to read downloaded WebVTT file: {error:#}"),
                )
            }
        })
}

fn transcribe_audio(
    context: &DownloadContext<'_>,
    url: &str,
    metadata: VideoSummary,
    language: Option<&str>,
    request: &RequestKey,
) -> Result<CacheRecord> {
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
    drop(selection);

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
    .map_err(|error| {
        if failure_kind(&error) == Some(YouTubeErrorKind::Network)
            || failure_kind(&error) == Some(YouTubeErrorKind::RateLimited)
            || failure_kind(&error) == Some(YouTubeErrorKind::Forbidden)
        {
            error
        } else {
            youtube_error(
                YouTubeErrorKind::AudioDownload,
                format!("could not download YouTube audio: {error:#}"),
            )
        }
    })?;
    ensure_success(&output, "yt-dlp audio download").map_err(|error| {
        if failure_kind(&error).is_some_and(|kind| {
            matches!(
                kind,
                YouTubeErrorKind::RateLimited
                    | YouTubeErrorKind::Forbidden
                    | YouTubeErrorKind::Network
            )
        }) {
            error
        } else {
            youtube_error(
                YouTubeErrorKind::AudioDownload,
                format!("could not download YouTube audio: {error:#}"),
            )
        }
    })?;
    let audio_path =
        find_single_audio_artifact(context.directory, MAX_AUDIO_BYTES)?.ok_or_else(|| {
            youtube_error(
                YouTubeErrorKind::AudioDownload,
                "yt-dlp produced no audio file",
            )
        })?;

    let model_hash = if context.cache.is_some() {
        match model_sha256(&model_path) {
            Ok(hash) => Some(hash),
            Err(error) => {
                polyscribe_core::dlog!(
                    1,
                    "Could not fingerprint local Whisper model for caching: {error:#}"
                );
                None
            }
        }
    } else {
        None
    };
    let speaker = "speaker1";
    let progress = |percent| polyscribe_core::dlog!(1, "local YouTube transcription: {percent}%");
    let mut session = ModelSession::with_model(model_path);
    let entries = transcribe_with_decoder_timeout(
        &mut session,
        backend,
        &audio_path,
        speaker,
        language_for_whisper(language).as_deref(),
        Some(&progress),
        AUDIO_DECODER_TIMEOUT,
    )
    .map_err(|error| {
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
    let model_hash = model_hash.unwrap_or_default();
    let record = CacheRecord {
        schema_version: CACHE_SCHEMA_VERSION,
        normalization_version: YOUTUBE_NORMALIZATION_VERSION,
        identity: CacheIdentity {
            request: request.clone(),
            source: SourceIdentity::LocalWhisper {
                model_sha256: model_hash,
                model_stat: stat,
                backend: format!("{backend:?}").to_ascii_lowercase(),
                language_hint: language.map(ToString::to_string),
            },
        },
        acquired_at: chrono::Utc::now().to_rfc3339(),
        metadata,
        original_cues: cues.clone(),
        cues,
        normalization: NormalizationStats::default(),
    };
    store_cache(context.cache, &record);
    Ok(record)
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

fn find_single_artifact(
    directory: &Path,
    prefix: &str,
    extension: &str,
    max_bytes: u64,
) -> Result<Option<PathBuf>> {
    let mut result = None;
    for entry in
        std::fs::read_dir(directory).context("list temporary YouTube download directory")?
    {
        let entry = entry.context("read temporary YouTube download directory entry")?;
        let path = entry.path();
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with(prefix)
            || path.extension().and_then(|value| value.to_str()) != Some(extension)
        {
            continue;
        }
        let metadata = std::fs::symlink_metadata(&path).context("inspect downloaded artifact")?;
        if !metadata.file_type().is_file() || metadata.len() == 0 || metadata.len() > max_bytes {
            return Err(youtube_error(
                YouTubeErrorKind::Downloader,
                "downloaded subtitle artifact is not a regular file within the configured size limit",
            ));
        }
        if result.replace(path).is_some() {
            return Err(youtube_error(
                YouTubeErrorKind::Downloader,
                "yt-dlp produced multiple subtitle files for one selected track",
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
    video_id: &str,
    format: &str,
    output_path: Option<&Path>,
    metadata: &VideoSummary,
    source: &SourceIdentity,
    cues: &[CaptionCue],
    normalization: &NormalizationStats,
) -> Result<()> {
    let (source_name, caption_language, is_automatic, is_translated) = match source {
        SourceIdentity::YoutubeCaptions {
            selected_language,
            automatic,
            translated,
        } => (
            "youtube-captions",
            Some(selected_language.as_str()),
            Some(*automatic),
            Some(*translated),
        ),
        SourceIdentity::LocalWhisper { .. } => ("local-whisper", None, None, None),
    };
    let rendered = match format {
        "txt" => render_txt(cues),
        "srt" => render_srt(cues),
        "json" => serde_json::to_string_pretty(&JsonTranscript {
            source: source_name,
            video_id,
            title: &metadata.title,
            uploader: &metadata.uploader,
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
        std::fs::write(path, &rendered)
            .with_context(|| format!("failed to write transcript to {}", path.display()))?;
        eprintln!("Wrote {} cues to {}", cues.len(), path.display());
    } else {
        let mut stdout = std::io::stdout();
        stdout.write_all(rendered.as_bytes())?;
        if !rendered.ends_with('\n') {
            stdout.write_all(b"\n")?;
        }
        stdout.flush()?;
    }
    match source {
        SourceIdentity::YoutubeCaptions {
            selected_language,
            automatic,
            ..
        } => eprintln!(
            "source: youtube-captions, video: {video_id}, language: {selected_language}, track: {}, cues: {}",
            if *automatic { "automatic" } else { "manual" },
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
    if path_text.contains("..") {
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
    fn metadata_track_parsing_only_keeps_vtt_and_marks_translation() {
        let inventory = parse_tracks(
            Some(&serde_json::json!({
                "en": [
                    {"ext":"vtt", "url":"https://example.test/sub.vtt"},
                    {"ext":"srv3", "url":"https://example.test/sub.xml"}
                ],
                "de": [{"ext":"vtt", "url":"https://example.test/sub.vtt?tlang=de"}]
            })),
            true,
            "automatic_captions",
        )
        .unwrap();
        assert!(inventory.has_formats);
        assert_eq!(inventory.tracks.len(), 2);
        assert!(
            inventory
                .tracks
                .iter()
                .find(|track| track.language == "en")
                .is_some_and(|track| !track.is_translated)
        );
        assert!(
            inventory
                .tracks
                .iter()
                .find(|track| track.language == "de")
                .is_some_and(|track| track.is_translated)
        );
        assert!(inventory.tracks.iter().all(|track| track.is_automatic));
    }

    #[test]
    fn malformed_caption_inventory_is_not_treated_as_empty() {
        assert!(parse_tracks(Some(&serde_json::Value::Null), false, "subtitles").is_err());
        assert!(parse_tracks(Some(&serde_json::json!({"en": null})), false, "subtitles").is_err());
        assert!(
            parse_tracks(
                Some(&serde_json::json!({"en": [{"url": "https://example.test/sub"}]})),
                false,
                "subtitles"
            )
            .is_err()
        );
        assert!(
            parse_tracks(None, false, "subtitles")
                .unwrap()
                .tracks
                .is_empty()
        );
    }

    #[test]
    fn unsupported_formats_are_distinguishable_from_no_captions() {
        let inventory = parse_tracks(
            Some(&serde_json::json!({"en": [{"ext": "srv3"}]})),
            false,
            "subtitles",
        )
        .unwrap();
        assert!(inventory.has_formats);
        assert!(inventory.tracks.is_empty());
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

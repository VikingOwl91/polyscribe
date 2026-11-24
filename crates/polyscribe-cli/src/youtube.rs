// SPDX-License-Identifier: MIT

//! Cache-first YouTube audio acquisition and local transcription.

use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use polyscribe_core::backend::{
    BackendKind, ModelSession, TranscribeBackend, select_backend, transcribe_with_decoder_timeout,
};
use polyscribe_core::find_model_file;
use polyscribe_core::ui::progress::TranscriptProgress;
use polyscribe_core::voxtral_cloud::{CloudVoxtralBackend, DEFAULT_VOXTRAL_CLOUD_MODEL};
use polyscribe_core::youtube::{
    CaptionCue, NormalizationStats, TranscriptFrontmatter, VideoMetadata,
    YOUTUBE_NORMALIZATION_VERSION, build_frontmatter, canonical_file_name, iso_upload_date,
    render_srt, render_txt, validate_youtube_url,
};
use serde::Serialize;

use crate::cli::Engine;
use crate::transcribe::{ResolvedEngine, resolve_engine};
use crate::youtube_cache::{
    CACHE_SCHEMA_VERSION, CacheIdentity, CacheRecord, ExecutableIdentity, Lookup, ModelStat,
    RequestKey, SourceIdentity, VideoSummary, YoutubeCache, file_stat, model_stat,
};
use crate::youtube_process::{self, ProcessFailure, ProcessOutput, ProcessPolicy};

const YTDLP_TIMEOUT: Duration = Duration::from_secs(120);
const AUDIO_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const AUDIO_DECODER_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const MAX_METADATA_BYTES: usize = 50 * 1024 * 1024;
const MAX_AUDIO_BYTES: u64 = 250 * 1024 * 1024;
const MAX_STDERR_BYTES: usize = 32 * 1024;
const MAX_COOKIE_BYTES: u64 = 4 * 1024 * 1024;
const AUDIO_POLICY_VERSION: u32 = 2;

struct DownloadContext<'a> {
    ytdlp: &'a Path,
    environment: &'a [(OsString, OsString)],
    directory: &'a Path,
    cache: Option<&'a YoutubeCache>,
    cookies: Option<&'a Path>,
}

// Contains credentials: never derive Debug or serialize this buffer.
struct PreparedCookies {
    bytes: Vec<u8>,
    identity: ModelStat,
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
    AuthenticationRequired,
    InvalidCookies,
    Network,
    Downloader,
    DownloaderTimeout,
    DownloaderOutputTooLarge,
    MalformedMetadata,
    AudioDownload,
    AudioDecode,
    Whisper,
    Cloud,
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
            Self::AuthenticationRequired => "authentication_required",
            Self::InvalidCookies => "invalid_cookies",
            Self::Network => "network",
            Self::Downloader => "downloader",
            Self::DownloaderTimeout => "downloader_timeout",
            Self::DownloaderOutputTooLarge => "downloader_output_too_large",
            Self::MalformedMetadata => "malformed_metadata",
            Self::AudioDownload => "audio_download",
            Self::AudioDecode => "audio_decode",
            Self::Whisper => "whisper",
            Self::Cloud => "cloud",
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
    channel: &'a Option<String>,
    channel_url: &'a Option<String>,
    upload_date: &'a Option<String>,
    duration_seconds: Option<f64>,
    transcribed_at: &'a str,
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
    cookies: Option<PathBuf>,
    no_cookies: bool,
    captions_only: bool,
    _audio_only: bool,
    no_frontmatter: bool,
    engine: Engine,
    voxtral_model: Option<String>,
    yes: bool,
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
    if yes {
        polyscribe_core::voxtral_cloud::set_cloud_consent(true);
    }
    let resolved_engine = resolve_engine(engine);
    if matches!(resolved_engine, ResolvedEngine::AutoCloud) {
        polyscribe_core::voxtral_cloud::warn_auto_cloud_once();
    }
    polyscribe_core::dlog!(1, "Resolved engine: {resolved_engine:?}");
    let cookies = select_cookies(cookies.as_deref(), no_cookies)?
        .map(prepare_cookies)
        .transpose()?;
    let voxtral_config = VoxtralConfigSnapshot::from_environment();
    let request = RequestKey {
        cache_schema_version: CACHE_SCHEMA_VERSION,
        normalization_version: YOUTUBE_NORMALIZATION_VERSION,
        selection_policy_version: AUDIO_POLICY_VERSION,
        video_id: yt_url.video_id.clone(),
        requested_language: language.clone(),
        voxtral_executable: voxtral_config.path_bytes.clone(),
        voxtral_executable_identity: voxtral_config.executable_stat.clone(),
        cookie_file_identity: cookies.as_ref().map(|cookies| cookies.identity.clone()),
        engine,
        voxtral_model: voxtral_model.clone(),
    };

    // Unix file identity separates authenticated requests without retaining secrets.
    let cache = if cfg!(unix) || cookies.is_none() {
        open_cache()
    } else {
        None
    };
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
                polyscribe_core::ilog!("Using cached YouTube transcript.");
                return write_transcript(
                    &source_url,
                    &yt_url.video_id,
                    &format,
                    output.as_deref(),
                    &record,
                    no_frontmatter,
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
    let mut scratch_builder = tempfile::Builder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        scratch_builder.permissions(std::fs::Permissions::from_mode(0o700));
    }
    let scratch = scratch_builder
        .tempdir()
        .context("create private temporary directory for yt-dlp")?;
    let cookie_snapshot = cookies
        .map(|cookies| snapshot_cookies(&cookies, scratch.path()))
        .transpose()?;
    let environment = youtube_process::safe_environment(scratch.path());
    let download_context = DownloadContext {
        ytdlp: &ytdlp,
        environment: &environment,
        directory: scratch.path(),
        cache: cache.as_ref(),
        cookies: cookie_snapshot.as_deref(),
    };
    polyscribe_core::ilog!("Fetching YouTube metadata...");
    let metadata = fetch_metadata(
        &ytdlp,
        &environment,
        &yt_url.video_id,
        &yt_url.canonical_url,
        cookie_snapshot.as_deref(),
    )?;
    let record = transcribe_audio(
        &download_context,
        &yt_url.canonical_url,
        summarize_metadata(&metadata),
        language.as_deref(),
        &request,
        &voxtral_config,
        resolved_engine,
        voxtral_model,
    )?;
    write_transcript(
        &source_url,
        &yt_url.video_id,
        &format,
        output.as_deref(),
        &record,
        no_frontmatter,
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
        channel: metadata.channel.clone(),
        channel_url: metadata.channel_url.clone(),
        upload_date: metadata.upload_date.clone(),
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

fn common_args(cookies: Option<&Path>) -> Vec<OsString> {
    let mut args: Vec<OsString> = [
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
    .collect();
    if let Some(path) = cookies {
        args.extend(["--cookies".into(), path.as_os_str().to_os_string()]);
    }
    args
}

fn select_cookies(explicit: Option<&Path>, disabled: bool) -> Result<Option<std::fs::File>> {
    if disabled {
        return Ok(None);
    }
    if let Some(path) = explicit {
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NONBLOCK);
        }
        return options
            .open(path)
            .map(Some)
            .map_err(|_| invalid_cookie_file());
    }
    let Some(dirs) = polyscribe_core::config::ConfigService::project_dirs() else {
        return Ok(None);
    };
    let cookies = open_default_cookies(dirs.config_dir())?;
    if cookies.is_some() {
        polyscribe_core::ilog!("Using stored YouTube cookies.");
    }
    Ok(cookies)
}

fn invalid_cookie_file() -> anyhow::Error {
    youtube_error(
        YouTubeErrorKind::InvalidCookies,
        "cookies.txt must be a readable, nonempty regular cookie file of at most 4 MiB in Netscape format",
    )
}

fn unsafe_default_cookies() -> anyhow::Error {
    youtube_error(
        YouTubeErrorKind::InvalidCookies,
        "cannot safely use stored youtube/cookies.txt: use an owned directory with mode 700 and an owned regular file with mode 600 (400 is also accepted); symlinks and writable parent directories are not allowed. Use --cookies FILE to select another file or --no-cookies to run anonymously",
    )
}

#[cfg(unix)]
fn open_default_cookies(config_dir: &Path) -> Result<Option<std::fs::File>> {
    use std::ffi::{CStr, CString};
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::MetadataExt;

    fn open_at(
        directory: &std::fs::File,
        name: &CStr,
        flags: i32,
    ) -> std::io::Result<std::fs::File> {
        let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
        if fd < 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(unsafe { std::fs::File::from_raw_fd(fd) })
        }
    }

    if !config_dir.is_absolute()
        || config_dir
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
    {
        return Err(unsafe_default_cookies());
    }
    let flags = libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK;
    let mut directory = std::fs::File::open("/").map_err(|_| unsafe_default_cookies())?;
    let components: Vec<_> = config_dir
        .components()
        .chain([Component::Normal(std::ffi::OsStr::new("youtube"))])
        .filter_map(|part| match part {
            Component::Normal(name) => Some(name),
            _ => None,
        })
        .collect();
    let effective_uid = unsafe { libc::geteuid() };
    let mut directories = Vec::new();
    for (index, name) in components.iter().enumerate() {
        let name = CString::new(name.as_bytes()).map_err(|_| unsafe_default_cookies())?;
        directory = match open_at(&directory, &name, flags | libc::O_DIRECTORY) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) => return Err(unsafe_default_cookies()),
        };
        let stat = directory.metadata().map_err(|_| unsafe_default_cookies())?;
        directories.push((index, stat));
    }
    let file = match open_at(&directory, c"cookies.txt", flags) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(unsafe_default_cookies()),
    };
    // A missing file means anonymous operation. Validate all held descriptors
    // before reading any credentials, without reopening a pathname.
    for (index, stat) in directories {
        let is_private_directory = index + 1 == components.len();
        let is_app_directory = index + 2 == components.len();
        let safe = if is_private_directory {
            stat.uid() == effective_uid && stat.mode() & 0o077 == 0
        } else if is_app_directory {
            stat.uid() == effective_uid && stat.mode() & 0o022 == 0
        } else {
            (stat.uid() == effective_uid || stat.uid() == 0)
                && (stat.mode() & 0o022 == 0
                    || (stat.uid() == 0 && stat.mode() & libc::S_ISVTX != 0))
        };
        if !safe {
            return Err(unsafe_default_cookies());
        }
    }
    let stat = file.metadata().map_err(|_| unsafe_default_cookies())?;
    if !stat.is_file() || stat.uid() != effective_uid || stat.mode() & 0o077 != 0 {
        return Err(unsafe_default_cookies());
    }
    Ok(Some(file))
}

#[cfg(not(unix))]
fn open_default_cookies(_config_dir: &Path) -> Result<Option<std::fs::File>> {
    // Automatic credential discovery requires verified Unix ownership/permissions.
    // Explicit --cookies remains available on platforms without these checks.
    Ok(None)
}

fn prepare_cookies(source: std::fs::File) -> Result<PreparedCookies> {
    let invalid = invalid_cookie_file;
    let metadata = source.metadata().map_err(|_| invalid())?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_COOKIE_BYTES {
        return Err(invalid());
    }
    let identity = file_stat(&source).map_err(|_| invalid())?;
    let mut bytes = Vec::new();
    (&source)
        .take(MAX_COOKIE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| invalid())?;
    if bytes.is_empty() || bytes.len() as u64 > MAX_COOKIE_BYTES {
        return Err(invalid());
    }
    if identity != file_stat(&source).map_err(|_| invalid())? || bytes.len() as u64 != identity.size
    {
        return Err(youtube_error(
            YouTubeErrorKind::InvalidCookies,
            "cookie file changed while being read; retry after the export is complete",
        ));
    }
    Ok(PreparedCookies { bytes, identity })
}

fn snapshot_cookies(cookies: &PreparedCookies, directory: &Path) -> Result<PathBuf> {
    let invalid = invalid_cookie_file;
    let path = directory.join("cookies.txt");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut destination = options.open(&path).map_err(|_| invalid())?;
    destination
        .write_all(&cookies.bytes)
        .map_err(|_| invalid())?;
    Ok(path)
}

fn fetch_metadata(
    ytdlp: &Path,
    environment: &[(OsString, OsString)],
    expected_video_id: &str,
    url: &str,
    cookies: Option<&Path>,
) -> Result<VideoMetadata> {
    let mut args = common_args(cookies);
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
    let uploader = non_empty_string(value.get("uploader"));
    // yt-dlp's `channel` falls back to `uploader` when absent.
    let channel = non_empty_string(value.get("channel")).or_else(|| uploader.clone());
    let channel_url = non_empty_string(value.get("channel_url"));
    let upload_date = value
        .get("upload_date")
        .and_then(serde_json::Value::as_str)
        .and_then(iso_upload_date);
    Ok(VideoMetadata {
        id: id.to_string(),
        title: non_empty_string(value.get("title")),
        uploader,
        channel,
        channel_url,
        upload_date,
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
    let mut args = common_args(context.cookies);
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
                | YouTubeErrorKind::AuthenticationRequired
                | YouTubeErrorKind::InvalidCookies
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

#[allow(clippy::too_many_arguments)]
fn transcribe_audio(
    context: &DownloadContext<'_>,
    url: &str,
    metadata: VideoSummary,
    language: Option<&str>,
    request: &RequestKey,
    voxtral_config: &VoxtralConfigSnapshot,
    engine: ResolvedEngine,
    voxtral_model: Option<String>,
) -> Result<CacheRecord> {
    let audio_path = download_audio(context, url)?;
    polyscribe_core::ilog!("YouTube audio downloaded; starting transcription...");
    let (source, cues) = transcribe_with_selected_engine(
        engine,
        || transcribe_cloud(&audio_path, language, voxtral_model.clone()),
        || {
            transcribe_with_whisper_fallback(
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
            )
        },
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

/// Run the engine selected for this run. `cloud` transcribes the downloaded
/// audio with the Mistral cloud engine; `local` is the unchanged local path
/// (the optional `POLYSCRIBE_VOXTRAL` wrapper, then local Whisper).
///
/// - `whisper`: only the local path runs, exactly as before.
/// - `voxtral-cloud`: only the cloud runs; a cloud error aborts with
///   `YouTubeErrorKind::Cloud` and there is no fallback.
/// - `auto` (resolved to the cloud): the cloud runs first; on error a visible
///   warning is printed and the local path runs once. The returned source
///   identity is the one of the engine that actually produced the result, so
///   a fallback result is cached under the engine that really ran.
fn transcribe_with_selected_engine(
    engine: ResolvedEngine,
    cloud: impl FnOnce() -> Result<(SourceIdentity, Vec<CaptionCue>)>,
    local: impl FnOnce() -> Result<(SourceIdentity, Vec<CaptionCue>)>,
) -> Result<(SourceIdentity, Vec<CaptionCue>)> {
    match engine {
        ResolvedEngine::Whisper => local(),
        ResolvedEngine::VoxtralCloud => cloud().map_err(|error| match failure_kind(&error) {
            Some(_) => error,
            None => youtube_error(
                YouTubeErrorKind::Cloud,
                format!("Mistral cloud transcription failed: {error:#}"),
            ),
        }),
        ResolvedEngine::AutoCloud => match cloud() {
            Ok(result) => Ok(result),
            Err(cloud_error) => {
                // The cloud error text is already scrubbed of the API key by
                // the cloud backend, so it can be shown.
                polyscribe_core::ui::warn(format!(
                    "The Mistral cloud transcription failed ({cloud_error:#}); \
                     falling back to the local engine."
                ));
                local().map_err(|local_error| {
                    let kind = failure_kind(&local_error).unwrap_or(YouTubeErrorKind::Whisper);
                    youtube_error(
                        kind,
                        format!(
                            "Mistral cloud transcription failed ({cloud_error:#}); \
                             local fallback failed: {local_error:#}"
                        ),
                    )
                })
            }
        },
    }
}

/// Transcribe the downloaded audio file with the Mistral cloud engine
/// (`CloudVoxtralBackend`). Only the audio file is uploaded; no cookies, URL,
/// or video metadata are sent. The API key never appears in any error
/// message (the backend scrubs it). The cloud path attaches no speaker label.
fn transcribe_cloud(
    audio_path: &Path,
    language: Option<&str>,
    voxtral_model: Option<String>,
) -> Result<(SourceIdentity, Vec<CaptionCue>)> {
    let model = voxtral_model.unwrap_or_else(|| DEFAULT_VOXTRAL_CLOUD_MODEL.to_string());
    polyscribe_core::ilog!("Transcribing with the Mistral cloud engine (model {model})...");
    let mut backend = CloudVoxtralBackend::new(model.clone()).map_err(|error| {
        youtube_error(
            YouTubeErrorKind::Cloud,
            format!("could not set up the Mistral cloud engine: {error:#}"),
        )
    })?;
    let transcript_progress =
        TranscriptProgress::start("Transcribing YouTube audio (Mistral cloud)");
    let progress = |percent: i32| {
        transcript_progress.update(percent);
        if !transcript_progress.is_enabled() {
            polyscribe_core::dlog!(1, "cloud YouTube transcription: {percent}%");
        }
    };
    let result = backend.transcribe(
        audio_path,
        "",
        language_for_whisper(language).as_deref(),
        None,
        Some(&progress),
    );
    transcript_progress.finish(if result.is_ok() {
        "Transcript ready"
    } else {
        "Transcription failed"
    });
    let entries = result.map_err(|error| {
        youtube_error(
            YouTubeErrorKind::Cloud,
            format!("Mistral cloud transcription failed: {error:#}"),
        )
    })?;
    if entries.is_empty() {
        return Err(youtube_error(
            YouTubeErrorKind::Cloud,
            "Mistral cloud produced no transcript cues",
        ));
    }
    let cues = entries_to_cues(entries).map_err(|reason| {
        youtube_error(
            YouTubeErrorKind::Cloud,
            format!("Mistral cloud produced invalid transcript cues: {reason}"),
        )
    })?;
    Ok((
        SourceIdentity::CloudVoxtral {
            model,
            language_hint: language.map(ToString::to_string),
        },
        cues,
    ))
}

/// Convert backend output entries to validated caption cues, sorted by start
/// time. An empty speaker label becomes `None` (no speaker). Shared by the
/// local Whisper path and the cloud path.
fn entries_to_cues(
    entries: Vec<polyscribe_core::OutputEntry>,
) -> std::result::Result<Vec<CaptionCue>, &'static str> {
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
    crate::youtube_transcript::normalize_and_validate_cues(&mut cues)?;
    Ok(cues)
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
    // No speaker label: there is no diarization for a single unknown speaker,
    // and a fixed label would be misleading in the txt output. The empty
    // label becomes `speaker: None` in the cues.
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
        "",
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
    let cues = entries_to_cues(entries).map_err(|reason| {
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
        YouTubeErrorKind::AuthenticationRequired => format!(
            "{action} failed: YouTube requires bot verification; install or replace youtube/cookies.txt in the PolyScribe config directory, or provide an exported YouTube cookie file with --cookies FILE. --refresh only bypasses the transcript cache and cannot resolve this challenge"
        ),
        YouTubeErrorKind::InvalidCookies => {
            format!("{action} failed: --cookies requires a cookie file in Netscape format")
        }
        YouTubeErrorKind::Network => {
            format!("{action} failed: network error while contacting YouTube")
        }
        _ => format!("{action} failed"),
    };
    Err(youtube_error(kind, message))
}

fn classify_downloader_youtube_error(stderr: &str) -> YouTubeErrorKind {
    let lowercase = stderr.to_ascii_lowercase();
    if lowercase.contains("does not look like a netscape format cookies file")
        || lowercase.contains("cookies file must be netscape formatted")
    {
        YouTubeErrorKind::InvalidCookies
    } else if lowercase.contains("sign in to confirm you’re not a bot")
        || lowercase.contains("sign in to confirm you're not a bot")
    {
        YouTubeErrorKind::AuthenticationRequired
    } else if lowercase.contains("http error 429")
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
    no_frontmatter: bool,
) -> Result<()> {
    let request = &record.identity.request;
    let metadata = &record.metadata;
    let source = &record.identity.source;
    let cues = &record.cues;
    let normalization = &record.normalization;
    let (
        source_name,
        engine,
        model,
        language,
        caption_language,
        is_automatic,
        is_translated,
        backend,
    ) = match source {
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
            None,
        ),
        SourceIdentity::LocalWhisper {
            model_name,
            language_hint,
            backend,
            ..
        } => (
            "local-whisper",
            "whisper",
            model_name.as_str(),
            language_hint.as_deref(),
            None,
            None,
            None,
            Some(backend.as_str()),
        ),
        SourceIdentity::CloudVoxtral {
            model,
            language_hint,
        } => (
            "cloud-voxtral",
            "voxtral-cloud",
            model.as_str(),
            // The language is the hint that was sent, never a detected or
            // guessed value; without a hint it stays null.
            language_hint.as_deref(),
            None,
            None,
            None,
            None,
        ),
    };
    let rendered = match format {
        // srt cannot carry metadata; it stays plain cues in every mode.
        "txt" if !no_frontmatter => {
            let head = build_frontmatter(&TranscriptFrontmatter {
                title: metadata.title.as_deref(),
                video_id,
                channel: metadata.channel.as_deref(),
                channel_url: metadata.channel_url.as_deref(),
                upload_date: metadata.upload_date.as_deref(),
                duration_seconds: metadata.duration_seconds,
                language,
                transcribed_at: &record.acquired_at,
                engine,
                model,
                backend,
            });
            format!("{head}{}", render_txt(cues))
        }
        "txt" => render_txt(cues),
        "srt" => render_srt(cues),
        "json" => serde_json::to_string_pretty(&JsonTranscript {
            source: source_name,
            source_url,
            video_id,
            title: &metadata.title,
            uploader: &metadata.uploader,
            channel: &metadata.channel,
            channel_url: &metadata.channel_url,
            upload_date: &metadata.upload_date,
            duration_seconds: metadata.duration_seconds,
            transcribed_at: &record.acquired_at,
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
        // An existing directory gets the canonical file name built from the
        // cached metadata; every component is fs-safe (no '..', '/', or
        // control characters).
        let target = if path.is_dir() {
            path.join(canonical_file_name(
                metadata.upload_date.as_deref(),
                metadata.channel.as_deref(),
                video_id,
                metadata.title.as_deref(),
                format,
            ))
        } else {
            path.to_path_buf()
        };
        std::fs::write(&target, &rendered).with_context(|| {
            format!(
                "failed to write transcript to {}",
                polyscribe_core::ui::terminal_safe_path(&target)
            )
        })?;
        eprintln!(
            "Wrote {} cues to {}",
            cues.len(),
            polyscribe_core::ui::terminal_safe_path(&target)
        );
    } else {
        let mut stdout = std::io::stdout();
        stdout.write_all(rendered.as_bytes())?;
        if !rendered.ends_with('\n') {
            stdout.write_all(b"\n")?;
        }
        stdout.flush()?;
    }
    eprintln!("{}", source_summary_line(source, video_id, cues.len()));
    Ok(())
}

/// The short stderr summary line naming the source that produced the
/// transcript.
fn source_summary_line(source: &SourceIdentity, video_id: &str, cue_count: usize) -> String {
    match source {
        SourceIdentity::LocalVoxtral {
            model, language, ..
        } => format!(
            "source: local-voxtral, video: {video_id}, model: {model}, language: {}, cues: {cue_count}",
            language.as_deref().unwrap_or("unknown")
        ),
        SourceIdentity::LocalWhisper { backend, .. } => format!(
            "source: local-whisper, video: {video_id}, backend: {backend}, cues: {cue_count}"
        ),
        SourceIdentity::CloudVoxtral {
            model,
            language_hint,
            ..
        } => format!(
            "source: cloud-voxtral, video: {video_id}, model: {model}, language: {}, cues: {cue_count}",
            language_hint.as_deref().unwrap_or("unknown")
        ),
    }
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

    #[test]
    fn cookie_snapshot_keeps_the_prepared_bytes_and_identity_after_source_changes() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source-cookies.txt");
        let original = "# Netscape HTTP Cookie File\n# synthetic-original\n";
        std::fs::write(&source, original).unwrap();
        let prepared = prepare_cookies(std::fs::File::open(&source).unwrap()).unwrap();
        std::fs::write(
            &source,
            "# Netscape HTTP Cookie File\n# synthetic-replacement-longer\n",
        )
        .unwrap();
        assert_ne!(
            prepared.identity,
            file_stat(&std::fs::File::open(&source).unwrap()).unwrap()
        );
        let snapshot = snapshot_cookies(&prepared, directory.path()).unwrap();
        assert_eq!(std::fs::read_to_string(snapshot).unwrap(), original);
        assert_eq!(prepared.identity.size, original.len() as u64);
    }

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
        let args: Vec<String> = common_args(None)
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
            None,
            false,
            false,
            false,
            false,
            Engine::Whisper,
            None,
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

    fn test_cue() -> CaptionCue {
        CaptionCue {
            start_ms: 0,
            end_ms: 1200,
            text: "hello".to_string(),
            speaker: None,
            speaker_marker: false,
        }
    }

    fn test_record(source: SourceIdentity) -> CacheRecord {
        CacheRecord {
            schema_version: CACHE_SCHEMA_VERSION,
            normalization_version: YOUTUBE_NORMALIZATION_VERSION,
            identity: CacheIdentity {
                request: RequestKey {
                    cache_schema_version: CACHE_SCHEMA_VERSION,
                    normalization_version: YOUTUBE_NORMALIZATION_VERSION,
                    selection_policy_version: AUDIO_POLICY_VERSION,
                    video_id: "dQw4w9WgXcQ".to_string(),
                    requested_language: None,
                    voxtral_executable: None,
                    voxtral_executable_identity: None,
                    cookie_file_identity: None,
                    engine: Engine::Whisper,
                    voxtral_model: None,
                },
                source,
            },
            acquired_at: "2026-01-02T03:04:05+00:00".to_string(),
            metadata: VideoSummary::default(),
            original_cues: vec![test_cue()],
            cues: vec![test_cue()],
            normalization: NormalizationStats::default(),
        }
    }

    fn local_source() -> (SourceIdentity, Vec<CaptionCue>) {
        (
            SourceIdentity::LocalWhisper {
                model_name: "ggml-base.bin".to_string(),
                model_stat: ModelStat {
                    device: 0,
                    inode: 0,
                    size: 1,
                    modified_seconds: 0,
                    modified_nanoseconds: 0,
                    changed_seconds: 0,
                    changed_nanoseconds: 0,
                },
                backend: "cpu".to_string(),
                language_hint: None,
            },
            vec![test_cue()],
        )
    }

    fn cloud_source() -> (SourceIdentity, Vec<CaptionCue>) {
        (
            SourceIdentity::CloudVoxtral {
                model: "voxtral-mini-transcribe-3".to_string(),
                language_hint: None,
            },
            vec![test_cue()],
        )
    }

    #[test]
    fn engine_whisper_runs_the_local_path_only() {
        let mut cloud_calls = 0;
        let (source, cues) = transcribe_with_selected_engine(
            ResolvedEngine::Whisper,
            || {
                cloud_calls += 1;
                Ok(cloud_source())
            },
            || Ok(local_source()),
        )
        .unwrap();
        assert_eq!(cloud_calls, 0);
        assert!(matches!(source, SourceIdentity::LocalWhisper { .. }));
        assert_eq!(cues.len(), 1);
    }

    #[test]
    fn engine_voxtral_cloud_success_keeps_the_cloud_identity_and_skips_local() {
        let mut local_calls = 0;
        let (source, _) = transcribe_with_selected_engine(
            ResolvedEngine::VoxtralCloud,
            || Ok(cloud_source()),
            || {
                local_calls += 1;
                Ok(local_source())
            },
        )
        .unwrap();
        assert_eq!(local_calls, 0);
        let SourceIdentity::CloudVoxtral {
            model,
            language_hint,
        } = source
        else {
            panic!("expected the cloud source identity");
        };
        assert_eq!(model, "voxtral-mini-transcribe-3");
        assert_eq!(language_hint, None);
    }

    #[test]
    fn engine_voxtral_cloud_aborts_with_cloud_kind_and_no_fallback_on_error() {
        let mut local_calls = 0;
        let error = transcribe_with_selected_engine(
            ResolvedEngine::VoxtralCloud,
            || Err(anyhow!("mock cloud failure without any key material")),
            || {
                local_calls += 1;
                Ok(local_source())
            },
        )
        .unwrap_err();
        assert_eq!(local_calls, 0, "explicit voxtral-cloud must not fall back");
        assert_eq!(failure_kind(&error), Some(YouTubeErrorKind::Cloud));
        let message = format!("{error:#}");
        assert!(message.contains("cloud"), "{message}");
        assert!(message.contains("mock cloud failure"), "{message}");
    }

    #[test]
    fn engine_auto_cloud_falls_back_to_local_and_keeps_the_real_source() {
        let mut cloud_calls = 0;
        let mut local_calls = 0;
        let (source, cues) = transcribe_with_selected_engine(
            ResolvedEngine::AutoCloud,
            || {
                cloud_calls += 1;
                Err(youtube_error(YouTubeErrorKind::Cloud, "mock cloud failure"))
            },
            || {
                local_calls += 1;
                Ok(local_source())
            },
        )
        .unwrap();
        assert_eq!(cloud_calls, 1);
        assert_eq!(local_calls, 1);
        // The cached identity is the engine that really produced the result.
        assert!(matches!(source, SourceIdentity::LocalWhisper { .. }));
        assert_eq!(cues.len(), 1);
    }

    #[test]
    fn engine_auto_cloud_reports_both_errors_when_the_fallback_fails_too() {
        let error = transcribe_with_selected_engine(
            ResolvedEngine::AutoCloud,
            || Err(youtube_error(YouTubeErrorKind::Cloud, "cloud reason")),
            || Err(youtube_error(YouTubeErrorKind::Whisper, "whisper reason")),
        )
        .unwrap_err();
        assert_eq!(failure_kind(&error), Some(YouTubeErrorKind::Whisper));
        let message = format!("{error:#}");
        assert!(message.contains("cloud reason"), "{message}");
        assert!(message.contains("whisper reason"), "{message}");
    }

    /// Serializes tests that mutate process environment variables.
    static ENV_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

    #[test]
    fn resolve_engine_auto_follows_only_the_token_file_and_selects_cloud_first() {
        use polyscribe_core::voxtral_cloud::{ENV_API_KEY, ENV_TOKEN_FILE};
        let _guard = ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        let old_key = std::env::var(ENV_API_KEY).ok();
        let old_file = std::env::var_os(ENV_TOKEN_FILE);
        // Always point the override into the temp dir, so the real token
        // file of this machine is never read.
        let dir = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var(ENV_TOKEN_FILE, dir.path().join("absent")) };
        unsafe { std::env::remove_var(ENV_API_KEY) };
        // No token file: auto is local, even with MISTRAL_API_KEY set.
        assert_eq!(resolve_engine(Engine::Auto), ResolvedEngine::Whisper);
        unsafe { std::env::set_var(ENV_API_KEY, "env-only-key") };
        assert_eq!(resolve_engine(Engine::Auto), ResolvedEngine::Whisper);

        let token = dir.path().join("mistral_token");
        std::fs::write(&token, "file-key\n").unwrap();
        unsafe { std::env::set_var(ENV_TOKEN_FILE, &token) };
        assert_eq!(resolve_engine(Engine::Auto), ResolvedEngine::AutoCloud);
        // With the token file configured, auto tries the cloud first and
        // falls back to the local path on a cloud error.
        let mut local_calls = 0;
        let (source, _) = transcribe_with_selected_engine(
            resolve_engine(Engine::Auto),
            || Err(anyhow!("offline mock cloud failure")),
            || {
                local_calls += 1;
                Ok(local_source())
            },
        )
        .unwrap();
        assert_eq!(local_calls, 1);
        assert!(matches!(source, SourceIdentity::LocalWhisper { .. }));

        match old_key {
            Some(v) => unsafe { std::env::set_var(ENV_API_KEY, v) },
            None => unsafe { std::env::remove_var(ENV_API_KEY) },
        }
        match old_file {
            Some(v) => unsafe { std::env::set_var(ENV_TOKEN_FILE, v) },
            None => unsafe { std::env::remove_var(ENV_TOKEN_FILE) },
        }
    }

    #[test]
    fn entries_to_cues_drops_the_empty_speaker_and_sorts_by_start() {
        use polyscribe_core::OutputEntry;
        let entries = vec![
            OutputEntry {
                id: 0,
                speaker: String::new(),
                start: 2.0,
                end: 3.0,
                text: "second".to_string(),
            },
            OutputEntry {
                id: 0,
                speaker: "alice".to_string(),
                start: 0.5,
                end: 1.0,
                text: "first".to_string(),
            },
        ];
        let cues = entries_to_cues(entries).unwrap();
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "first");
        assert_eq!(cues[0].start_ms, 500);
        assert_eq!(cues[0].speaker.as_deref(), Some("alice"));
        assert_eq!(cues[1].text, "second");
        assert_eq!(cues[1].speaker, None, "an empty label must become None");
        let rendered = polyscribe_core::youtube::render_txt(&cues);
        assert!(!rendered.contains("speaker1"), "{rendered}");

        let invalid = vec![OutputEntry {
            id: 0,
            speaker: String::new(),
            start: 0.0,
            end: 1.0,
            text: "   ".to_string(),
        }];
        assert!(entries_to_cues(invalid).is_err());
    }

    #[test]
    fn write_transcript_renders_the_cloud_source_in_json_and_txt() {
        let dir = tempfile::tempdir().unwrap();
        let record = test_record(SourceIdentity::CloudVoxtral {
            model: "voxtral-mini-transcribe-3".to_string(),
            language_hint: None,
        });

        let json_path = dir.path().join("out.json");
        write_transcript(
            "https://youtu.be/dQw4w9WgXcQ",
            "dQw4w9WgXcQ",
            "json",
            Some(&json_path),
            &record,
            false,
        )
        .unwrap();
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
        assert_eq!(json["source"], "cloud-voxtral");
        assert_eq!(json["engine"], "voxtral-cloud");
        assert_eq!(json["model"], "voxtral-mini-transcribe-3");
        assert!(
            json["language"].is_null(),
            "no hint means null, never a guess"
        );
        assert!(json["cues"][0].get("speaker").is_none());

        let txt_path = dir.path().join("out.txt");
        write_transcript(
            "https://youtu.be/dQw4w9WgXcQ",
            "dQw4w9WgXcQ",
            "txt",
            Some(&txt_path),
            &record,
            false,
        )
        .unwrap();
        let txt = std::fs::read_to_string(&txt_path).unwrap();
        assert!(txt.contains("engine: \"voxtral-cloud\""), "{txt}");
        assert!(
            txt.contains("model: \"voxtral-mini-transcribe-3\""),
            "{txt}"
        );
        assert!(txt.contains("language: null"), "{txt}");
        assert!(txt.contains("hello"), "{txt}");
        assert!(!txt.contains("speaker1"), "{txt}");
    }

    #[test]
    fn write_transcript_cloud_language_is_the_hint_never_a_guess() {
        let dir = tempfile::tempdir().unwrap();
        let mut record = test_record(SourceIdentity::CloudVoxtral {
            model: "some-model".to_string(),
            language_hint: Some("de".to_string()),
        });
        record.identity.request.requested_language = Some("de".to_string());
        let json_path = dir.path().join("out.json");
        write_transcript(
            "https://youtu.be/dQw4w9WgXcQ",
            "dQw4w9WgXcQ",
            "json",
            Some(&json_path),
            &record,
            false,
        )
        .unwrap();
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
        assert_eq!(json["language"], "de");
        assert_eq!(json["requested_language"], "de");
        assert_eq!(json["engine"], "voxtral-cloud");
        assert_eq!(json["model"], "some-model");
    }

    #[test]
    fn write_transcript_local_records_carry_no_speaker_label() {
        let dir = tempfile::tempdir().unwrap();
        let record = test_record(SourceIdentity::LocalWhisper {
            model_name: "ggml-base.bin".to_string(),
            model_stat: ModelStat {
                device: 0,
                inode: 0,
                size: 1,
                modified_seconds: 0,
                modified_nanoseconds: 0,
                changed_seconds: 0,
                changed_nanoseconds: 0,
            },
            backend: "cpu".to_string(),
            language_hint: Some("en".to_string()),
        });
        let txt_path = dir.path().join("out.txt");
        write_transcript("u", "dQw4w9WgXcQ", "txt", Some(&txt_path), &record, false).unwrap();
        let txt = std::fs::read_to_string(&txt_path).unwrap();
        assert!(!txt.contains("speaker1"), "{txt}");
        assert!(txt.contains("engine: \"whisper\""), "{txt}");

        let json_path = dir.path().join("out.json");
        write_transcript("u", "dQw4w9WgXcQ", "json", Some(&json_path), &record, false).unwrap();
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
        assert_eq!(json["source"], "local-whisper");
        assert!(json["cues"][0].get("speaker").is_none());
    }

    #[test]
    fn source_summary_line_names_each_source() {
        let cloud = SourceIdentity::CloudVoxtral {
            model: "some-model".to_string(),
            language_hint: None,
        };
        let line = source_summary_line(&cloud, "vid", 3);
        assert!(line.contains("source: cloud-voxtral"), "{line}");
        assert!(line.contains("model: some-model"), "{line}");
        assert!(line.contains("cues: 3"), "{line}");
        let whisper = local_source().0;
        let line = source_summary_line(&whisper, "vid", 2);
        assert!(line.contains("source: local-whisper"), "{line}");
        assert!(line.contains("backend: cpu"), "{line}");
    }
}

// SPDX-License-Identifier: MIT

//! Cloud transcription engine backed by Mistral's Audio Transcriptions API
//! (Voxtral), `POST /v1/audio/transcriptions` on `https://api.mistral.ai`.
//!
//! The audio is decoded locally in windows (the same `chunked` module the
//! whisper pipeline uses), every window is encoded as a small WAV file and
//! uploaded as one multipart request, and the segment timestamps of the
//! response are shifted by the window's position in the recording. The API
//! caps one request (context window 65,536 tokens shared by audio and
//! vocabulary, multi-hour/size limits on uploads), so chunking is what keeps
//! long recordings transcribable; a 30 s window is about 1 MB of WAV and far
//! below every documented limit.
//!
//! Sources (checked 2026, see the project report for the full list):
//! - https://docs.mistral.ai/studio/audio/speech-to-text/transcribe-files
//! - https://docs.mistral.ai/api/endpoint/audio/transcriptions
//! - https://docs.mistral.ai/studio/audio/specs-and-limits/limits
//!
//! Privacy: this engine is pure opt-in and there is no automatic fallback
//! from a local engine to this one. The API key is read from
//! `MISTRAL_API_KEY` or, when that is unset, from the file `mistral_token` in
//! the polyscribe config directory (`~/.config/polyscribe/mistral_token` on
//! Linux; `POLYSCRIBE_MISTRAL_TOKEN_FILE` overrides the path), is never stored in this struct (so `Debug` cannot
//! leak it), never logged, and never written to files or cache keys; error
//! messages scrub it from response bodies. Before the first upload of a run
//! a one-line warning is written to stderr (suppressible with `--yes` or
//! `POLYSCRIBE_CLOUD_OK=1`).

use std::env;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, anyhow};
use reqwest::blocking::{Client, multipart};
use serde::Deserialize;

use crate::OutputEntry;
use crate::backend::{BackendKind, TranscribeBackend};
use crate::chunked::{AudioWindow, ChunkConfig, ChunkedAudio, WHISPER_SAMPLE_RATE};
use crate::prelude::*;

/// Default model of the cloud engine, a file-transcription Voxtral model from
/// the Mistral API docs (the page lists `voxtral-mini-transcribe-3` and
/// `voxtral-small-transcribe-3`).
pub const DEFAULT_VOXTRAL_CLOUD_MODEL: &str = "voxtral-mini-transcribe-3";

/// Base URL of the Mistral API.
pub const MISTRAL_API_BASE: &str = "https://api.mistral.ai";
/// Path of the Audio Transcriptions endpoint.
pub const TRANSCRIPTIONS_PATH: &str = "/v1/audio/transcriptions";
/// The only place the API key is read from.
pub const ENV_API_KEY: &str = "MISTRAL_API_KEY";
/// Overrides the path of the token file.
pub const ENV_TOKEN_FILE: &str = "POLYSCRIBE_MISTRAL_TOKEN_FILE";
/// File name of the token file in the polyscribe config directory.
const TOKEN_FILE_NAME: &str = "mistral_token";
/// Environment variable that suppresses the one-time cloud warning.
pub const ENV_CLOUD_OK: &str = "POLYSCRIBE_CLOUD_OK";

/// Per-request timeout: a 30 s chunk can take a while under server load.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// How often a failed request is retried (in addition to the first attempt).
const MAX_RETRIES: u32 = 3;
/// Backoff base for retries; the wait doubles per attempt up to `RETRY_CAP`.
const RETRY_BASE: Duration = Duration::from_secs(2);
const RETRY_CAP: Duration = Duration::from_secs(30);

/// A transcription backend that uploads audio to the Mistral cloud API.
///
/// The API key is deliberately not a field: it is read from the environment
/// per transcription run, so no `Debug`/`Display` output of this struct can
/// contain it.
#[derive(Debug)]
pub struct CloudVoxtralBackend {
    model: String,
    endpoint: String,
    client: Client,
    retry_base: Duration,
}

impl CloudVoxtralBackend {
    /// A backend for the public Mistral endpoint with the given model id.
    pub fn new(model: impl Into<String>) -> Result<Self> {
        Self::with_endpoint(model, format!("{MISTRAL_API_BASE}{TRANSCRIPTIONS_PATH}"))
    }

    /// A backend for a custom endpoint (used by the tests to point at a local
    /// mock server).
    pub fn with_endpoint(model: impl Into<String>, endpoint: impl Into<String>) -> Result<Self> {
        let client = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| anyhow!("Failed to build the HTTP client for the Mistral API: {e}"))?;
        Ok(Self {
            model: model.into(),
            endpoint: endpoint.into(),
            client,
            retry_base: RETRY_BASE,
        })
    }

    /// The API key: `MISTRAL_API_KEY` first, then the token file, else an
    /// actionable error. The key itself never appears in an error message.
    fn api_key() -> Result<String> {
        if let Ok(key) = env::var(ENV_API_KEY)
            && !key.trim().is_empty()
        {
            return Ok(key.trim().to_string());
        }
        if let Some(path) = token_file_path()
            && let Some(key) = read_token_file(&path)?
        {
            return Ok(key);
        }
        let shown = token_file_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<config dir>/mistral_token".to_string());
        Err(anyhow!(
            "No Mistral API key found. The voxtral-cloud engine needs one (create it at \
             https://console.mistral.ai): set {ENV_API_KEY}, or put it in {shown}."
        )
        .into())
    }

    /// Upload one audio chunk and return the parsed response.
    ///
    /// Retries with exponential backoff only on HTTP 429 and 5xx, at most
    /// `MAX_RETRIES` times; any other 4xx fails immediately. The API key is
    /// only ever sent as the `Authorization: Bearer` header and is scrubbed
    /// from every response body that reaches an error message.
    fn upload_chunk(
        &self,
        api_key: &str,
        bytes: &[u8],
        file_name: &str,
        language: Option<&str>,
    ) -> Result<TranscriptionResponse> {
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let mut form = multipart::Form::new()
                .text("model", self.model.clone())
                .text("timestamp_granularities", "segment");
            if let Some(lang) = language {
                form = form.text("language", lang.to_string());
            }
            let part = multipart::Part::bytes(bytes.to_vec())
                .file_name(file_name.to_string())
                .mime_str("audio/wav")
                .map_err(|e| anyhow!("Failed to build the audio upload part: {e}"))?;
            let form = form.part("file", part);

            let resp = self
                .client
                .post(&self.endpoint)
                .bearer_auth(api_key)
                .multipart(form)
                .send()
                .with_context(|| {
                    format!(
                        "Failed to send the transcription request to {}",
                        self.endpoint
                    )
                })?;
            let status = resp.status();
            let body = resp.text().unwrap_or_default();

            if status.is_success() {
                let parsed =
                    serde_json::from_str::<TranscriptionResponse>(&body).with_context(|| {
                        format!(
                            "The transcription response is not valid JSON (HTTP {}): {}",
                            status.as_u16(),
                            scrub(&body, api_key)
                        )
                    });
                return Ok(parsed?);
            }

            let message = format!(
                "Mistral API error: HTTP {} {}",
                status.as_u16(),
                scrub(&body, api_key)
            );
            let retryable = status.as_u16() == 429 || status.is_server_error();
            if retryable && attempt <= MAX_RETRIES {
                let backoff = (self.retry_base * 2u32.pow(attempt - 1)).min(RETRY_CAP);
                crate::dlog!(
                    1,
                    "{}; retrying in {:?} (attempt {}/{})",
                    message,
                    backoff,
                    attempt,
                    MAX_RETRIES
                );
                std::thread::sleep(backoff);
                continue;
            }
            if retryable {
                return Err(
                    anyhow!("{message} (after {attempt} attempts, no further retry)").into(),
                );
            }
            return Err(anyhow!("{message}").into());
        }
    }
}

impl TranscribeBackend for CloudVoxtralBackend {
    /// The cloud engine does no local GPU compute. `BackendKind` has no cloud
    /// variant (it describes the whisper backends), so this reports `Cpu` as
    /// the local-compute placeholder; the CLI reports the engine separately.
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn transcribe(
        &mut self,
        audio_path: &Path,
        speaker: &str,
        language: Option<&str>,
        _gpu_layers: Option<u32>,
        progress: Option<&(dyn Fn(i32) + Send + Sync)>,
    ) -> Result<Vec<OutputEntry>> {
        let api_key = Self::api_key()?;
        warn_cloud_upload_once();

        let report = |p: i32| {
            if let Some(cb) = progress {
                cb(p);
            }
        };
        report(0);

        // The same windowed decoder the whisper pipeline uses: bounded memory,
        // and one upload per 30 s window.
        let mut windows = ChunkedAudio::new(audio_path, ChunkConfig::default())?;
        let mut entries: Vec<OutputEntry> = Vec::new();
        let mut done = 0u64;
        loop {
            let Some(window) = windows.next_window()? else {
                break;
            };
            done += 1;
            let wav = encode_wav_pcm16(&window.samples, WHISPER_SAMPLE_RATE);
            let file_name = format!("chunk_{done:04}.wav");
            let response = self.upload_chunk(&api_key, &wav, &file_name, language)?;
            crate::dlog!(1, "chunk {done}: detected language {:?}", response.language);
            entries.extend(map_response(&response, &window, speaker));
            report(((5 + done * 10).min(95)) as i32);
            // The windows do not overlap on this path: every sample is
            // uploaded exactly once.
            windows.seek_to(window.start_sample + window.samples.len() as u64);
        }
        if done == 0 {
            return Err(anyhow!(
                "The input contains no audio samples: {}",
                audio_path.display()
            )
            .into());
        }
        report(100);
        Ok(entries)
    }
}

/// Path of the token file: `POLYSCRIBE_MISTRAL_TOKEN_FILE`, else
/// `mistral_token` in the polyscribe config directory.
fn token_file_path() -> Option<PathBuf> {
    if let Some(p) = env::var_os(ENV_TOKEN_FILE).filter(|p| !p.is_empty()) {
        return Some(PathBuf::from(p));
    }
    crate::config::ConfigService::project_dirs().map(|d| d.config_dir().join(TOKEN_FILE_NAME))
}

/// Read the key from the token file. A missing file is `Ok(None)`; an
/// unreadable file is an error that names the path but never the content.
/// On Unix a file readable by group or others is reported on stderr.
fn read_token_file(path: &Path) -> Result<Option<String>> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(anyhow!(
                "Cannot read the Mistral token file {}: {}",
                path.display(),
                e.kind()
            )
            .into());
        }
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(path)
            && meta.permissions().mode() & 0o077 != 0
        {
            eprintln!(
                "warning: {} is readable by other users; restrict it with `chmod 600`.",
                path.display()
            );
        }
    }
    let key = raw.trim();
    Ok((!key.is_empty()).then(|| key.to_string()))
}

/// Whether the user already consented to cloud uploads (`--yes` or
/// `POLYSCRIBE_CLOUD_OK=1`).
fn cloud_consent_given() -> bool {
    CLOUD_CONSENT.load(Ordering::SeqCst) || env::var(ENV_CLOUD_OK).is_ok_and(|v| v == "1")
}

static CLOUD_CONSENT: AtomicBool = AtomicBool::new(false);
static CLOUD_WARNED: AtomicBool = AtomicBool::new(false);

/// Record that the user consented to cloud uploads already (CLI `--yes`).
pub fn set_cloud_consent(granted: bool) {
    CLOUD_CONSENT.store(granted, Ordering::SeqCst);
}

/// Write the one-line data-protection warning before the first upload of a
/// run, unless it was suppressed.
fn warn_cloud_upload_once() {
    if cloud_consent_given() {
        return;
    }
    if CLOUD_WARNED.swap(true, Ordering::SeqCst) {
        return;
    }
    crate::ui::warn(
        "Uploading audio to the Mistral cloud API (api.mistral.ai) for transcription. \
         Suppress this warning with --yes or POLYSCRIBE_CLOUD_OK=1.",
    );
}

/// The response of `POST /v1/audio/transcriptions`. Only the fields PolyScribe
/// uses are modeled; unknown fields are ignored.
#[derive(Debug, Deserialize)]
struct TranscriptionResponse {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    segments: Vec<TranscriptionSegment>,
    #[serde(default)]
    language: Option<String>,
}

/// One segment of the transcript, with its times in seconds relative to the
/// uploaded chunk.
#[derive(Debug, Deserialize)]
struct TranscriptionSegment {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    start: Option<f64>,
    #[serde(default)]
    end: Option<f64>,
    // Present in the API response when `diarize` is set; PolyScribe attaches
    // its own speaker label, so it is not read. Modeled for documentation.
    #[serde(default)]
    #[allow(dead_code)]
    speaker_id: Option<String>,
}

/// Map a chunk response to output entries.
///
/// Segment timestamps of the response are relative to the chunk, so they are
/// shifted by the window's start. When the API delivers no usable segments,
/// the full text becomes one entry with the honest chunk bounds; no times are
/// invented.
fn map_response(
    response: &TranscriptionResponse,
    window: &AudioWindow,
    speaker: &str,
) -> Vec<OutputEntry> {
    let offset = window.start_seconds();
    let mut entries = Vec::new();
    for segment in &response.segments {
        let (Some(start), Some(end), Some(text)) = (&segment.start, &segment.end, &segment.text)
        else {
            continue;
        };
        let text = text.trim();
        if text.is_empty() || !start.is_finite() || !end.is_finite() || end <= start {
            continue;
        }
        entries.push(OutputEntry {
            id: 0,
            speaker: speaker.to_string(),
            start: start + offset,
            end: end + offset,
            text: text.to_string(),
        });
    }
    if entries.is_empty()
        && let Some(text) = &response.text
    {
        let text = text.trim();
        if !text.is_empty() {
            entries.push(OutputEntry {
                id: 0,
                speaker: speaker.to_string(),
                start: offset,
                end: window.end_seconds(),
                text: text.to_string(),
            });
        }
    }
    entries
}

/// Replace the API key in a text that is about to be shown in an error
/// message, and truncate the result. Defense in depth: the key should never
/// appear in a response body, but if it ever does, it must not be reported.
fn scrub(text: &str, api_key: &str) -> String {
    let scrubbed = if api_key.is_empty() {
        text.to_string()
    } else {
        text.replace(api_key, "***")
    };
    let mut out: String = scrubbed.chars().take(500).collect();
    if scrubbed.len() > 500 {
        out.push_str("...");
    }
    out
}

/// Encode 16 kHz mono f32 PCM as a 16-bit WAV file (in memory).
fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read as _, Write as _};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex, OnceLock};

    /// Serializes tests that mutate process environment variables.
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    const TEST_KEY: &str = "test-secret-key-123";

    struct RecordedRequest {
        head: String,
        body: Vec<u8>,
    }

    /// What the mock answers with.
    enum Reply {
        Static(u16, String),
        /// Answer with the given status and echo the request's Authorization
        /// header in the body (to prove that errors never leak the key).
        EchoAuthHeader(u16),
    }

    /// A minimal HTTP/1.1 mock server on a thread: one canned reply per
    /// incoming connection, every request recorded.
    fn spawn_mock(replies: Vec<Reply>) -> (String, Arc<Mutex<Vec<RecordedRequest>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        std::thread::spawn(move || {
            for reply in replies {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .unwrap();
                let request = read_request(&mut stream);
                let (status, body) = match reply {
                    Reply::Static(status, body) => (status, body),
                    Reply::EchoAuthHeader(status) => {
                        let auth = request
                            .head
                            .lines()
                            .find(|l| l.to_ascii_lowercase().starts_with("authorization:"))
                            .unwrap_or("")
                            .to_string();
                        (status, auth)
                    }
                };
                recorded.lock().unwrap().push(request);
                let reason = match status {
                    200 => "OK",
                    401 => "Unauthorized",
                    429 => "Too Many Requests",
                    _ => "Internal Server Error",
                };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });
        (format!("http://{addr}/v1/audio/transcriptions"), requests)
    }

    /// Read one request: headers up to the empty line, then Content-Length
    /// bytes of body.
    fn read_request(stream: &mut TcpStream) -> RecordedRequest {
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            match stream.read(&mut byte) {
                Ok(0) | Err(_) => break,
                Ok(_) => head.push(byte[0]),
            }
            if head.ends_with(b"\r\n\r\n") || head.len() > 64 * 1024 {
                break;
            }
        }
        let head_text = String::from_utf8_lossy(&head).to_string();
        let content_length = head_text
            .lines()
            .find_map(|l| {
                l.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .and_then(|v| v.trim().parse::<usize>().ok())
            })
            .unwrap_or(0);
        let mut body = vec![0u8; content_length];
        let mut read = 0;
        while read < content_length {
            match stream.read(&mut body[read..]) {
                Ok(0) | Err(_) => break,
                Ok(n) => read += n,
            }
        }
        body.truncate(read);
        RecordedRequest {
            head: head_text,
            body,
        }
    }

    fn test_backend(endpoint: &str) -> CloudVoxtralBackend {
        let mut backend =
            CloudVoxtralBackend::with_endpoint(DEFAULT_VOXTRAL_CLOUD_MODEL, endpoint).unwrap();
        // Keep the retry tests fast.
        backend.retry_base = Duration::from_millis(1);
        backend
    }

    fn ok_body() -> String {
        r#"{"model":"voxtral-mini-transcribe-3","text":" Hello world ","language":"en","segments":[{"type":"transcription_segment","text":" Hello ","start":0.5,"end":1.5,"speaker_id":null},{"type":"transcription_segment","text":"world","start":2.0,"end":3.0,"speaker_id":"S1"}],"usage":{"prompt_audio_seconds":3.0}}"#
            .to_string()
    }

    fn window(start_sample: u64, samples: usize) -> AudioWindow {
        AudioWindow {
            start_sample,
            samples: vec![0.0; samples],
        }
    }

    #[test]
    fn test_success_segments_are_mapped_with_the_chunk_offset() {
        let (endpoint, requests) = spawn_mock(vec![Reply::Static(200, ok_body())]);
        let backend = test_backend(&endpoint);
        let response = backend
            .upload_chunk(TEST_KEY, b"wav-bytes", "chunk_0001.wav", Some("de"))
            .unwrap();
        assert_eq!(response.segments.len(), 2);
        assert_eq!(response.language.as_deref(), Some("en"));

        // The window starts at 30 s: the segment times are shifted by it.
        let entries = map_response(&response, &window(30 * 16_000, 480_000), "alice");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].start, 30.5);
        assert_eq!(entries[0].end, 31.5);
        assert_eq!(entries[0].text, "Hello");
        assert_eq!(entries[0].speaker, "alice");
        assert_eq!(entries[1].start, 32.0);
        assert_eq!(entries[1].end, 33.0);
        assert_eq!(entries[1].text, "world");

        let recorded = requests.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        let head = recorded[0].head.to_ascii_lowercase();
        assert!(
            head.contains("authorization: bearer test-secret-key-123"),
            "{head}"
        );
        assert!(head.contains("content-type: multipart/form-data"), "{head}");
        let body = String::from_utf8_lossy(&recorded[0].body);
        assert!(body.contains("voxtral-mini-transcribe-3"), "{body}");
        assert!(body.contains("timestamp_granularities"), "{body}");
        assert!(body.contains("segment"), "{body}");
        assert!(body.contains("language"), "{body}");
        assert!(body.contains("de"), "{body}");
        assert!(body.contains("chunk_0001.wav"), "{body}");
        assert!(body.contains("audio/wav"), "{body}");
        assert!(body.contains("wav-bytes"), "{body}");
    }

    #[test]
    fn test_429_is_retried_with_backoff_and_then_succeeds() {
        let (endpoint, requests) = spawn_mock(vec![
            Reply::Static(429, r#"{"message":"rate limited"}"#.to_string()),
            Reply::Static(200, ok_body()),
        ]);
        let backend = test_backend(&endpoint);
        let response = backend
            .upload_chunk(TEST_KEY, b"x", "chunk_0001.wav", None)
            .unwrap();
        assert_eq!(response.segments.len(), 2);
        assert_eq!(requests.lock().unwrap().len(), 2);
    }

    #[test]
    fn test_5xx_exhausts_the_bounded_retries() {
        let (endpoint, requests) = spawn_mock(vec![
            Reply::Static(500, "boom1".to_string()),
            Reply::Static(500, "boom2".to_string()),
            Reply::Static(503, "boom3".to_string()),
            Reply::Static(500, "boom4".to_string()),
        ]);
        let backend = test_backend(&endpoint);
        let err = backend
            .upload_chunk(TEST_KEY, b"x", "chunk_0001.wav", None)
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("HTTP 500"), "{msg}");
        assert!(msg.contains("no further retry"), "{msg}");
        // 1 initial attempt + MAX_RETRIES retries, and not one more.
        assert_eq!(requests.lock().unwrap().len(), 4);
    }

    #[test]
    fn test_401_is_not_retried_and_the_key_is_masked() {
        let (endpoint, requests) = spawn_mock(vec![
            Reply::EchoAuthHeader(401),
            Reply::Static(200, ok_body()),
        ]);
        let backend = test_backend(&endpoint);
        let err = backend
            .upload_chunk(TEST_KEY, b"x", "chunk_0001.wav", None)
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("HTTP 401"), "{msg}");
        assert!(
            !msg.contains(TEST_KEY),
            "the API key must never appear in an error message: {msg}"
        );
        assert!(msg.contains("***"), "the echoed key is scrubbed: {msg}");
        // A 4xx (other than 429) is never retried.
        assert_eq!(requests.lock().unwrap().len(), 1);
    }

    #[test]
    fn test_broken_json_response_is_an_error() {
        let (endpoint, _requests) =
            spawn_mock(vec![Reply::Static(200, "this is not json".to_string())]);
        let backend = test_backend(&endpoint);
        let err = backend
            .upload_chunk(TEST_KEY, b"x", "chunk_0001.wav", None)
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("not valid JSON"), "{msg}");
        assert!(!msg.contains(TEST_KEY), "{msg}");
    }

    #[test]
    fn test_response_without_segments_falls_back_to_chunk_bounds() {
        let response: TranscriptionResponse =
            serde_json::from_str(r#"{"text":" Hello world ","segments":[]}"#).unwrap();
        let w = window(60 * 16_000, 480_000); // 30 s window starting at 60 s
        let entries = map_response(&response, &w, "bob");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].start, 60.0);
        assert_eq!(entries[0].end, 90.0);
        assert_eq!(entries[0].text, "Hello world");
        assert_eq!(entries[0].speaker, "bob");
    }

    #[test]
    fn test_segments_without_times_are_dropped_and_text_fallback_used() {
        let response: TranscriptionResponse = serde_json::from_str(
            r#"{"text":"full text","segments":[{"text":"no times"},{"text":"  ","start":1.0,"end":2.0},{"text":"zero","start":2.0,"end":2.0}]}"#,
        )
        .unwrap();
        let entries = map_response(&response, &window(0, 16_000), "bob");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "full text");
        assert_eq!(entries[0].start, 0.0);
        assert_eq!(entries[0].end, 1.0);
    }

    #[test]
    fn test_empty_response_produces_no_entries() {
        let response: TranscriptionResponse = serde_json::from_str("{}").unwrap();
        assert!(map_response(&response, &window(0, 16_000), "bob").is_empty());
    }

    #[test]
    fn test_missing_api_key_is_a_clear_error() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        let old = env::var(ENV_API_KEY).ok();
        let old_file = env::var_os(ENV_TOKEN_FILE);
        let dir = tempfile::tempdir().unwrap();
        unsafe { env::remove_var(ENV_API_KEY) };
        unsafe { env::set_var(ENV_TOKEN_FILE, dir.path().join("absent")) };
        let err = CloudVoxtralBackend::api_key().unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("MISTRAL_API_KEY"), "{msg}");
        assert!(msg.contains("absent"), "{msg}");
        match old {
            Some(v) => unsafe { env::set_var(ENV_API_KEY, v) },
            None => unsafe { env::remove_var(ENV_API_KEY) },
        }
        match old_file {
            Some(v) => unsafe { env::set_var(ENV_TOKEN_FILE, v) },
            None => unsafe { env::remove_var(ENV_TOKEN_FILE) },
        }
    }

    #[test]
    fn test_key_comes_from_the_token_file_and_env_wins() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        let old = env::var(ENV_API_KEY).ok();
        let old_file = env::var_os(ENV_TOKEN_FILE);
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("mistral_token");
        std::fs::write(&file, "  file-secret-key \n").unwrap();
        unsafe { env::set_var(ENV_TOKEN_FILE, &file) };
        unsafe { env::remove_var(ENV_API_KEY) };
        assert_eq!(CloudVoxtralBackend::api_key().unwrap(), "file-secret-key");
        unsafe { env::set_var(ENV_API_KEY, "env-secret-key") };
        assert_eq!(CloudVoxtralBackend::api_key().unwrap(), "env-secret-key");
        // An empty file counts as no key, and the error does not leak content.
        unsafe { env::remove_var(ENV_API_KEY) };
        std::fs::write(&file, "\n").unwrap();
        let msg = format!("{:#}", CloudVoxtralBackend::api_key().unwrap_err());
        assert!(msg.contains("No Mistral API key"), "{msg}");
        match old {
            Some(v) => unsafe { env::set_var(ENV_API_KEY, v) },
            None => unsafe { env::remove_var(ENV_API_KEY) },
        }
        match old_file {
            Some(v) => unsafe { env::set_var(ENV_TOKEN_FILE, v) },
            None => unsafe { env::remove_var(ENV_TOKEN_FILE) },
        }
    }

    #[test]
    fn test_cloud_consent_flag_and_env() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        let old = env::var(ENV_CLOUD_OK).ok();
        set_cloud_consent(true);
        assert!(cloud_consent_given());
        set_cloud_consent(false);
        unsafe { env::remove_var(ENV_CLOUD_OK) };
        assert!(!cloud_consent_given());
        unsafe { env::set_var(ENV_CLOUD_OK, "1") };
        assert!(cloud_consent_given());
        match old {
            Some(v) => unsafe { env::set_var(ENV_CLOUD_OK, v) },
            None => unsafe { env::remove_var(ENV_CLOUD_OK) },
        }
    }

    #[test]
    fn test_encode_wav_pcm16_header_and_samples() {
        let samples = [0.0f32, 0.5, -0.5, 1.0, -2.0]; // the last one clamps
        let wav = encode_wav_pcm16(&samples, WHISPER_SAMPLE_RATE);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1); // mono
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            16_000
        );
        assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16); // bits
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]), 10);
        assert_eq!(wav.len(), 54);
        let s = |i: usize| i16::from_le_bytes([wav[44 + i * 2], wav[45 + i * 2]]);
        assert_eq!(s(0), 0);
        assert_eq!(s(1), 16384); // 0.5 * 32767 rounded
        assert_eq!(s(2), -16384);
        assert_eq!(s(3), 32767);
        assert_eq!(s(4), -32767); // clamped
    }

    #[test]
    fn test_full_transcribe_over_mock_with_a_real_wav_file() {
        // The windowed decode needs ffmpeg; skip when it is not installed.
        if std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_err()
        {
            eprintln!("SKIPPED: ffmpeg not available");
            return;
        }
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        let old_key = env::var(ENV_API_KEY).ok();
        unsafe { env::set_var(ENV_API_KEY, TEST_KEY) };

        // A 2 s sine wave as a WAV file, written with the same encoder the
        // backend uses for its chunks.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("input.wav");
        let samples: Vec<f32> = (0..2 * WHISPER_SAMPLE_RATE as usize)
            .map(|i| {
                (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / WHISPER_SAMPLE_RATE as f32).sin()
                    * 0.2
            })
            .collect();
        std::fs::write(&path, encode_wav_pcm16(&samples, WHISPER_SAMPLE_RATE)).unwrap();

        let (endpoint, requests) = spawn_mock(vec![Reply::Static(200, ok_body())]);
        let mut backend = test_backend(&endpoint);
        let progress_values = std::sync::Mutex::new(Vec::new());
        let entries = backend
            .transcribe(
                &path,
                "carol",
                Some("en"),
                None,
                Some(&|p| {
                    progress_values.lock().unwrap().push(p);
                }),
            )
            .unwrap();
        // One window (2 s < 30 s), two segments at offset 0.
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].start, 0.5);
        assert_eq!(entries[0].end, 1.5);
        assert_eq!(entries[0].speaker, "carol");
        assert_eq!(entries[1].text, "world");
        assert_eq!(requests.lock().unwrap().len(), 1);
        let progress_values = progress_values.lock().unwrap();
        assert!(progress_values.contains(&100), "{progress_values:?}");

        match old_key {
            Some(v) => unsafe { env::set_var(ENV_API_KEY, v) },
            None => unsafe { env::remove_var(ENV_API_KEY) },
        }
    }
}

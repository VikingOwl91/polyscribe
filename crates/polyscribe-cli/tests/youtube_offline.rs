// SPDX-License-Identifier: MIT

//! Offline end-to-end checks for the YouTube command with a fake yt-dlp.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const URL: &str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ";

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_polyscribe")
}

fn quote(path: &Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', "'\\''"))
}

fn fake_downloader(root: &Path, metadata: &str, captions: &str, extra: &str) -> (PathBuf, PathBuf) {
    let executable = root.join("fake-yt-dlp");
    let calls = root.join("calls.log");
    let script = format!(
        "#!/bin/sh
case \"$*\" in
  *--dump-single-json*)
    printf 'metadata\\n' >> {calls}
    printf '%s\\n' '{metadata}'
    ;;
  *--write-subs*|*--write-auto-subs*)
    printf 'captions\\n' >> {calls}
    output=''
    language='en'
    while [ $# -gt 0 ]; do
      case \"$1\" in
        --output) output=$2; shift 2 ;;
        --sub-langs) language=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    cat > \"$output.$language.vtt\" <<'VTT'
{captions}VTT
    ;;
  *)
    {extra}
    ;;
esac
",
        calls = quote(&calls),
    );
    std::fs::write(&executable, script).unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    (executable, calls)
}

fn run(root: &Path, executable: &Path, args: &[&str]) -> Output {
    Command::new(bin())
        .args(["youtube", URL])
        .args(args)
        .env("POLYSCRIBE_YTDLP", executable)
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("HOME", root)
        .env_remove("WHISPER_MODEL")
        .output()
        .expect("launch polyscribe")
}

fn call_log(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(ToString::to_string)
        .collect()
}

#[test]
fn captions_are_normalized_cached_and_refreshed_without_network_access() {
    let root = tempfile::tempdir().unwrap();
    let metadata = r#"{"id":"dQw4w9WgXcQ","title":"Fixture video","uploader":"Offline","duration":3,"subtitles":{"en":[{"ext":"vtt","url":"https://example.test/en.vtt","name":"English"}]},"automatic_captions":{}}"#;
    let captions = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nThe quick brown\n\n00:00:01.500 --> 00:00:02.500\nThe quick brown fox\n";
    let (executable, calls) = fake_downloader(
        root.path(),
        metadata,
        captions,
        "printf 'unexpected audio download\\n' >> /dev/null; exit 90",
    );

    let first = run(
        root.path(),
        &executable,
        &["--language", "en", "--format", "txt"],
    );
    assert!(
        first.status.success(),
        "first run failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&first.stdout).trim(),
        "The quick brown fox"
    );
    assert_eq!(call_log(&calls), ["metadata", "captions"]);

    let cached = run(
        root.path(),
        &executable,
        &["--language", "en", "--format", "srt"],
    );
    assert!(
        cached.status.success(),
        "cached run failed: {}",
        String::from_utf8_lossy(&cached.stderr)
    );
    assert!(String::from_utf8_lossy(&cached.stdout).contains("00:00:01,000 --> 00:00:02,500"));
    assert!(String::from_utf8_lossy(&cached.stdout).contains("The quick brown fox"));
    assert_eq!(call_log(&calls), ["metadata", "captions"]);

    let refreshed = run(
        root.path(),
        &executable,
        &["--language", "en", "--format", "txt", "--refresh"],
    );
    assert!(
        refreshed.status.success(),
        "refresh failed: {}",
        String::from_utf8_lossy(&refreshed.stderr)
    );
    assert_eq!(
        call_log(&calls),
        ["metadata", "captions", "metadata", "captions"]
    );
}

#[test]
fn explicit_language_mismatch_and_rate_limits_do_not_trigger_audio_fallback() {
    let root = tempfile::tempdir().unwrap();
    let metadata = r#"{"id":"dQw4w9WgXcQ","subtitles":{"de":[{"ext":"vtt","url":"https://example.test/de.vtt"}]},"automatic_captions":{}}"#;
    let (executable, calls) = fake_downloader(
        root.path(),
        metadata,
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHallo\n",
        "printf 'audio\\n' >> /dev/null; exit 90",
    );
    let mismatch = run(root.path(), &executable, &["--language", "en"]);
    assert!(!mismatch.status.success());
    assert!(String::from_utf8_lossy(&mismatch.stderr).contains("requested_language_unavailable"));
    assert_eq!(call_log(&calls), ["metadata"]);

    let rate_root = tempfile::tempdir().unwrap();
    let rate_script = rate_root.path().join("fake-yt-dlp");
    let rate_calls = rate_root.path().join("calls.log");
    std::fs::write(
        &rate_script,
        format!(
            "#!/bin/sh\nprintf 'metadata\\n' >> {}\nprintf 'ERROR: HTTP Error 429: Too Many Requests\\n' >&2\nexit 1\n",
            quote(&rate_calls)
        ),
    )
    .unwrap();
    std::fs::set_permissions(&rate_script, std::fs::Permissions::from_mode(0o700)).unwrap();
    let limited = run(rate_root.path(), &rate_script, &[]);
    assert!(!limited.status.success());
    assert!(String::from_utf8_lossy(&limited.stderr).contains("rate_limited"));
    assert_eq!(call_log(&rate_calls), ["metadata"]);
}

#[test]
fn malformed_subtitle_data_is_not_reinterpreted_as_missing_captions() {
    let root = tempfile::tempdir().unwrap();
    let metadata = r#"{"id":"dQw4w9WgXcQ","subtitles":{"en":[{"ext":"vtt","url":"https://example.test/en.vtt"}]},"automatic_captions":{}}"#;
    let (executable, calls) = fake_downloader(
        root.path(),
        metadata,
        "This is not a WebVTT document\n",
        "printf 'audio\\n' >> /dev/null; exit 90",
    );
    let result = run(root.path(), &executable, &["--language", "en"]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("malformed_subtitle"));
    assert_eq!(call_log(&calls), ["metadata", "captions"]);
}

#[test]
fn captions_only_fails_without_trying_local_audio() {
    let root = tempfile::tempdir().unwrap();
    let metadata = r#"{"id":"dQw4w9WgXcQ","subtitles":{},"automatic_captions":{}}"#;
    let (executable, calls) = fake_downloader(
        root.path(),
        metadata,
        "",
        "printf 'audio\\n' >> /dev/null; exit 90",
    );
    let result = run(root.path(), &executable, &["--captions-only"]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("no_captions"));
    assert_eq!(call_log(&calls), ["metadata"]);
}

#[test]
fn malformed_caption_metadata_must_not_trigger_audio_fallback() {
    let root = tempfile::tempdir().unwrap();
    let metadata = r#"{"id":"dQw4w9WgXcQ","subtitles":null,"automatic_captions":{}}"#;
    let (executable, calls) = fake_downloader(
        root.path(),
        metadata,
        "",
        "printf 'audio\\n' >> 'CALL_LOG'; exit 90",
    );
    let script = std::fs::read_to_string(&executable)
        .unwrap()
        .replace("CALL_LOG", &quote(&calls));
    std::fs::write(&executable, script).unwrap();
    let model = root.path().join("fake-model.bin");
    std::fs::write(&model, b"not a model").unwrap();

    let result = Command::new(bin())
        .args(["youtube", URL])
        .env("POLYSCRIBE_YTDLP", &executable)
        .env("XDG_CACHE_HOME", root.path().join("cache"))
        .env("HOME", root.path())
        .env("WHISPER_MODEL", &model)
        .output()
        .expect("launch polyscribe");
    assert!(!result.status.success());
    assert_eq!(
        call_log(&calls),
        ["metadata"],
        "invalid caption metadata was mistaken for confirmed caption absence: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn confirmed_caption_absence_allows_audio_fallback_with_language_hint() {
    let root = tempfile::tempdir().unwrap();
    let metadata = r#"{"id":"dQw4w9WgXcQ","subtitles":{},"automatic_captions":{}}"#;
    let (executable, calls) = fake_downloader(
        root.path(),
        metadata,
        "",
        "printf 'audio\\n' >> 'CALL_LOG'; exit 90",
    );
    let script = std::fs::read_to_string(&executable)
        .unwrap()
        .replace("CALL_LOG", &quote(&calls));
    std::fs::write(&executable, script).unwrap();
    let model = root.path().join("fake-model.bin");
    std::fs::write(&model, b"not a model").unwrap();

    let result = Command::new(bin())
        .args(["youtube", URL, "--language", "en"])
        .env("POLYSCRIBE_YTDLP", &executable)
        .env("XDG_CACHE_HOME", root.path().join("cache"))
        .env("HOME", root.path())
        .env("WHISPER_MODEL", &model)
        .output()
        .expect("launch polyscribe");
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("audio_download"),
        "unexpected error: {stderr}"
    );
    assert!(!stderr.contains("requested_language_unavailable"));
    assert_eq!(call_log(&calls), ["metadata", "audio"]);
}

#[test]
fn unsupported_caption_formats_do_not_trigger_audio_fallback() {
    let root = tempfile::tempdir().unwrap();
    let metadata = r#"{"id":"dQw4w9WgXcQ","subtitles":{"en":[{"ext":"srv3","url":"https://example.test/en.xml"}]},"automatic_captions":{}}"#;
    let (executable, calls) = fake_downloader(
        root.path(),
        metadata,
        "",
        "printf 'unexpected audio\\n' >> /dev/null; exit 90",
    );
    let result = run(root.path(), &executable, &[]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("unsupported_caption_format"));
    assert_eq!(call_log(&calls), ["metadata"]);
}

#[test]
fn missing_selected_caption_artifact_allows_audio_fallback() {
    let root = tempfile::tempdir().unwrap();
    let executable = root.path().join("fake-yt-dlp");
    let calls = root.path().join("calls.log");
    let model = root.path().join("fake-model.bin");
    std::fs::write(&model, b"not a model").unwrap();
    std::fs::write(
        &executable,
        format!(
            "#!/bin/sh
case \"$*\" in
  *--dump-single-json*)
    printf 'metadata\\n' >> {calls}
    printf '%s\\n' '{{\"id\":\"dQw4w9WgXcQ\",\"subtitles\":{{\"en\":[{{\"ext\":\"vtt\",\"url\":\"https://example.test/en.vtt\"}}]}},\"automatic_captions\":{{}}}}'
    ;;
  *--write-subs*|*--write-auto-subs*)
    printf 'captions\\n' >> {calls}
    exit 0
    ;;
  *)
    printf 'audio\\n' >> {calls}
    exit 90
    ;;
esac
",
            calls = quote(&calls),
        ),
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();

    let result = Command::new(bin())
        .args(["youtube", URL, "--language", "en"])
        .env("POLYSCRIBE_YTDLP", &executable)
        .env("XDG_CACHE_HOME", root.path().join("cache"))
        .env("HOME", root.path())
        .env("WHISPER_MODEL", &model)
        .output()
        .expect("launch polyscribe");
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("audio_download"));
    assert_eq!(call_log(&calls), ["metadata", "captions", "audio"]);
}

#[test]
fn caption_http_and_network_failures_do_not_trigger_audio_fallback() {
    for (failure, expected_kind) in [
        ("ERROR: HTTP Error 403: Forbidden", "forbidden"),
        ("Unable to download webpage: timed out", "network"),
    ] {
        let root = tempfile::tempdir().unwrap();
        let executable = root.path().join("fake-yt-dlp");
        let calls = root.path().join("calls.log");
        std::fs::write(
            &executable,
            format!(
                "#!/bin/sh
case \"$*\" in
  *--dump-single-json*)
    printf 'metadata\\n' >> {calls}
    printf '%s\\n' '{{\"id\":\"dQw4w9WgXcQ\",\"subtitles\":{{\"en\":[{{\"ext\":\"vtt\",\"url\":\"https://example.test/en.vtt\"}}]}},\"automatic_captions\":{{}}}}'
    ;;
  *--write-subs*|*--write-auto-subs*)
    printf 'captions\\n' >> {calls}
    printf '%s\\n' '{failure}' >&2
    exit 1
    ;;
  *)
    printf 'audio\\n' >> {calls}
    exit 0
    ;;
esac
",
                calls = quote(&calls),
            ),
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();

        let result = run(root.path(), &executable, &["--language", "en"]);
        assert!(!result.status.success());
        assert!(
            String::from_utf8_lossy(&result.stderr).contains(expected_kind),
            "expected {expected_kind}, got {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(call_log(&calls), ["metadata", "captions"]);
    }
}

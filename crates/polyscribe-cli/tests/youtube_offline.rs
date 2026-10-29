// SPDX-License-Identifier: MIT

//! Offline end-to-end checks for the YouTube audio transcription flow.

#![cfg(unix)]

use std::ffi::OsStr;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const URL: &str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ";
const METADATA_WITH_CAPTIONS: &str = r#"{"id":"dQw4w9WgXcQ","title":"Fixture video","uploader":"Offline","duration":3,"subtitles":{"en":[{"ext":"vtt","url":"https://example.test/en.vtt","name":"English"}]},"automatic_captions":{"de":[{"ext":"vtt","url":"https://example.test/de.vtt"} ]}}"#;
const VOXTRAL_RESPONSE: &str = r#"{"protocol_version":1,"engine":"voxtral","model":"fake-model","language":"en","cues":[{"start_ms":0,"end_ms":1200,"text":"hello"}]}"#;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_polyscribe")
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn quote(path: &Path) -> String {
    shell_quote(&path.to_string_lossy())
}

fn write_executable(path: &Path, contents: &str) {
    std::fs::write(path, contents).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
}

fn fake_downloader(root: &Path, metadata: &str, audio_failure: Option<&str>) -> (PathBuf, PathBuf) {
    let executable = root.join("fake-yt-dlp");
    let calls = root.join("yt-dlp-calls.log");
    let audio_action = if let Some(message) = audio_failure {
        format!("printf '%s\\n' {} >&2\nexit 1\n", shell_quote(message))
    } else {
        r#"
output=''
want_output=0
for arg do
  if [ "$want_output" -eq 1 ]; then
    output=$arg
    want_output=0
    continue
  fi
  if [ "$arg" = '--output' ]; then want_output=1; fi
done
case "$output" in
  *'.%(ext)s') audio_file="${output%.%(ext)s}.wav" ;;
  *) printf '%s\n' 'missing audio output template' >&2; exit 90 ;;
esac
printf '%s\n' 'fake audio data' > "$audio_file"
"#
        .to_string()
    };
    let script = r#"#!/bin/sh
set -eu
phase=audio
for arg do
  if [ "$arg" = '--dump-single-json' ]; then phase=metadata; fi
done
printf '%s' "$phase" >> @CALLS@
for arg do printf '\t%s' "$arg" >> @CALLS@; done
printf '\n' >> @CALLS@
if [ "$phase" = metadata ]; then
  printf '%s\n' @METADATA@
  exit 0
fi
@AUDIO_ACTION@
"#
    .replace("@CALLS@", &quote(&calls))
    .replace("@METADATA@", &shell_quote(metadata))
    .replace("@AUDIO_ACTION@", &audio_action);
    write_executable(&executable, &script);
    (executable, calls)
}

fn fake_voxtral(root: &Path, succeeds: bool) -> (PathBuf, PathBuf) {
    fake_voxtral_for(root, "fake-voxtral", succeeds, VOXTRAL_RESPONSE)
}

fn fake_voxtral_for(
    root: &Path,
    executable_name: &str,
    succeeds: bool,
    response: &str,
) -> (PathBuf, PathBuf) {
    let executable = root.join(executable_name);
    let calls = root.join("voxtral-calls.log");
    let action = if succeeds {
        format!(
            "printf '%s\\n' {}\nprintf '%s\\n' 'fake Voxtral diagnostic' >&2\n",
            shell_quote(response)
        )
    } else {
        "printf '%s\\n' 'fake Voxtral diagnostic before failure' >&2\nexit 23\n".to_string()
    };
    let script = r#"#!/bin/sh
set -eu
printf '%s' 'invoke' >> @CALLS@
for arg do printf '\t%s' "$arg" >> @CALLS@; done
printf '\n' >> @CALLS@
version=''
input=''
language=''
while [ $# -gt 0 ]; do
  case "$1" in
    --protocol-version)
      [ $# -ge 2 ] || exit 64
      version=$2
      shift 2
      ;;
    --input)
      [ $# -ge 2 ] || exit 64
      input=$2
      shift 2
      ;;
    --language)
      [ $# -ge 2 ] || exit 64
      language=$2
      shift 2
      ;;
    *) printf '%s\n' 'unexpected Voxtral argument' >&2; exit 64 ;;
  esac
done
if [ "$version" != '1' ] || [ -z "$input" ] || [ ! -s "$input" ]; then
  printf '%s\n' 'invalid Voxtral invocation or missing audio' >&2
  exit 65
fi
if [ -n "$language" ] && [ "$language" != 'en' ]; then
  printf '%s\n' 'unexpected language hint' >&2
  exit 66
fi
if IFS= read -r stdin_line; then
  printf '%s\n' 'Voxtral stdin was not closed' >&2
  exit 67
fi
@ACTION@
"#
    .replace("@CALLS@", &quote(&calls))
    .replace("@ACTION@", &action);
    write_executable(&executable, &script);
    (executable, calls)
}

fn run_with_voxtral(
    root: &Path,
    downloader: &Path,
    args: &[&str],
    voxtral: Option<&Path>,
) -> Output {
    run_with_voxtral_url(root, downloader, URL, args, voxtral)
}

fn run_with_voxtral_url(
    root: &Path,
    downloader: &Path,
    url: &str,
    args: &[&str],
    voxtral: Option<&Path>,
) -> Output {
    let temp = root.join("tmp");
    std::fs::create_dir_all(&temp).unwrap();
    let mut command = Command::new(bin());
    command
        .args(["youtube", url])
        .args(args)
        .current_dir(root)
        .env("POLYSCRIBE_YTDLP", downloader)
        .env("HOME", root)
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("TMPDIR", temp)
        .env_remove("POLYSCRIBE_VOXTRAL")
        .env_remove("POLYSCRIBE_MODELS_DIR")
        .env_remove("WHISPER_MODEL");
    if let Some(voxtral) = voxtral {
        command.env("POLYSCRIBE_VOXTRAL", voxtral);
    }
    command.output().expect("launch polyscribe")
}

#[derive(Debug, PartialEq, Eq)]
struct ProcessCall {
    phase: String,
    args: Vec<String>,
}

fn log_records(path: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .map(|line| line.split('\t').map(ToString::to_string).collect())
        .collect()
}

fn downloader_calls(path: &Path) -> Vec<ProcessCall> {
    log_records(path)
        .into_iter()
        .filter_map(|mut record| {
            if record.is_empty() {
                return None;
            }
            Some(ProcessCall {
                phase: record.remove(0),
                args: record,
            })
        })
        .collect()
}

fn assert_no_caption_arguments(calls: &[ProcessCall]) {
    for call in calls {
        assert!(
            call.args.iter().all(|argument| !matches!(
                argument.as_str(),
                "--write-subs" | "--write-auto-subs" | "--sub-langs"
            )),
            "caption download arguments were requested: {call:?}"
        );
    }
}

fn argument_after<'a>(call: &'a ProcessCall, option: &str) -> &'a str {
    let index = call
        .args
        .iter()
        .position(|argument| argument == option)
        .unwrap_or_else(|| panic!("{option} missing from {call:?}"));
    call.args
        .get(index + 1)
        .unwrap_or_else(|| panic!("value for {option} missing from {call:?}"))
}

fn assert_metadata_then_audio(calls: &[ProcessCall]) -> PathBuf {
    assert_eq!(
        calls
            .iter()
            .map(|call| call.phase.as_str())
            .collect::<Vec<_>>(),
        ["metadata", "audio"]
    );
    assert_no_caption_arguments(calls);
    assert!(
        calls[0]
            .args
            .iter()
            .any(|argument| argument == "--dump-single-json")
    );
    assert!(
        calls[0]
            .args
            .iter()
            .any(|argument| argument == "--skip-download")
    );
    assert_eq!(argument_after(&calls[1], "--format"), "bestaudio/best");
    for option in ["--retries", "--fragment-retries", "--extractor-retries"] {
        assert_eq!(argument_after(&calls[0], option), "0");
        assert_eq!(argument_after(&calls[1], option), "0");
    }
    let template = PathBuf::from(argument_after(&calls[1], "--output"));
    assert_eq!(template.file_name(), Some(OsStr::new("audio.%(ext)s")));
    template.with_file_name("audio.wav")
}

fn voxtral_input_path(record: &[String], language: Option<&str>) -> PathBuf {
    assert_eq!(record.first().map(String::as_str), Some("invoke"));
    assert_eq!(
        record.get(1).map(String::as_str),
        Some("--protocol-version")
    );
    assert_eq!(record.get(2).map(String::as_str), Some("1"));
    assert_eq!(record.get(3).map(String::as_str), Some("--input"));
    let input = PathBuf::from(record.get(4).expect("Voxtral input path"));
    assert!(input.is_absolute(), "Voxtral input path must be absolute");
    if let Some(language) = language {
        assert_eq!(
            record.get(5..),
            Some(
                ["--language", language]
                    .into_iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .as_slice()
            )
        );
    } else {
        assert_eq!(record.len(), 5);
    }
    input
}

#[test]
fn default_and_audio_only_download_bestaudio_without_caption_requests() {
    for legacy_flag in [None, Some("--audio-only"), Some("--prefer-manual")] {
        let root = tempfile::tempdir().unwrap();
        let (downloader, downloader_log) =
            fake_downloader(root.path(), METADATA_WITH_CAPTIONS, None);
        let (voxtral, voxtral_log) = fake_voxtral(root.path(), true);
        let mut args = vec!["--language", "en", "--format", "txt"];
        if let Some(flag) = legacy_flag {
            args.push(flag);
        }

        let result = run_with_voxtral(root.path(), &downloader, &args, Some(&voxtral));
        assert!(
            result.status.success(),
            "audio mode failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );

        let calls = downloader_calls(&downloader_log);
        let audio_path = assert_metadata_then_audio(&calls);
        let voxtral_calls = log_records(&voxtral_log);
        assert_eq!(voxtral_calls.len(), 1);
        assert_eq!(
            voxtral_input_path(&voxtral_calls[0], Some("en")),
            audio_path
        );
        assert!(
            !audio_path.exists(),
            "temporary audio must be removed after exit"
        );
    }
}

#[test]
fn voxtral_success_emits_json_only_and_diagnostics_on_stderr() {
    let root = tempfile::tempdir().unwrap();
    let (downloader, downloader_log) = fake_downloader(root.path(), METADATA_WITH_CAPTIONS, None);
    let (voxtral, voxtral_log) = fake_voxtral(root.path(), true);

    let result = run_with_voxtral_url(
        root.path(),
        &downloader,
        "https://youtu.be/dQw4w9WgXcQ",
        &["--language", "en", "--format", "json"],
        Some(&voxtral),
    );
    assert!(
        result.status.success(),
        "Voxtral command failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let transcript: serde_json::Value = serde_json::from_slice(&result.stdout)
        .expect("stdout should contain only the JSON transcript");
    assert_eq!(transcript["source"], "local-voxtral");
    assert_eq!(transcript["source_url"], "https://youtu.be/dQw4w9WgXcQ");
    assert_eq!(transcript["video_id"], "dQw4w9WgXcQ");
    assert_eq!(transcript["title"], "Fixture video");
    assert_eq!(transcript["uploader"], "Offline");
    assert_eq!(transcript["duration_seconds"], 3.0);
    assert_eq!(transcript["engine"], "voxtral");
    assert_eq!(transcript["model"], "fake-model");
    assert_eq!(transcript["language"], "en");
    assert_eq!(transcript["requested_language"], "en");
    assert_eq!(transcript["cues"][0]["text"], "hello");
    assert_eq!(transcript["cues"][0]["start_ms"], 0);
    assert_eq!(transcript["cues"][0]["end_ms"], 1200);

    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("source: local-voxtral"), "stderr: {stderr}");
    let audio_path = assert_metadata_then_audio(&downloader_calls(&downloader_log));
    let voxtral_calls = log_records(&voxtral_log);
    assert_eq!(voxtral_calls.len(), 1);
    assert_eq!(
        voxtral_input_path(&voxtral_calls[0], Some("en")),
        audio_path
    );
    assert!(
        !audio_path.exists(),
        "temporary audio must be removed after exit"
    );
}

#[test]
fn cache_hit_reuses_transcript_across_formats_without_running_processes() {
    let root = tempfile::tempdir().unwrap();
    let (downloader, downloader_log) = fake_downloader(root.path(), METADATA_WITH_CAPTIONS, None);
    let (voxtral, voxtral_log) = fake_voxtral(root.path(), true);

    let first = run_with_voxtral(
        root.path(),
        &downloader,
        &["--language", "en", "--format", "txt"],
        Some(&voxtral),
    );
    assert!(
        first.status.success(),
        "first run failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&first.stdout).trim(), "hello");
    let audio_path = assert_metadata_then_audio(&downloader_calls(&downloader_log));
    assert!(
        !audio_path.exists(),
        "temporary audio must be removed after exit"
    );

    let cached = run_with_voxtral(
        root.path(),
        &downloader,
        &["--language", "en", "--format", "srt"],
        Some(&voxtral),
    );
    assert!(
        cached.status.success(),
        "cached SRT run failed: {}",
        String::from_utf8_lossy(&cached.stderr)
    );
    assert!(String::from_utf8_lossy(&cached.stdout).contains("00:00:00,000 --> 00:00:01,200"));
    assert!(String::from_utf8_lossy(&cached.stdout).contains("hello"));

    let cached_json = run_with_voxtral(
        root.path(),
        &downloader,
        &["--language", "en", "--format", "json"],
        Some(&voxtral),
    );
    assert!(
        cached_json.status.success(),
        "cached JSON run failed: {}",
        String::from_utf8_lossy(&cached_json.stderr)
    );
    let transcript: serde_json::Value = serde_json::from_slice(&cached_json.stdout).unwrap();
    assert_eq!(transcript["source"], "local-voxtral");
    assert_eq!(transcript["engine"], "voxtral");
    assert_eq!(transcript["cues"][0]["text"], "hello");
    assert_eq!(downloader_calls(&downloader_log).len(), 2);
    assert_eq!(log_records(&voxtral_log).len(), 1);
}

#[test]
fn replacing_voxtral_at_the_same_path_invalidates_its_cache_entry() {
    let root = tempfile::tempdir().unwrap();
    let (downloader, downloader_log) = fake_downloader(root.path(), METADATA_WITH_CAPTIONS, None);
    let first_response = VOXTRAL_RESPONSE.replace("fake-model", "model-aa");
    let (voxtral, voxtral_log) =
        fake_voxtral_for(root.path(), "fake-voxtral", true, &first_response);

    let first = run_with_voxtral(
        root.path(),
        &downloader,
        &["--language", "en", "--format", "json"],
        Some(&voxtral),
    );
    assert!(first.status.success());
    let first_transcript: serde_json::Value = serde_json::from_slice(&first.stdout).unwrap();
    assert_eq!(first_transcript["model"], "model-aa");

    let replacement_response = VOXTRAL_RESPONSE.replace("fake-model", "model-bb");
    let (replacement, _) = fake_voxtral_for(
        root.path(),
        "replacement-voxtral",
        true,
        &replacement_response,
    );
    std::fs::rename(replacement, &voxtral).unwrap();

    let second = run_with_voxtral(
        root.path(),
        &downloader,
        &["--language", "en", "--format", "json"],
        Some(&voxtral),
    );
    assert!(
        second.status.success(),
        "replacement run failed: {}",
        String::from_utf8_lossy(&second.stderr)
    );
    let second_transcript: serde_json::Value = serde_json::from_slice(&second.stdout).unwrap();
    assert_eq!(second_transcript["model"], "model-bb");
    assert_eq!(downloader_calls(&downloader_log).len(), 4);
    assert_eq!(log_records(&voxtral_log).len(), 2);
}

#[test]
fn configuration_change_during_voxtral_run_skips_cache_write() {
    let root = tempfile::tempdir().unwrap();
    let (downloader, downloader_log) = fake_downloader(root.path(), METADATA_WITH_CAPTIONS, None);
    let voxtral = root.path().join("fake-voxtral");
    let replacement = root.path().join("replacement-voxtral");
    let voxtral_log = root.path().join("voxtral-calls.log");
    let first_response = VOXTRAL_RESPONSE.replace("fake-model", "model-aa");
    let second_response = VOXTRAL_RESPONSE.replace("fake-model", "model-bb");
    let replacement_script = format!(
        "#!/bin/sh\nset -eu\nprintf '%s\\n' 'invoke' >> {}\nprintf '%s\\n' {}\n",
        quote(&voxtral_log),
        shell_quote(&second_response)
    );
    write_executable(&replacement, &replacement_script);
    let first_script = format!(
        "#!/bin/sh\nset -eu\nprintf '%s\\n' 'invoke' >> {}\nmv {} \"$0\"\nprintf '%s\\n' {}\n",
        quote(&voxtral_log),
        quote(&replacement),
        shell_quote(&first_response)
    );
    write_executable(&voxtral, &first_script);

    let first = run_with_voxtral(
        root.path(),
        &downloader,
        &["--language", "en", "--format", "json"],
        Some(&voxtral),
    );
    assert!(
        first.status.success(),
        "changed-config run failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    let first_transcript: serde_json::Value = serde_json::from_slice(&first.stdout).unwrap();
    assert_eq!(first_transcript["model"], "model-aa");
    assert_eq!(cache_json_count(root.path()), 0);

    let second = run_with_voxtral(
        root.path(),
        &downloader,
        &["--language", "en", "--format", "json"],
        Some(&voxtral),
    );
    assert!(second.status.success());
    let second_transcript: serde_json::Value = serde_json::from_slice(&second.stdout).unwrap();
    assert_eq!(second_transcript["model"], "model-bb");
    assert_eq!(cache_json_count(root.path()), 2);

    let cached = run_with_voxtral(
        root.path(),
        &downloader,
        &["--language", "en", "--format", "txt"],
        Some(&voxtral),
    );
    assert!(cached.status.success());
    assert_eq!(downloader_calls(&downloader_log).len(), 4);
    assert_eq!(log_records(&voxtral_log).len(), 2);
}

#[test]
fn restored_voxtral_symlink_change_skips_cache_write() {
    let root = tempfile::tempdir().unwrap();
    let (base_downloader, downloader_log) =
        fake_downloader(root.path(), METADATA_WITH_CAPTIONS, None);
    let original_response = VOXTRAL_RESPONSE.replace("fake-model", "model-aa");
    let (original, voxtral_log) =
        fake_voxtral_for(root.path(), "original-voxtral", true, &original_response);
    let symlink = root.path().join("configured-voxtral");
    let saved_symlink = root.path().join("saved-voxtral-link");
    let attacker = root.path().join("replacement-voxtral");
    let marker = root.path().join("replacement-installed");
    std::os::unix::fs::symlink(&original, &symlink).unwrap();

    let attacker_response = VOXTRAL_RESPONSE.replace("fake-model", "model-bb");
    let attacker_script = format!(
        "#!/bin/sh\nset -eu\nprintf '%s\\n' 'invoke' >> {}\ninput=''\nwhile [ $# -gt 0 ]; do\n  case \"$1\" in\n    --input) input=$2; shift 2 ;;\n    *) shift ;;\n  esac\ndone\n[ -s \"$input\" ]\nrm {}\nmv {} {}\nprintf '%s\\n' {}\n",
        quote(&voxtral_log),
        quote(&symlink),
        quote(&saved_symlink),
        quote(&symlink),
        shell_quote(&attacker_response)
    );
    write_executable(&attacker, &attacker_script);

    let swapping_downloader = root.path().join("swapping-yt-dlp");
    let swapping_script = format!(
        "#!/bin/sh\nset -eu\nphase=audio\nfor arg do\n  if [ \"$arg\" = '--dump-single-json' ]; then phase=metadata; fi\ndone\nif [ \"$phase\" = audio ] && [ ! -e {} ]; then\n  mv {} {}\n  ln -s {} {}\n  : > {}\nfi\nexec {} \"$@\"\n",
        quote(&marker),
        quote(&symlink),
        quote(&saved_symlink),
        quote(&attacker),
        quote(&symlink),
        quote(&marker),
        quote(&base_downloader)
    );
    write_executable(&swapping_downloader, &swapping_script);

    let first = run_with_voxtral(
        root.path(),
        &swapping_downloader,
        &["--language", "en", "--format", "json"],
        Some(&symlink),
    );
    assert!(
        first.status.success(),
        "symlink swap run failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    let first_transcript: serde_json::Value = serde_json::from_slice(&first.stdout).unwrap();
    assert_eq!(first_transcript["model"], "model-bb");
    assert_eq!(cache_json_count(root.path()), 0);

    let second = run_with_voxtral(
        root.path(),
        &swapping_downloader,
        &["--language", "en", "--format", "json"],
        Some(&symlink),
    );
    assert!(second.status.success());
    let second_transcript: serde_json::Value = serde_json::from_slice(&second.stdout).unwrap();
    assert_eq!(second_transcript["model"], "model-aa");
    assert_eq!(cache_json_count(root.path()), 2);

    let cached = run_with_voxtral(
        root.path(),
        &swapping_downloader,
        &["--language", "en", "--format", "txt"],
        Some(&symlink),
    );
    assert!(cached.status.success());
    assert_eq!(downloader_calls(&downloader_log).len(), 4);
    assert_eq!(log_records(&voxtral_log).len(), 2);
}

#[test]
fn restored_parent_symlink_change_skips_cache_write() {
    let root = tempfile::tempdir().unwrap();
    let (base_downloader, downloader_log) =
        fake_downloader(root.path(), METADATA_WITH_CAPTIONS, None);
    let original_dir = root.path().join("original-bin");
    let replacement_dir = root.path().join("replacement-bin");
    std::fs::create_dir_all(&original_dir).unwrap();
    std::fs::create_dir_all(&replacement_dir).unwrap();
    let original_response = VOXTRAL_RESPONSE.replace("fake-model", "model-aa");
    let (_, voxtral_log) = fake_voxtral_for(&original_dir, "voxtral", true, &original_response);
    let replacement_response = VOXTRAL_RESPONSE.replace("fake-model", "model-bb");
    let configured_dir = root.path().join("current-bin");
    let saved_dir = root.path().join("saved-current-bin");
    let symlink_path = configured_dir.join("voxtral");
    let marker = root.path().join("parent-symlink-replaced");
    std::os::unix::fs::symlink(&original_dir, &configured_dir).unwrap();

    let swapping_downloader = root.path().join("swapping-parent-yt-dlp");
    let swapping_script = format!(
        "#!/bin/sh\nset -eu\nphase=audio\nfor arg do\n  if [ \"$arg\" = '--dump-single-json' ]; then phase=metadata; fi\ndone\nif [ \"$phase\" = audio ] && [ ! -e {} ]; then\n  mv {} {}\n  ln -s {} {}\n  : > {}\nfi\nexec {} \"$@\"\n",
        quote(&marker),
        quote(&configured_dir),
        quote(&saved_dir),
        quote(&replacement_dir),
        quote(&configured_dir),
        quote(&marker),
        quote(&base_downloader)
    );
    write_executable(&swapping_downloader, &swapping_script);

    let replacement_runner = replacement_dir.join("voxtral");
    let runner_script = format!(
        "#!/bin/sh\nset -eu\nprintf '%s\\n' 'invoke' >> {}\ninput=''\nwhile [ $# -gt 0 ]; do\n  case \"$1\" in\n    --input) input=$2; shift 2 ;;\n    *) shift ;;\n  esac\ndone\n[ -s \"$input\" ]\nrm {}\nmv {} {}\nprintf '%s\\n' {}\n",
        quote(&voxtral_log),
        quote(&configured_dir),
        quote(&saved_dir),
        quote(&configured_dir),
        shell_quote(&replacement_response)
    );
    write_executable(&replacement_runner, &runner_script);

    let first = run_with_voxtral(
        root.path(),
        &swapping_downloader,
        &["--language", "en", "--format", "json"],
        Some(&symlink_path),
    );
    assert!(
        first.status.success(),
        "parent symlink swap run failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    let first_transcript: serde_json::Value = serde_json::from_slice(&first.stdout).unwrap();
    assert_eq!(first_transcript["model"], "model-bb");
    assert_eq!(cache_json_count(root.path()), 0);

    let second = run_with_voxtral(
        root.path(),
        &swapping_downloader,
        &["--language", "en", "--format", "json"],
        Some(&symlink_path),
    );
    assert!(second.status.success());
    let second_transcript: serde_json::Value = serde_json::from_slice(&second.stdout).unwrap();
    assert_eq!(second_transcript["model"], "model-aa");
    assert_eq!(cache_json_count(root.path()), 2);

    let cached = run_with_voxtral(
        root.path(),
        &swapping_downloader,
        &["--language", "en", "--format", "txt"],
        Some(&symlink_path),
    );
    assert!(cached.status.success());
    assert_eq!(downloader_calls(&downloader_log).len(), 4);
    assert_eq!(log_records(&voxtral_log).len(), 2);
}

fn cache_json_count(root: &Path) -> usize {
    std::fs::read_dir(root.join("cache/polyscribe/youtube/v3"))
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension() == Some(OsStr::new("json")))
        .count()
}

#[test]
fn refresh_downloads_audio_and_runs_voxtral_again() {
    let root = tempfile::tempdir().unwrap();
    let (downloader, downloader_log) = fake_downloader(root.path(), METADATA_WITH_CAPTIONS, None);
    let (voxtral, voxtral_log) = fake_voxtral(root.path(), true);

    for args in [
        vec!["--language", "en", "--format", "txt"],
        vec!["--language", "en", "--format", "txt", "--refresh"],
    ] {
        let result = run_with_voxtral(root.path(), &downloader, &args, Some(&voxtral));
        assert!(
            result.status.success(),
            "refresh flow failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }

    let calls = downloader_calls(&downloader_log);
    assert_eq!(
        calls
            .iter()
            .map(|call| call.phase.as_str())
            .collect::<Vec<_>>(),
        ["metadata", "audio", "metadata", "audio"]
    );
    assert_no_caption_arguments(&calls);
    for audio_call in calls.iter().filter(|call| call.phase == "audio") {
        assert_eq!(argument_after(audio_call, "--format"), "bestaudio/best");
        let template = PathBuf::from(argument_after(audio_call, "--output"));
        assert_eq!(template.file_name(), Some(OsStr::new("audio.%(ext)s")));
        assert!(!template.with_file_name("audio.wav").exists());
    }
    let voxtral_calls = log_records(&voxtral_log);
    assert_eq!(voxtral_calls.len(), 2);
    for call in &voxtral_calls {
        let audio_path = voxtral_input_path(call, Some("en"));
        assert!(
            !audio_path.exists(),
            "temporary audio must be removed after exit"
        );
    }
}

#[test]
fn captions_only_fails_before_any_process_invocation() {
    let root = tempfile::tempdir().unwrap();
    let (downloader, downloader_log) = fake_downloader(root.path(), METADATA_WITH_CAPTIONS, None);
    let (voxtral, voxtral_log) = fake_voxtral(root.path(), true);

    let result = run_with_voxtral(
        root.path(),
        &downloader,
        &["--captions-only"],
        Some(&voxtral),
    );
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("unsupported_mode"));
    assert!(downloader_calls(&downloader_log).is_empty());
    assert!(log_records(&voxtral_log).is_empty());
}

#[test]
fn audio_rate_limit_is_classified_before_voxtral_can_run() {
    let root = tempfile::tempdir().unwrap();
    let (downloader, downloader_log) = fake_downloader(
        root.path(),
        METADATA_WITH_CAPTIONS,
        Some("ERROR: HTTP Error 429: Too Many Requests"),
    );
    let (voxtral, voxtral_log) = fake_voxtral(root.path(), true);

    let result = run_with_voxtral(
        root.path(),
        &downloader,
        &["--language", "en"],
        Some(&voxtral),
    );
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("rate_limited"), "stderr: {stderr}");
    assert!(!stderr.contains("audio_download"), "stderr: {stderr}");
    let calls = downloader_calls(&downloader_log);
    assert_eq!(
        calls
            .iter()
            .map(|call| call.phase.as_str())
            .collect::<Vec<_>>(),
        ["metadata", "audio"]
    );
    assert_no_caption_arguments(&calls);
    assert_eq!(argument_after(&calls[1], "--format"), "bestaudio/best");
    assert!(log_records(&voxtral_log).is_empty());
}

#[test]
fn voxtral_failure_tries_whisper_once_and_preserves_whisper_error_classification() {
    let root = tempfile::tempdir().unwrap();
    let (downloader, downloader_log) = fake_downloader(root.path(), METADATA_WITH_CAPTIONS, None);
    let (voxtral, voxtral_log) = fake_voxtral(root.path(), false);

    let result = run_with_voxtral(
        root.path(),
        &downloader,
        &["--language", "en"],
        Some(&voxtral),
    );
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("Voxtral failed"), "stderr: {stderr}");
    assert!(
        stderr.contains("local Whisper fallback failed"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("no local Whisper model is available"),
        "stderr: {stderr}"
    );
    assert!(stderr.contains("WHISPER_MODEL"), "stderr: {stderr}");
    assert!(!stderr.contains("audio_download"), "stderr: {stderr}");

    let calls = downloader_calls(&downloader_log);
    let audio_path = assert_metadata_then_audio(&calls);
    let voxtral_calls = log_records(&voxtral_log);
    assert_eq!(voxtral_calls.len(), 1, "Voxtral fallback must run once");
    assert_eq!(
        voxtral_input_path(&voxtral_calls[0], Some("en")),
        audio_path
    );
    assert!(
        !audio_path.exists(),
        "temporary audio must be removed after exit"
    );
}

#[test]
fn unconfigured_voxtral_tries_whisper_after_downloading_audio() {
    let root = tempfile::tempdir().unwrap();
    let (downloader, downloader_log) = fake_downloader(root.path(), METADATA_WITH_CAPTIONS, None);

    let result = run_with_voxtral(root.path(), &downloader, &["--language", "en"], None);
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("Voxtral is not configured"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("trying local Whisper once"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("local Whisper fallback failed"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("no local Whisper model is available"),
        "stderr: {stderr}"
    );
    let calls = downloader_calls(&downloader_log);
    let audio_path = assert_metadata_then_audio(&calls);
    assert!(
        !audio_path.exists(),
        "temporary audio must be removed after exit"
    );
}

#[test]
fn malformed_voxtral_output_falls_back_once_to_whisper() {
    let root = tempfile::tempdir().unwrap();
    let (downloader, downloader_log) = fake_downloader(root.path(), METADATA_WITH_CAPTIONS, None);
    let (voxtral, voxtral_log) =
        fake_voxtral_for(root.path(), "fake-voxtral", true, "not protocol json");

    let result = run_with_voxtral(
        root.path(),
        &downloader,
        &["--language", "en"],
        Some(&voxtral),
    );
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("valid protocol-v1 transcript"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("local Whisper fallback failed"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("no local Whisper model is available"),
        "stderr: {stderr}"
    );
    assert!(!stderr.contains("audio_download"), "stderr: {stderr}");
    assert_eq!(downloader_calls(&downloader_log).len(), 2);
    assert_eq!(log_records(&voxtral_log).len(), 1);
}

#[test]
fn unexpected_metadata_video_id_is_not_echoed_to_terminal() {
    let root = tempfile::tempdir().unwrap();
    let metadata = r#"{"id":"\u001b[31mnot-the-requested-video","title":"Fixture"}"#;
    let (downloader, downloader_log) = fake_downloader(root.path(), metadata, None);

    let result = run_with_voxtral(root.path(), &downloader, &[], None);
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("different video id"), "stderr: {stderr}");
    assert!(
        !stderr.contains('\u{1b}'),
        "terminal escape leaked to stderr"
    );
    assert_eq!(
        downloader_calls(&downloader_log)
            .iter()
            .map(|call| call.phase.as_str())
            .collect::<Vec<_>>(),
        ["metadata"]
    );
}

// SPDX-License-Identifier: MIT

use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use polyscribe_core::stream::CancellationToken;
use polyscribe_core::youtube::CaptionCue;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer};

const PROCESS_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const STDOUT_LIMIT: usize = 8 * 1024 * 1024;
const STDERR_LIMIT: usize = 32 * 1024;
const MAX_MODEL_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VoxtralError {
    NotConfigured,
    InvalidConfiguration,
    Spawn { kind: io::ErrorKind },
    Timeout,
    Cancelled,
    NonzeroExit { code: Option<i32> },
    StdoutLimit,
    MalformedOutput,
    EmptyOutput,
    Wait { kind: io::ErrorKind },
    Read { kind: io::ErrorKind },
    ReaderPanicked,
    WorkingDirectory,
}

impl std::fmt::Display for VoxtralError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotConfigured => {
                formatter.write_str("POLYSCRIBE_VOXTRAL is not set; local Whisper will be used")
            }
            Self::InvalidConfiguration => formatter.write_str(
                "POLYSCRIBE_VOXTRAL must name an absolute path to a local executable file",
            ),
            Self::Spawn { kind } => {
                write!(
                    formatter,
                    "could not start the local Voxtral executable ({kind:?}); check its path and execute permissions"
                )
            }
            Self::Timeout => {
                formatter.write_str("local Voxtral transcription exceeded its process deadline")
            }
            Self::Cancelled => formatter.write_str("local Voxtral transcription was cancelled"),
            Self::NonzeroExit { code: Some(code) } => {
                write!(formatter, "local Voxtral process exited with code {code}")
            }
            Self::NonzeroExit { code: None } => {
                formatter.write_str("local Voxtral process exited without an exit code")
            }
            Self::StdoutLimit => {
                formatter.write_str("local Voxtral output exceeded the 8 MiB size limit")
            }
            Self::MalformedOutput => formatter.write_str(
                "local Voxtral executable did not return a valid protocol-v1 transcript",
            ),
            Self::EmptyOutput => {
                formatter.write_str("local Voxtral executable returned no transcript output")
            }
            Self::Wait { kind } => {
                write!(
                    formatter,
                    "could not read the local Voxtral process status ({kind:?})"
                )
            }
            Self::Read { kind } => {
                write!(
                    formatter,
                    "could not read local Voxtral process output ({kind:?})"
                )
            }
            Self::ReaderPanicked => {
                formatter.write_str("a local Voxtral output reader stopped unexpectedly")
            }
            Self::WorkingDirectory => {
                formatter.write_str("could not resolve the working directory for local Voxtral")
            }
        }
    }
}

impl std::error::Error for VoxtralError {}

#[derive(Debug)]
pub(crate) struct VoxtralTranscript {
    pub model: String,
    pub language: Option<String>,
    pub cues: Vec<CaptionCue>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProtocolResponse {
    protocol_version: u32,
    engine: String,
    model: String,
    language: RequiredNullableString,
    cues: Vec<ProtocolCue>,
}

struct RequiredNullableString(Option<String>);

impl<'de> Deserialize<'de> for RequiredNullableString {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NullableStringVisitor;

        impl Visitor<'_> for NullableStringVisitor {
            type Value = RequiredNullableString;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a string or null")
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(RequiredNullableString(None))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(RequiredNullableString(Some(value.to_owned())))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(RequiredNullableString(Some(value)))
            }
        }

        deserializer.deserialize_any(NullableStringVisitor)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProtocolCue {
    start_ms: u64,
    end_ms: u64,
    text: String,
}

pub(crate) fn transcribe(
    executable: Option<OsString>,
    audio_path: &Path,
    language: Option<&str>,
    temp_dir: &Path,
    cancellation: &CancellationToken,
) -> Result<VoxtralTranscript, VoxtralError> {
    transcribe_with_timeout(
        executable,
        audio_path,
        language,
        temp_dir,
        cancellation,
        PROCESS_TIMEOUT,
    )
}

pub(crate) fn transcribe_with_timeout(
    executable: Option<OsString>,
    audio_path: &Path,
    language: Option<&str>,
    temp_dir: &Path,
    cancellation: &CancellationToken,
    timeout: Duration,
) -> Result<VoxtralTranscript, VoxtralError> {
    let executable = configured_executable(executable)?;
    transcribe_with_executable(
        &executable,
        audio_path,
        language,
        temp_dir,
        cancellation,
        timeout,
    )
}

fn transcribe_with_executable(
    executable: &Path,
    audio_path: &Path,
    language: Option<&str>,
    temp_dir: &Path,
    cancellation: &CancellationToken,
    timeout: Duration,
) -> Result<VoxtralTranscript, VoxtralError> {
    let arguments = command_arguments(audio_path, language)?;
    let environment = safe_environment(temp_dir);
    let output = run_executable(executable, &arguments, &environment, cancellation, timeout)?;
    if !output.status.success() {
        return Err(VoxtralError::NonzeroExit {
            code: output.status.code(),
        });
    }
    parse_output(&output.stdout)
}

fn configured_executable(value: Option<OsString>) -> Result<PathBuf, VoxtralError> {
    let Some(value) = value else {
        return Err(VoxtralError::NotConfigured);
    };
    let path = PathBuf::from(value);
    if !path.is_absolute() || !path.is_file() {
        return Err(VoxtralError::InvalidConfiguration);
    }
    Ok(path)
}

fn command_arguments(
    audio_path: &Path,
    language: Option<&str>,
) -> Result<Vec<OsString>, VoxtralError> {
    if language.is_some_and(|value| !valid_language_code(value)) {
        return Err(VoxtralError::InvalidConfiguration);
    }
    let absolute_audio_path = if audio_path.is_absolute() {
        audio_path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|_| VoxtralError::WorkingDirectory)?
            .join(audio_path)
    };
    let mut arguments = vec![
        OsString::from("--protocol-version"),
        OsString::from("1"),
        OsString::from("--input"),
        absolute_audio_path.into_os_string(),
    ];
    if let Some(language) = language {
        arguments.push(OsString::from("--language"));
        arguments.push(OsString::from(language));
    }
    Ok(arguments)
}

fn valid_language_code(value: &str) -> bool {
    crate::youtube_transcript::valid_language_code(value)
}

fn safe_environment(temp_dir: &Path) -> Vec<(OsString, OsString)> {
    environment_from(temp_dir, |key| std::env::var_os(key))
}

fn environment_from<F>(temp_dir: &Path, mut get: F) -> Vec<(OsString, OsString)>
where
    F: FnMut(&str) -> Option<OsString>,
{
    let mut environment = Vec::new();
    for key in [
        "PATH",
        "LANG",
        "LC_ALL",
        "HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
    ] {
        if let Some(value) = get(key) {
            environment.push((OsString::from(key), value));
        }
    }
    for key in ["TMPDIR", "TEMP", "TMP"] {
        environment.push((OsString::from(key), temp_dir.as_os_str().to_os_string()));
    }
    environment
}

fn run_executable(
    executable: &Path,
    arguments: &[OsString],
    environment: &[(OsString, OsString)],
    cancellation: &CancellationToken,
    timeout: Duration,
) -> Result<crate::youtube_process::ProcessOutput, VoxtralError> {
    crate::youtube_process::run(
        executable,
        arguments,
        environment,
        crate::youtube_process::ProcessPolicy {
            timeout,
            stdout_limit: STDOUT_LIMIT,
            stderr_limit: STDERR_LIMIT,
        },
        cancellation,
    )
    .map_err(|failure| match failure {
        crate::youtube_process::ProcessFailure::Spawn(error) => {
            VoxtralError::Spawn { kind: error.kind() }
        }
        crate::youtube_process::ProcessFailure::Timeout => VoxtralError::Timeout,
        crate::youtube_process::ProcessFailure::StdoutLimit => VoxtralError::StdoutLimit,
        crate::youtube_process::ProcessFailure::Cancelled => VoxtralError::Cancelled,
        crate::youtube_process::ProcessFailure::Wait(error) => {
            VoxtralError::Wait { kind: error.kind() }
        }
        crate::youtube_process::ProcessFailure::Read(error) => {
            VoxtralError::Read { kind: error.kind() }
        }
        crate::youtube_process::ProcessFailure::ReaderPanicked => VoxtralError::ReaderPanicked,
    })
}

fn parse_output(stdout: &[u8]) -> Result<VoxtralTranscript, VoxtralError> {
    if stdout.len() > STDOUT_LIMIT {
        return Err(VoxtralError::StdoutLimit);
    }
    if stdout.is_empty() {
        return Err(VoxtralError::EmptyOutput);
    }
    let response: ProtocolResponse =
        serde_json::from_slice(stdout).map_err(|_| VoxtralError::MalformedOutput)?;
    validate_response(response)
}

fn validate_response(response: ProtocolResponse) -> Result<VoxtralTranscript, VoxtralError> {
    if response.protocol_version != 1
        || response.engine != "voxtral"
        || response.model.trim().is_empty()
        || response.model.len() > MAX_MODEL_BYTES
        || response.model.chars().any(char::is_control)
    {
        return Err(VoxtralError::MalformedOutput);
    }
    let language = response.language.0;
    if language
        .as_deref()
        .is_some_and(|value| !valid_language_code(value))
    {
        return Err(VoxtralError::MalformedOutput);
    }
    if response.cues.is_empty() || response.cues.len() > crate::youtube_transcript::MAX_CUES {
        return Err(VoxtralError::MalformedOutput);
    }

    let mut cues = response
        .cues
        .into_iter()
        .map(|cue| CaptionCue {
            start_ms: cue.start_ms,
            end_ms: cue.end_ms,
            text: cue.text,
            speaker: None,
            speaker_marker: false,
        })
        .collect::<Vec<_>>();
    crate::youtube_transcript::normalize_and_validate_cues(&mut cues)
        .map_err(|_| VoxtralError::MalformedOutput)?;

    Ok(VoxtralTranscript {
        model: response.model,
        language,
        cues,
    })
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::os::unix::fs::PermissionsExt;

    fn script(directory: &Path, name: &str, body: &str) -> PathBuf {
        let path = directory.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    fn response(language: &str) -> String {
        format!(
            r#"{{"protocol_version":1,"engine":"voxtral","model":"local-voxtral-test","language":{language},"cues":[{{"start_ms":0,"end_ms":1200,"text":"hello"}}]}}"#
        )
    }

    fn invoke(
        executable: &Path,
        audio_path: &Path,
        language: Option<&str>,
        temp_dir: &Path,
    ) -> Result<VoxtralTranscript, VoxtralError> {
        invoke_with_timeout(executable, audio_path, language, temp_dir, PROCESS_TIMEOUT)
    }

    fn invoke_with_timeout(
        executable: &Path,
        audio_path: &Path,
        language: Option<&str>,
        temp_dir: &Path,
        timeout: Duration,
    ) -> Result<VoxtralTranscript, VoxtralError> {
        transcribe_with_executable(
            executable,
            audio_path,
            language,
            temp_dir,
            &CancellationToken::new(),
            timeout,
        )
    }

    fn audio_file(directory: &Path) -> PathBuf {
        let path = directory.join("audio.wav");
        std::fs::write(&path, b"audio bytes").unwrap();
        path
    }

    #[test]
    fn parses_success_and_preserves_reported_model_language_and_cues() {
        let directory = tempfile::tempdir().unwrap();
        let executable = script(
            directory.path(),
            "fake voxtral",
            &format!("printf '%s' '{}'", response("\"fr-CA\"")),
        );
        let audio = audio_file(directory.path());
        let transcript = invoke(&executable, &audio, Some("en-US"), directory.path()).unwrap();

        assert_eq!(transcript.model, "local-voxtral-test");
        assert_eq!(transcript.language.as_deref(), Some("fr-CA"));
        assert_eq!(transcript.cues.len(), 1);
        assert_eq!(transcript.cues[0].text, "hello");
        assert_eq!(transcript.cues[0].speaker, None);
        assert!(!transcript.cues[0].speaker_marker);
    }

    #[test]
    fn unset_executable_configuration_is_typed() {
        assert_eq!(
            configured_executable(None),
            Err(VoxtralError::NotConfigured)
        );
    }

    #[test]
    fn malformed_executable_configuration_is_typed() {
        let directory = tempfile::tempdir().unwrap();
        assert_eq!(
            configured_executable(Some(OsString::from("relative/executable"))),
            Err(VoxtralError::InvalidConfiguration)
        );
        assert_eq!(
            configured_executable(Some(directory.path().as_os_str().to_os_string())),
            Err(VoxtralError::InvalidConfiguration)
        );
    }

    #[test]
    fn nonzero_exit_wins_over_valid_json() {
        let directory = tempfile::tempdir().unwrap();
        let executable = script(
            directory.path(),
            "nonzero",
            &format!("printf '%s' '{}'; exit 19", response("null")),
        );
        let audio = audio_file(directory.path());
        assert_eq!(
            invoke(&executable, &audio, None, directory.path(),).unwrap_err(),
            VoxtralError::NonzeroExit { code: Some(19) }
        );
    }

    #[test]
    fn process_failure_diagnostics_are_typed_and_do_not_expose_child_stderr() {
        let directory = tempfile::tempdir().unwrap();
        let executable = script(
            directory.path(),
            "nonzero with private diagnostic",
            "printf '%s\\n' 'token=private-value' >&2; exit 19",
        );
        let audio = audio_file(directory.path());
        let error = invoke(&executable, &audio, None, directory.path()).unwrap_err();

        assert_eq!(error, VoxtralError::NonzeroExit { code: Some(19) });
        let diagnostic = error.to_string();
        assert!(diagnostic.contains("code 19"));
        assert!(!diagnostic.contains("private-value"));
    }

    #[test]
    fn process_timeout_is_reported_as_a_typed_failure() {
        let directory = tempfile::tempdir().unwrap();
        let executable = script(directory.path(), "timeout", "sleep 10");
        let audio = audio_file(directory.path());

        let error = invoke_with_timeout(
            &executable,
            &audio,
            None,
            directory.path(),
            Duration::from_millis(80),
        )
        .unwrap_err();
        assert_eq!(error, VoxtralError::Timeout);
        assert!(error.to_string().contains("process deadline"));
    }

    #[test]
    fn nullable_language_is_required_and_unknown_fields_are_rejected() {
        let valid = response("null");
        assert_eq!(parse_output(valid.as_bytes()).unwrap().language, None);
        let missing_language = valid.replace("\"language\":null,", "");
        assert_eq!(
            parse_output(missing_language.as_bytes()).unwrap_err(),
            VoxtralError::MalformedOutput
        );
        let unknown_field = valid.replace("\"cues\":", "\"unexpected\":true,\"cues\":");
        assert_eq!(
            parse_output(unknown_field.as_bytes()).unwrap_err(),
            VoxtralError::MalformedOutput
        );
        assert_eq!(
            parse_output(&[0xff]).unwrap_err(),
            VoxtralError::MalformedOutput
        );
    }

    #[test]
    fn blank_lines_in_cue_text_are_rejected() {
        let valid = response("null");
        let blank_line = valid.replace("\"text\":\"hello\"", "\"text\":\"hello\\n\\n2\"");
        assert_eq!(
            parse_output(blank_line.as_bytes()).unwrap_err(),
            VoxtralError::MalformedOutput
        );
    }

    #[test]
    fn harmless_trailing_line_endings_are_normalized() {
        let valid = response("null");
        for ending in ["\\n", "\\r", "\\r\\n", "\\n\\r\\n"] {
            let trailing =
                valid.replace("\"text\":\"hello\"", &format!("\"text\":\"hello{ending}\""));
            let parsed = parse_output(trailing.as_bytes());
            assert!(
                parsed.is_ok(),
                "trailing line ending {ending:?}: {parsed:?}"
            );
            assert_eq!(parsed.unwrap().cues[0].text, "hello");
        }
    }

    #[test]
    fn parser_enforces_the_same_raw_stdout_limit_as_the_process_supervisor() {
        let valid = response("null");
        let mut exactly_limit = valid.clone();
        exactly_limit.extend(std::iter::repeat_n(' ', STDOUT_LIMIT - exactly_limit.len()));
        assert_eq!(exactly_limit.len(), STDOUT_LIMIT);
        assert!(parse_output(exactly_limit.as_bytes()).is_ok());

        let mut too_large = valid;
        too_large.extend(std::iter::repeat_n(' ', STDOUT_LIMIT + 1 - too_large.len()));
        assert_eq!(too_large.len(), STDOUT_LIMIT + 1);
        assert_eq!(
            parse_output(too_large.as_bytes()).unwrap_err(),
            VoxtralError::StdoutLimit
        );
    }

    #[test]
    fn malformed_and_empty_output_have_distinct_typed_failures() {
        let directory = tempfile::tempdir().unwrap();
        let audio = audio_file(directory.path());
        for (name, body, expected) in [
            (
                "malformed",
                "printf '{not-json}'",
                VoxtralError::MalformedOutput,
            ),
            ("empty", ":", VoxtralError::EmptyOutput),
        ] {
            let executable = script(directory.path(), name, body);
            assert_eq!(
                invoke(&executable, &audio, None, directory.path(),).unwrap_err(),
                expected
            );
        }
    }

    #[test]
    fn arguments_preserve_executable_and_input_paths_literally() {
        let directory = tempfile::tempdir().unwrap();
        let executable = script(
            directory.path(),
            "fake executable with spaces",
            "printf '%s\\n' \"$@\"",
        );
        let audio = directory
            .path()
            .join("audio with spaces;$(printf INJECTED)*.wav");
        std::fs::write(&audio, b"audio bytes").unwrap();
        assert_eq!(
            configured_executable(Some(executable.as_os_str().to_os_string())).unwrap(),
            executable
        );
        let arguments = command_arguments(&audio, Some("en-US")).unwrap();
        let output = run_executable(
            &executable,
            &arguments,
            &safe_environment(directory.path()),
            &CancellationToken::new(),
            PROCESS_TIMEOUT,
        )
        .unwrap();

        assert!(output.status.success());
        let args = String::from_utf8(output.stdout).unwrap();
        assert_eq!(
            args.lines().collect::<Vec<_>>(),
            [
                "--protocol-version",
                "1",
                "--input",
                audio.to_str().unwrap(),
                "--language",
                "en-US"
            ]
        );
    }

    #[test]
    fn child_environment_contains_only_the_allowlist_and_private_temp_paths() {
        let directory = tempfile::tempdir().unwrap();
        let executable = Path::new("/usr/bin/env");
        let allowed = [
            "PATH",
            "LANG",
            "LC_ALL",
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_CACHE_HOME",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
        ];
        let mut source = HashMap::new();
        for key in allowed {
            source.insert(key, OsString::from(format!("allowed-{key}")));
        }
        source.insert("HTTPS_PROXY", OsString::from("https://proxy.invalid"));
        source.insert("http_proxy", OsString::from("http://proxy.invalid"));
        source.insert("POLYSCRIBE_VOXTRAL", OsString::from("/secret/wrapper"));
        source.insert("POLYSCRIBE_TOKEN", OsString::from("secret-token"));
        let environment = environment_from(directory.path(), |key| source.get(key).cloned());
        let output = run_executable(
            executable,
            &[],
            &environment,
            &CancellationToken::new(),
            PROCESS_TIMEOUT,
        )
        .unwrap();
        assert!(output.status.success());
        let reported: HashMap<String, String> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .filter_map(|line| {
                line.split_once('=')
                    .map(|(key, value)| (key.to_string(), value.to_string()))
            })
            .collect();

        assert!(reported.keys().all(|key| {
            allowed.contains(&key.as_str())
                || matches!(key.as_str(), "TMPDIR" | "TEMP" | "TMP" | "_")
        }));
        for key in allowed {
            let expected = format!("allowed-{key}");
            assert_eq!(
                reported.get(key).map(String::as_str),
                Some(expected.as_str())
            );
        }
        let temp_path = directory.path().to_str();
        assert_eq!(reported.get("TMPDIR").map(String::as_str), temp_path);
        assert_eq!(reported.get("TEMP").map(String::as_str), temp_path);
        assert_eq!(reported.get("TMP").map(String::as_str), temp_path);
        assert!(!reported.contains_key("HTTPS_PROXY"));
        assert!(!reported.contains_key("http_proxy"));
        assert!(!reported.contains_key("POLYSCRIBE_VOXTRAL"));
        assert!(!reported.contains_key("POLYSCRIBE_TOKEN"));
    }
}

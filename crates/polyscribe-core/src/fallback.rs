// SPDX-License-Identifier: MIT

//! A reusable cloud-to-local fallback wrapper for [`TranscribeBackend`].
//!
//! [`FallbackBackend`] tries a primary backend (typically the Mistral cloud
//! engine) and, when that fails, warns on stderr and continues with a
//! lazily built local backend (whisper, selected via [`select_backend`]).
//! The fallback is sticky: after the first error the rest of the run stays
//! on the fallback, so a failing cloud does not cause repeated slow
//! failures. The fallback only ever goes towards the local engine; there is
//! no fallback from a local engine to the cloud.

use std::path::Path;

use anyhow::anyhow;

use crate::OutputEntry;
use crate::backend::{BackendKind, TranscribeBackend, select_backend};
use crate::prelude::*;

/// Lazily builds the fallback backend on the first primary error.
pub type FallbackFactory =
    Box<dyn FnOnce() -> Result<Box<dyn TranscribeBackend + Send + Sync>> + Send + Sync>;

/// A backend that tries a primary backend first and falls back to a lazily
/// constructed one after the first primary error.
pub struct FallbackBackend {
    primary: Box<dyn TranscribeBackend + Send + Sync>,
    fallback_factory: Option<FallbackFactory>,
    fallback: Option<Box<dyn TranscribeBackend + Send + Sync>>,
    /// Receives the fallback warning (stderr in production, a recorder in
    /// tests).
    on_fallback: Box<dyn Fn(&str) + Send + Sync>,
}

impl FallbackBackend {
    /// Wrap `primary` with a fallback that `fallback_factory` builds lazily
    /// on the first primary error.
    pub fn new(
        primary: Box<dyn TranscribeBackend + Send + Sync>,
        fallback_factory: impl FnOnce() -> Result<Box<dyn TranscribeBackend + Send + Sync>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self::with_notifier(primary, fallback_factory, |msg| crate::ui::warn(msg))
    }

    /// A cloud primary with a local whisper fallback, selected the same way
    /// the `transcribe` command selects its whisper backend.
    pub fn with_local_whisper_fallback(
        primary: Box<dyn TranscribeBackend + Send + Sync>,
        requested: BackendKind,
        verbose: bool,
    ) -> Self {
        Self::new(primary, move || {
            Ok(select_backend(requested, verbose)?.backend)
        })
    }

    /// Like [`FallbackBackend::new`], with a custom sink for the fallback
    /// warning.
    pub fn with_notifier(
        primary: Box<dyn TranscribeBackend + Send + Sync>,
        fallback_factory: impl FnOnce() -> Result<Box<dyn TranscribeBackend + Send + Sync>>
        + Send
        + Sync
        + 'static,
        on_fallback: impl Fn(&str) + Send + Sync + 'static,
    ) -> Self {
        Self {
            primary,
            fallback_factory: Some(Box::new(fallback_factory)),
            fallback: None,
            on_fallback: Box::new(on_fallback),
        }
    }

    /// Whether the run already fell back to the local backend.
    pub fn fell_back(&self) -> bool {
        self.fallback.is_some()
    }
}

impl TranscribeBackend for FallbackBackend {
    fn kind(&self) -> BackendKind {
        match self.fallback.as_ref() {
            Some(fallback) => fallback.kind(),
            None => self.primary.kind(),
        }
    }

    fn transcribe(
        &mut self,
        audio_path: &Path,
        speaker: &str,
        language: Option<&str>,
        gpu_layers: Option<u32>,
        progress: Option<&(dyn Fn(i32) + Send + Sync)>,
    ) -> Result<Vec<OutputEntry>> {
        if let Some(fallback) = self.fallback.as_mut() {
            // Sticky: after the first error the rest of the run stays on the
            // fallback and the primary is not tried again.
            return fallback.transcribe(audio_path, speaker, language, gpu_layers, progress);
        }
        match self
            .primary
            .transcribe(audio_path, speaker, language, gpu_layers, progress)
        {
            Ok(entries) => Ok(entries),
            Err(err) => {
                let factory = self
                    .fallback_factory
                    .take()
                    .expect("the fallback factory is present until the first fallback");
                // The primary's error text is already key-scrubbed, so it can
                // be shown.
                let message = format!(
                    "The cloud transcription engine failed ({err:#}); \
                     falling back to the local whisper engine for the rest of this run."
                );
                (self.on_fallback)(&message);
                let mut fallback = match factory() {
                    Ok(fallback) => fallback,
                    Err(factory_err) => {
                        return Err(anyhow!(
                            "{message} The local fallback could not be started either: {factory_err:#}"
                        )
                        .into());
                    }
                };
                let result =
                    fallback.transcribe(audio_path, speaker, language, gpu_layers, progress);
                self.fallback = Some(fallback);
                result
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A fake backend that either fails with a fixed message or returns one
    /// entry carrying `marker` as its text. Counts its calls.
    struct FakeBackend {
        fail: bool,
        marker: &'static str,
        calls: Arc<Mutex<usize>>,
    }

    impl FakeBackend {
        fn failing(calls: Arc<Mutex<usize>>) -> Self {
            Self {
                fail: true,
                marker: "primary",
                calls,
            }
        }
        fn ok(marker: &'static str, calls: Arc<Mutex<usize>>) -> Self {
            Self {
                fail: false,
                marker,
                calls,
            }
        }
    }

    impl TranscribeBackend for FakeBackend {
        fn kind(&self) -> BackendKind {
            BackendKind::Cpu
        }
        fn transcribe(
            &mut self,
            _audio_path: &Path,
            speaker: &str,
            _language: Option<&str>,
            _gpu_layers: Option<u32>,
            _progress: Option<&(dyn Fn(i32) + Send + Sync)>,
        ) -> Result<Vec<OutputEntry>> {
            *self.calls.lock().unwrap() += 1;
            if self.fail {
                return Err(anyhow!("primary exploded").into());
            }
            Ok(vec![OutputEntry {
                id: 0,
                speaker: speaker.to_string(),
                start: 0.0,
                end: 1.0,
                text: self.marker.to_string(),
            }])
        }
    }

    #[test]
    fn test_fallback_used_after_primary_error_warning_and_sticky() {
        let primary_calls = Arc::new(Mutex::new(0usize));
        let fallback_calls = Arc::new(Mutex::new(0usize));
        let warnings: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let warnings_clone = warnings.clone();
        let fallback_calls_clone = fallback_calls.clone();
        let mut backend = FallbackBackend::with_notifier(
            Box::new(FakeBackend::failing(primary_calls.clone())),
            move || {
                Ok(Box::new(FakeBackend::ok("local", fallback_calls_clone))
                    as Box<dyn TranscribeBackend + Send + Sync>)
            },
            move |msg| warnings_clone.lock().unwrap().push(msg.to_string()),
        );

        let path = Path::new("input.wav");
        // First call: primary fails, fallback is built and used.
        let entries = backend.transcribe(path, "alice", None, None, None).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].text, "local");
        assert_eq!(entries[0].speaker, "alice");
        assert!(backend.fell_back());
        assert_eq!(*primary_calls.lock().unwrap(), 1);
        assert_eq!(*fallback_calls.lock().unwrap(), 1);

        // The warning came exactly once and names the (scrubbed) error.
        {
            let guard = warnings.lock().unwrap();
            assert_eq!(guard.len(), 1);
            assert!(
                guard[0].contains("falling back to the local whisper engine"),
                "{}",
                guard[0]
            );
            assert!(guard[0].contains("primary exploded"), "{}", guard[0]);
        }

        // Second call: sticky, the primary is not tried again.
        let entries = backend.transcribe(path, "bob", None, None, None).unwrap();
        assert_eq!(entries[0].text, "local");
        assert_eq!(*primary_calls.lock().unwrap(), 1);
        assert_eq!(*fallback_calls.lock().unwrap(), 2);
        assert_eq!(warnings.lock().unwrap().len(), 1);
    }

    #[test]
    fn test_no_fallback_when_primary_succeeds() {
        let primary_calls = Arc::new(Mutex::new(0usize));
        let warnings: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let warnings_clone = warnings.clone();
        let mut backend = FallbackBackend::with_notifier(
            Box::new(FakeBackend::ok("cloud", primary_calls.clone())),
            || -> Result<Box<dyn TranscribeBackend + Send + Sync>> {
                panic!("the fallback must not be built when the primary succeeds")
            },
            move |msg| warnings_clone.lock().unwrap().push(msg.to_string()),
        );
        let entries = backend
            .transcribe(Path::new("a.wav"), "s", None, None, None)
            .unwrap();
        assert_eq!(entries[0].text, "cloud");
        assert!(!backend.fell_back());
        assert_eq!(*primary_calls.lock().unwrap(), 1);
        assert!(warnings.lock().unwrap().is_empty());
    }

    #[test]
    fn test_factory_failure_reports_both_errors() {
        let warnings: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let warnings_clone = warnings.clone();
        let mut backend = FallbackBackend::with_notifier(
            Box::new(FakeBackend::failing(Arc::new(Mutex::new(0usize)))),
            || -> Result<Box<dyn TranscribeBackend + Send + Sync>> {
                Err(anyhow!("no whisper model found").into())
            },
            move |msg| warnings_clone.lock().unwrap().push(msg.to_string()),
        );
        let err = backend
            .transcribe(Path::new("a.wav"), "s", None, None, None)
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("primary exploded"), "{msg}");
        assert!(msg.contains("no whisper model found"), "{msg}");
        assert!(!backend.fell_back());
        // The warning was still emitted before the factory ran.
        assert_eq!(warnings.lock().unwrap().len(), 1);
    }

    #[test]
    fn test_fallback_error_propagates_and_stays_sticky() {
        let primary_calls = Arc::new(Mutex::new(0usize));
        let fallback_calls = Arc::new(Mutex::new(0usize));
        let fallback_calls_clone = fallback_calls.clone();
        let mut backend = FallbackBackend::new(
            Box::new(FakeBackend::failing(primary_calls.clone())),
            move || {
                Ok(Box::new(FakeBackend::failing(fallback_calls_clone))
                    as Box<dyn TranscribeBackend + Send + Sync>)
            },
        );
        let path = Path::new("a.wav");
        let err = backend.transcribe(path, "s", None, None, None).unwrap_err();
        assert!(format!("{err:#}").contains("primary exploded"));
        assert!(backend.fell_back());
        // Even though the fallback failed too, the next call goes straight to
        // the fallback (sticky), the primary is not retried.
        let err = backend.transcribe(path, "s", None, None, None).unwrap_err();
        assert!(format!("{err:#}").contains("primary exploded"));
        assert_eq!(*primary_calls.lock().unwrap(), 1);
        assert_eq!(*fallback_calls.lock().unwrap(), 2);
    }
}

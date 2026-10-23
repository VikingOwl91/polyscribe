// SPDX-License-Identifier: MIT

use std::ffi::{OsStr, OsString};
use std::io::{self, Read};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use polyscribe_core::stream::CancellationToken;

#[derive(Clone, Copy)]
pub(crate) struct ProcessPolicy {
    pub timeout: Duration,
    pub stdout_limit: usize,
    pub stderr_limit: usize,
}

pub(crate) struct ProcessOutput {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr_tail: Vec<u8>,
}

#[derive(Debug)]
pub(crate) enum ProcessFailure {
    Spawn(io::Error),
    Timeout,
    Cancelled,
    StdoutLimit,
    Wait(io::Error),
    Read(io::Error),
    ReaderPanicked,
}

pub(crate) fn run(
    program: &Path,
    args: &[OsString],
    env: &[(OsString, OsString)],
    policy: ProcessPolicy,
    cancellation: &CancellationToken,
) -> Result<ProcessOutput, ProcessFailure> {
    let mut command = Command::new(program);
    command
        .args(args)
        .env_clear()
        .envs(env.iter().map(|(key, value)| (key, value)))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
        #[cfg(target_os = "linux")]
        {
            let parent_pid = unsafe { libc::getpid() };
            unsafe {
                command.pre_exec(move || {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) == -1 {
                        return Err(io::Error::last_os_error());
                    }
                    if libc::getppid() != parent_pid {
                        libc::_exit(1);
                    }
                    Ok(())
                });
            }
        }
    }

    let child = command.spawn().map_err(ProcessFailure::Spawn)?;
    let mut child = ChildGuard::new(child);
    let stdout = child
        .child_mut()
        .stdout
        .take()
        .expect("stdout was configured as piped");
    let stderr = child
        .child_mut()
        .stderr
        .take()
        .expect("stderr was configured as piped");
    let stdout_overflow = Arc::new(AtomicBool::new(false));
    let stdout_overflow_reader = Arc::clone(&stdout_overflow);
    let stdout_limit = policy.stdout_limit;
    let stdout_reader = thread::spawn(move || {
        read_capture(stdout, stdout_limit, false, Some(stdout_overflow_reader))
    });
    let stderr_limit = policy.stderr_limit;
    let stderr_reader = thread::spawn(move || read_capture(stderr, stderr_limit, true, None));

    let deadline = Instant::now() + policy.timeout;
    let status = loop {
        if cancellation.is_cancelled() {
            child.terminate();
            join_reader(stdout_reader)?;
            join_reader(stderr_reader)?;
            return Err(ProcessFailure::Cancelled);
        }
        if stdout_overflow.load(Ordering::Relaxed) {
            child.terminate();
            join_reader(stdout_reader)?;
            join_reader(stderr_reader)?;
            return Err(ProcessFailure::StdoutLimit);
        }
        if stdout_reader.is_finished()
            && stderr_reader.is_finished()
            && let Some(status) = child.child_mut().try_wait().map_err(ProcessFailure::Wait)?
        {
            break status;
        }
        if Instant::now() >= deadline {
            child.terminate();
            join_reader(stdout_reader)?;
            join_reader(stderr_reader)?;
            return Err(ProcessFailure::Timeout);
        }
        thread::sleep(Duration::from_millis(10));
    };

    let stdout = join_reader(stdout_reader)?;
    let stderr_tail = join_reader(stderr_reader)?;
    child.disarm();
    Ok(ProcessOutput {
        status,
        stdout,
        stderr_tail,
    })
}

fn read_capture<R: Read>(
    mut reader: R,
    limit: usize,
    retain_tail: bool,
    overflow: Option<Arc<AtomicBool>>,
) -> io::Result<Vec<u8>> {
    let mut retained = Vec::with_capacity(limit.min(64 * 1024));
    let mut buffer = [0u8; 8192];
    loop {
        let count = match reader.read(&mut buffer) {
            Ok(0) => return Ok(retained),
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if retain_tail {
            retained.extend_from_slice(&buffer[..count]);
            if retained.len() > limit {
                let remove = retained.len() - limit;
                retained.drain(..remove);
            }
        } else {
            let remaining = limit.saturating_sub(retained.len());
            retained.extend_from_slice(&buffer[..count.min(remaining)]);
            if count > remaining
                && let Some(overflow) = &overflow
            {
                overflow.store(true, Ordering::Relaxed);
            }
        }
    }
}

fn join_reader(reader: JoinHandle<io::Result<Vec<u8>>>) -> Result<Vec<u8>, ProcessFailure> {
    reader
        .join()
        .map_err(|_| ProcessFailure::ReaderPanicked)?
        .map_err(ProcessFailure::Read)
}

struct ChildGuard {
    child: Option<Child>,
    process_group: u32,
}

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self {
            process_group: child.id(),
            child: Some(child),
        }
    }

    fn child_mut(&mut self) -> &mut Child {
        self.child.as_mut().expect("child is still owned")
    }

    fn terminate(&mut self) {
        terminate_group(self.process_group);
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.child = None;
    }

    fn disarm(&mut self) {
        self.child = None;
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if self.child.is_some() {
            self.terminate();
        }
    }
}

#[cfg(unix)]
fn terminate_group(process_group: u32) {
    unsafe {
        libc::kill(-(process_group as libc::pid_t), libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn terminate_group(_process_group: u32) {}

pub(crate) fn safe_environment(temp_dir: &Path) -> Vec<(OsString, OsString)> {
    let mut env = Vec::new();
    if let Some(path) = std::env::var_os("PATH") {
        env.push((OsString::from("PATH"), path));
    }
    if let Some(lang) = std::env::var_os("LANG") {
        env.push((OsString::from("LANG"), lang));
    }
    if let Some(locale) = std::env::var_os("LC_ALL") {
        env.push((OsString::from("LC_ALL"), locale));
    }
    for key in [
        "TMPDIR",
        "TEMP",
        "TMP",
        "HOME",
        "XDG_CACHE_HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    ] {
        let path = if key == "HOME" || key == "TMPDIR" || key == "TEMP" || key == "TMP" {
            temp_dir.as_os_str()
        } else if key.contains("PROXY") || key.contains("proxy") || key.starts_with("SSL_") {
            if let Some(value) = std::env::var_os(key) {
                env.push((OsString::from(key), value));
            }
            continue;
        } else {
            temp_dir.as_os_str()
        };
        env.push((OsStr::new(key).to_os_string(), path.to_os_string()));
    }
    env
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    fn script(directory: &Path, body: &str) -> PathBuf {
        let path = directory.join("fake-yt-dlp");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    fn policy(timeout: Duration, stdout_limit: usize, stderr_limit: usize) -> ProcessPolicy {
        ProcessPolicy {
            timeout,
            stdout_limit,
            stderr_limit,
        }
    }

    #[test]
    fn captures_stdout_and_only_the_stderr_tail() {
        let directory = tempfile::tempdir().unwrap();
        let executable = script(directory.path(), "printf 'result'; printf '0123456789' >&2");
        let output = run(
            &executable,
            &[],
            &safe_environment(directory.path()),
            policy(Duration::from_secs(2), 32, 4),
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(output.status.success());
        assert_eq!(output.stdout, b"result");
        assert_eq!(output.stderr_tail, b"6789");
    }

    #[test]
    fn stdout_limit_terminates_the_process_group() {
        let directory = tempfile::tempdir().unwrap();
        let executable = script(directory.path(), "printf '0123456789'; sleep 10");
        let result = run(
            &executable,
            &[],
            &safe_environment(directory.path()),
            policy(Duration::from_secs(3), 4, 32),
            &CancellationToken::new(),
        );
        assert!(matches!(result, Err(ProcessFailure::StdoutLimit)));
    }

    #[test]
    fn timeout_kills_descendants_in_the_process_group() {
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("escaped-descendant");
        let executable = script(
            directory.path(),
            &format!(
                "(sleep 0.4; printf escaped > '{}') & sleep 10",
                marker.display()
            ),
        );
        let result = run(
            &executable,
            &[],
            &safe_environment(directory.path()),
            policy(Duration::from_millis(100), 32, 32),
            &CancellationToken::new(),
        );
        assert!(matches!(result, Err(ProcessFailure::Timeout)));
        std::thread::sleep(Duration::from_millis(600));
        assert!(!marker.exists());
    }
}

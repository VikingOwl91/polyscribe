use std::ffi::OsStr;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_polyscribe")
}

fn manifest_path(rel: &str) -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(rel);
    p
}

fn run_polyscribe<I, S>(args: I, timeout: Duration) -> std::io::Result<std::process::Output>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut child = Command::new(bin())
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear()
        .env("CI", "1")
        .env("NO_COLOR", "1")
        .spawn()?;

    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            let mut out = std::process::Output {
                status,
                stdout: Vec::new(),
                stderr: Vec::new(),
            };
            if let Some(mut s) = child.stdout.take() {
                use std::io::Read;
                let _ = std::io::copy(&mut s, &mut out.stdout);
            }
            if let Some(mut s) = child.stderr.take() {
                use std::io::Read;
                let _ = std::io::copy(&mut s, &mut out.stderr);
            }
            return Ok(out);
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "polyscribe timed out",
            ));
        }
        thread::sleep(Duration::from_millis(10))
    }
}

fn strip_ansi(s: &str) -> std::borrow::Cow<'_, str> {
    // Minimal stripper for ESC [ ... letter sequence
    if !s.as_bytes().contains(&0x1B) {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    let mut bytes = s.as_bytes().iter().copied().peekable();
    while let Some(b) = bytes.next() {
        if b == 0x1B {
            // Try to consume CSI sequence: ESC '[' ... cmd
            if matches!(bytes.peek(), Some(b'[')) {
                let _ = bytes.next(); // skip '['
                // Skip params/intermediates until a final byte in 0x40..=0x77E
                while let Some(&c) = bytes.peek() {
                    if (0x40..=0x7E).contains(&c) {
                        let _ = bytes.next();
                        break;
                    }
                    let _ = bytes.next();
                }
                continue;
            }
            // Skip single-char ESC sequences
            let _ = bytes.next();
            continue;
        }
        out.push(b as char);
    }
    std::borrow::Cow::Owned(out)
}

fn count_err_in_summary(stderr: &str) -> usize {
    stderr
        .lines()
        .map(|l| strip_ansi(l))
        // Drop trailing CR (Windows) and whitespace
        .map(|l| l.trim_end_matches('\r').trim_end().to_string())
        .filter(|l| match l.split_whitespace().last() {
            Some(tok) if tok == "ERR" => true,
            Some(tok)
                if tok.strip_suffix(":").is_some() && tok.strip_suffix(":") == Some("ERR") =>
            {
                true
            }
            Some(tok)
                if tok.strip_suffix(",").is_some() && tok.strip_suffix(",") == Some("ERR") =>
            {
                true
            }
            _ => false,
        })
        .count()
}

#[test]
fn continue_on_error_all_ok() {
    let input1 = manifest_path("input/1-s0wlz.json");
    let input2 = manifest_path("input/2-vikingowl.json");

    // Avoid temporaries: use &'static OsStr for flags.
    let out = run_polyscribe(
        &[
            input1.as_os_str(),
            input2.as_os_str(),
            OsStr::new("--continue-on-error"),
            OsStr::new("-m"),
        ],
        Duration::from_secs(30),
    )
    .expect("failed to run polyscribe");

    assert!(
        out.status.success(),
        "expected success, stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stderr = String::from_utf8_lossy(&out.stderr);

    // Should not contain any ERR rows in summary
    assert_eq!(
        count_err_in_summary(&stderr),
        0,
        "unexpected ERR rows: {}",
        stderr
    );
}

#[test]
fn continue_on_error_some_fail() {
    let input1 = manifest_path("input/1-s0wlz.json");
    let missing = manifest_path("input/does_not_exist.json");

    let out = run_polyscribe(
        &[
            input1.as_os_str(),
            missing.as_os_str(),
            OsStr::new("--continue-on-error"),
            OsStr::new("-m"),
        ],
        Duration::from_secs(30),
    )
    .expect("failed to run polyscribe");

    assert!(
        !out.status.success(),
        "expected failure exit, stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stderr = String::from_utf8_lossy(&out.stderr);

    // Expect at least one ERR row due to the missing file
    assert!(
        count_err_in_summary(&stderr) >= 1,
        "expected ERR rows in summary, stderr: {}",
        stderr
    );
}

#[test]
fn continue_on_error_all_fail() {
    let missing1 = manifest_path("input/does_not_exist_a.json");
    let missing2 = manifest_path("input/does_not_exist_b.json");

    let out = run_polyscribe(
        &[
            missing1.as_os_str(),
            missing2.as_os_str(),
            OsStr::new("--continue-on-error"),
            OsStr::new("-m"),
        ],
        Duration::from_secs(30),
    )
    .expect("failed to run polyscribe");

    assert!(
        !out.status.success(),
        "expected failure exit, stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stderr = String::from_utf8_lossy(&out.stderr);

    // Expect two ERR rows due to both files missing
    assert!(
        count_err_in_summary(&stderr) >= 2,
        "expected >=2 ERR rows in summary, stderr: {}",
        stderr
    );
}

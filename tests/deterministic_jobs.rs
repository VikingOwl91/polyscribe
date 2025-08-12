use std::ffi::OsStr;
use std::process::{Command, Stdio};
use std::time::Duration;

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

    let start = std::time::Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            let mut out = std::process::Output { status, stdout: Vec::new(), stderr: Vec::new() };
            if let Some(mut s) = child.stdout.take() { let _ = std::io::copy(&mut s, &mut out.stdout); }
            if let Some(mut s) = child.stderr.take() { let _ = std::io::copy(&mut s, &mut out.stderr); }
            return Ok(out);
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "polyscribe timed out"));
        }
        std::thread::sleep(std::time::Duration::from_millis(10))
    }
}

#[test]
fn merge_output_is_deterministic_across_job_counts() {
    let input1 = manifest_path("input/1-s0wlz.json");
    let input2 = manifest_path("input/2-vikingowl.json");

    let out_j1 = run_polyscribe(&[input1.as_os_str(), input2.as_os_str(), OsStr::new("-m"), OsStr::new("--jobs"), OsStr::new("1")], Duration::from_secs(30)).expect("run jobs=1");
    assert!(out_j1.status.success(), "jobs=1 failed, stderr: {}", String::from_utf8_lossy(&out_j1.stderr));

    let out_j4 = run_polyscribe(&[input1.as_os_str(), input2.as_os_str(), OsStr::new("-m"), OsStr::new("--jobs"), OsStr::new("4")], Duration::from_secs(30)).expect("run jobs=4");
    assert!(out_j4.status.success(), "jobs=4 failed, stderr: {}", String::from_utf8_lossy(&out_j4.stderr));

    let s1 = String::from_utf8(out_j1.stdout).expect("utf8");
    let s4 = String::from_utf8(out_j4.stdout).expect("utf8");

    assert_eq!(s1, s4, "merged JSON stdout differs between jobs=1 and jobs=4");
}

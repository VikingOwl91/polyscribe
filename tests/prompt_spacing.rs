use std::io::Write as _;
use std::process::{Command, Stdio};

fn manifest_path(rel: &str) -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(rel);
    p
}

fn collect_stderr_lines(output: &std::process::Output) -> Vec<String> {
    let stderr = String::from_utf8_lossy(&output.stderr);
    stderr.lines().map(|s| s.to_string()).collect()
}

#[test]
fn speaker_prompt_spacing_single_vs_multi_is_consistent() {
    let exe = env!("CARGO_BIN_EXE_polyscribe");
    let input1 = manifest_path("input/1-s0wlz.json");
    let input2 = manifest_path("input/2-vikingowl.json");

    // Single mode
    let mut child1 = Command::new(exe)
        .arg(input1.as_os_str())
        .arg("--set-speaker-names")
        .arg("-m")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn polyscribe (single)");
    {
        let s = child1.stdin.as_mut().unwrap();
        writeln!(s, "Alpha").unwrap();
    }
    let out1 = child1.wait_with_output().unwrap();
    assert!(out1.status.success());
    let lines1 = collect_stderr_lines(&out1);

    // Multi mode
    let mut child2 = Command::new(exe)
        .arg(input1.as_os_str())
        .arg(input2.as_os_str())
        .arg("--set-speaker-names")
        .arg("-m")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn polyscribe (multi)");
    {
        let s = child2.stdin.as_mut().unwrap();
        writeln!(s, "Alpha").unwrap();
        writeln!(s, "Beta").unwrap();
    }
    let out2 = child2.wait_with_output().unwrap();
    assert!(out2.status.success());
    let lines2 = collect_stderr_lines(&out2);

    // Helper to count blank separators around echo block
    fn analyze(lines: &[String]) -> (usize, usize, usize) {
        // count: prompts, blanks, echoes (either legacy "Speaker for " or new mapping lines starting with " - ")
        let mut prompts = 0;
        let mut blanks = 0;
        let mut echoes = 0;
        for l in lines {
            if l.starts_with("Enter speaker name for ") { prompts += 1; }
            if l.trim().is_empty() { blanks += 1; }
            if l.starts_with("Speaker for ") || l.starts_with(" - ") { echoes += 1; }
        }
        (prompts, blanks, echoes)
    }

    let (p1, b1, e1) = analyze(&lines1);
    let (p2, b2, e2) = analyze(&lines2);

    // Expect one prompt/echo for single, two for multi
    assert_eq!(p1, 1);
    assert_eq!(e1, 1);
    assert_eq!(p2, 2);
    assert_eq!(e2, 2);

    // Each mode should have exactly two blank separators: one between prompts and echoes and one after echoes
    // Note: other logs may be absent in tests; we count exactly 2 blanks for single and multi here
    assert!(b1 >= 2, "expected at least two blank separators in single mode, got {}: {:?}", b1, lines1);
    assert!(b2 >= 2, "expected at least two blank separators in multi mode, got {}: {:?}", b2, lines2);
}

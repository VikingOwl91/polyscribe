use assert_cmd::cargo::cargo_bin;
use std::fs;
use std::process::Command;

fn bin() -> std::path::PathBuf {
    cargo_bin("polyscribe")
}

#[test]
fn models_help_shows_global_output_flags() {
    let out = Command::new(bin())
        .args(["models", "--help"]) // subcommand help
        .output()
        .expect("failed to run polyscribe models --help");
    assert!(
        out.status.success(),
        "help exited non-zero: {:?}",
        out.status
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout not utf-8");
    assert!(
        stdout.contains("--json"),
        "--json not shown in help: {stdout}"
    );
    assert!(
        stdout.contains("--quiet"),
        "--quiet not shown in help: {stdout}"
    );
}

#[test]
fn models_version_contains_pkg_version() {
    let out = Command::new(bin())
        .args(["models", "--version"]) // propagate_version
        .output()
        .expect("failed to run polyscribe models --version");
    assert!(
        out.status.success(),
        "version exited non-zero: {:?}",
        out.status
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout not utf-8");
    let want = env!("CARGO_PKG_VERSION");
    assert!(
        stdout.contains(want),
        "version output missing {want}: {stdout}"
    );
}

#[test]
fn models_ls_json_quiet_emits_pure_json() {
    let out = Command::new(bin())
        .args(["models", "ls", "--json", "--quiet"]) // global flags
        .output()
        .expect("failed to run polyscribe models ls --json --quiet");
    assert!(out.status.success(), "ls exited non-zero: {:?}", out.status);
    let stdout = String::from_utf8(out.stdout).expect("stdout not utf-8");
    serde_json::from_str::<serde_json::Value>(stdout.trim()).expect("stdout is not valid JSON");
    // Expect no extra logs on stdout; stderr should be empty in success path
    assert!(out.stderr.is_empty(), "expected no stderr noise");
}

#[test]
fn models_clear_cache_offline_succeeds() {
    // Isolated XDG cache dir: clearing a (non-existent) cache file is a no-op success.
    let tmp = tempfile::tempdir().unwrap();
    let out = Command::new(bin())
        .args(["models", "clear-cache"])
        .env("XDG_CACHE_HOME", tmp.path().join("cache"))
        .env("XDG_CONFIG_HOME", tmp.path().join("config"))
        .output()
        .expect("failed to run polyscribe models clear-cache");
    assert!(
        out.status.success(),
        "clear-cache exited non-zero: {:?}: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout not utf-8");
    assert!(stdout.contains("cleared"), "unexpected stdout: {stdout}");
}

#[test]
fn models_download_no_interaction_skips_prompt() {
    // In non-interactive mode the interactive downloader skips itself and succeeds.
    let out = Command::new(bin())
        .args(["--no-interaction", "models", "download"])
        .output()
        .expect("failed to run polyscribe --no-interaction models download");
    assert!(
        out.status.success(),
        "download --no-interaction exited non-zero: {:?}: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn models_ls_discovers_and_persists_a_local_model() {
    let tmp = tempfile::tempdir().unwrap();
    let models_dir = tmp.path().join("models");
    let config_dir = tmp.path().join("config");
    let data_dir = tmp.path().join("data");
    fs::create_dir_all(&models_dir).unwrap();
    fs::create_dir_all(&config_dir).unwrap();
    fs::create_dir_all(&data_dir).unwrap();
    fs::write(models_dir.join("ggml-tiny.bin"), b"fake model bytes").unwrap();

    let run_ls = || {
        Command::new(bin())
            .args(["models", "ls"])
            .env("POLYSCRIBE_MODELS_DIR", &models_dir)
            .env("XDG_CONFIG_HOME", &config_dir)
            .env("XDG_DATA_HOME", &data_dir)
            .env_remove("POLYSCRIBE_CACHE_DIR")
            .output()
            .expect("failed to run polyscribe models ls")
    };

    let out = run_ls();
    assert!(
        out.status.success(),
        "models ls exited non-zero: {:?}: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ggml-tiny.bin"),
        "models ls did not discover the local model: {stdout}"
    );
    assert!(
        config_dir.join("polyscribe").join("models.json").is_file(),
        "models ls did not persist the discovered model"
    );

    // A fresh process reads the same manifest (restart persistence).
    let out = run_ls();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ggml-tiny.bin"),
        "the discovered model was not persisted: {stdout}"
    );
}
#[test]
fn models_update_skips_local_models_without_network() {
    let tmp = tempfile::tempdir().unwrap();
    let models_dir = tmp.path().join("models");
    let config_dir = tmp.path().join("config");
    let data_dir = tmp.path().join("data");
    fs::create_dir_all(&models_dir).unwrap();
    fs::create_dir_all(&config_dir).unwrap();
    fs::create_dir_all(&data_dir).unwrap();
    fs::write(models_dir.join("ggml-tiny.bin"), b"fake model bytes").unwrap();

    let env = |cmd: &mut Command| {
        cmd.env("POLYSCRIBE_MODELS_DIR", &models_dir)
            .env("XDG_CONFIG_HOME", &config_dir)
            .env("XDG_DATA_HOME", &data_dir)
            .env_remove("POLYSCRIBE_CACHE_DIR");
    };

    let mut ls = Command::new(bin());
    ls.args(["models", "ls"]);
    env(&mut ls);
    assert!(ls.output().unwrap().status.success(), "models ls failed");

    let mut update = Command::new(bin());
    update.args(["models", "update"]);
    env(&mut update);
    let out = update.output().expect("failed to run models update");
    assert!(
        out.status.success(),
        "models update exited non-zero: {:?}: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("skip ggml-tiny: local model"),
        "expected local skip message, got: {stdout}"
    );
    assert!(
        !stdout.contains("HTTP 401"),
        "local update must not contact Hugging Face: {stdout}"
    );
}

#[test]
fn models_rm_accepts_a_file_name() {
    let tmp = tempfile::tempdir().unwrap();
    let models_dir = tmp.path().join("models");
    let config_dir = tmp.path().join("config");
    let data_dir = tmp.path().join("data");
    fs::create_dir_all(&models_dir).unwrap();
    fs::create_dir_all(&config_dir).unwrap();
    fs::create_dir_all(&data_dir).unwrap();
    let model = models_dir.join("ggml-tiny.bin");
    fs::write(&model, b"fake model bytes").unwrap();

    let env = |cmd: &mut Command| {
        cmd.env("POLYSCRIBE_MODELS_DIR", &models_dir)
            .env("XDG_CONFIG_HOME", &config_dir)
            .env("XDG_DATA_HOME", &data_dir)
            .env_remove("POLYSCRIBE_CACHE_DIR");
    };

    let mut rm = Command::new(bin());
    rm.args(["models", "rm", "ggml-tiny.bin"]);
    env(&mut rm);
    let out = rm.output().expect("failed to run models rm");
    assert!(
        out.status.success(),
        "models rm exited non-zero: {:?}: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!model.exists(), "models rm did not remove the file");
}

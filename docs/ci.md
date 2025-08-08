# CI checklist and job outline

Checklist to keep docs and code healthy in CI
- Build: cargo build --all-targets --locked
- Tests: cargo test --all --locked
- Lints: cargo clippy --all-targets -- -D warnings
- Optional: check README and docs snippets (basic smoke run of examples scripts)
  - bash examples/update_models.sh (can be skipped offline)
  - bash examples/transcribe_file.sh (use a tiny sample file if available)

Example GitHub Actions job (outline)
- name: Rust
  runs-on: ubuntu-latest
  steps:
  - uses: actions/checkout@v4
  - uses: dtolnay/rust-toolchain@stable
  - name: Build
    run: cargo build --all-targets --locked
  - name: Test
    run: cargo test --all --locked
  - name: Clippy
    run: cargo clippy --all-targets -- -D warnings

Notes
- For GPU features, set up appropriate runners and add `--features gpu-cuda|gpu-hip|gpu-vulkan` where applicable.
- For docs-only changes, jobs still build/test to ensure doctests and examples compile when enabled.

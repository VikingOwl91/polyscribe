# Pull Request: PolyScribe workspace + plugin system

This PR refactors the repository into a multi-crate Cargo workspace and adds a minimal, working plugin system scaffold over NDJSON/stdio, while preserving existing CLI behavior. It also introduces a stub plugin `polyscribe-plugin-tubescribe` and documentation updates.

Differences & Adaptations
- The repository already contained most of the workspace and plugin scaffolding; this PR focuses on completing and verifying the setup, fixing a symlink path issue in the plugin Makefile, and adding documentation and minor cleanup.
- Existing CLI commands and flags are preserved; a new `plugins` command group is added (list/info/run) without breaking existing outputs.

## Commits

### 1) chore(workspace): scaffold workspace + move crates

Rationale
- Ensure workspace members and resolver are properly defined. The repository already contained these crates; this commit documents the layout and confirms no absolute paths are used.

Updated files (representative snapshots)
- Cargo.toml (workspace):
```
[workspace]
members = [
  "crates/polyscribe-core",
  "crates/polyscribe-protocol",
  "crates/polyscribe-host",
  "crates/polyscribe-cli",
  "plugins/polyscribe-plugin-tubescribe",
]
resolver = "2"
```

Repository tree after this commit (abridged)
```
.
├── Cargo.toml
├── crates
│   ├── polyscribe-cli
│   ├── polyscribe-core
│   ├── polyscribe-host
│   └── polyscribe-protocol
└── plugins
    └── polyscribe-plugin-tubescribe
```

### 2) feat(plugins): host/stdio runner + CLI plugin commands

Rationale
- Provide plugin discovery and stdio NDJSON JSON-RPC runner in host crate; add `plugins` subcommands to CLI. These were already implemented; this commit verifies and documents behavior.

Updated files (representative snapshots)
- crates/polyscribe-host/src/lib.rs: discover(), capabilities(), run_method().
- crates/polyscribe-cli/src/main.rs: `plugins list|info|run` wired to host, forwarding progress.

Repository tree after this commit: unchanged from above.

### 3) feat(plugin): add stub polyscribe-plugin-tubescribe + docs

Rationale (risky change explained)
- Fixed a symlink path issue in the Makefile by switching from $(PWD) to $(CURDIR) to avoid brittle relative paths. This ensures discovery finds the plugin consistently on all shells.
- Removed an unused import to keep clippy clean.
- Added README docs covering workspace layout and verification commands.

Updated files (full contents included in repo):
- plugins/polyscribe-plugin-tubescribe/Makefile
- plugins/polyscribe-plugin-tubescribe/src/main.rs
- README.md (appended Workspace & Plugins section)

Repository tree after this commit (abridged)
```
.
├── Cargo.toml
├── README.md
├── crates
│   ├── polyscribe-cli
│   ├── polyscribe-core
│   ├── polyscribe-host
│   └── polyscribe-protocol
└── plugins
    └── polyscribe-plugin-tubescribe
        ├── Cargo.toml
        ├── Makefile
        └── src/main.rs
```

## Verification commands
- Build the workspace:
  - cargo build --workspace --all-targets
- Show CLI help and plugin subcommands:
  - cargo run -p polyscribe-cli -- --help
- Discover plugins (before linking, likely empty):
  - cargo run -p polyscribe-cli -- plugins list
- Build and link the stub plugin:
  - make -C plugins/polyscribe-plugin-tubescribe link
- Discover again:
  - cargo run -p polyscribe-cli -- plugins list
- Show plugin capabilities:
  - cargo run -p polyscribe-cli -- plugins info tubescribe
- Run a plugin command and observe progress + JSON result:
  - cargo run -p polyscribe-cli -- plugins run tubescribe generate_metadata --json '{"input":{"kind":"text","summary":"hello world"}}'

All acceptance checks pass locally.

# Branching and releases

The branch model of PolyScribe, and how a release is prepared and published.
The rules here match the configuration of the repository: Gitea is the source
of truth, GitHub is a mirror, and no branch is protected by a technical rule
yet.

## Branches

| Branch | Purpose |
|---|---|
| `main` | The latest published state. It only moves to a release commit. |
| `dev` | The integration branch. Everything that is not released yet lands here first. |
| `feature/<topic>` | A short-lived branch for one piece of work. It is opened as a pull request against `dev` and deleted after the merge. |
| `fix/<topic>` | The same, for a fix. |

Rules:

- Work happens on a branch that is branched off `dev`, never directly on `dev`
  or `main`.
- A pull request targets `dev`. `main` is only advanced by a release, see
  below.
- A branch is deleted after its pull request is merged and the work is
  verified to be in `dev`. A branch that still carries commits nobody else
  has is kept until its content has been merged or deliberately dropped.
- `main` is fast-forwarded to the release commit. It is never reset and its
  history is never rewritten.
- No force-push to `main` or `dev`.

## Pull requests

- Gitea is the canonical Git repository. Pull requests are opened there and
  target `dev`; GitHub is the synchronized mirror and is not used for review.
- GitHub Actions is authoritative for merge decisions. Every required check
  must pass on the exact commit at the PR head before the merge. A green run
  for an earlier commit does not satisfy this gate.
- Gitea Actions checks are informational while no functional Gitea runner is
  available. A missing Gitea runner is not a reason to bypass a failing GitHub
  check.
- Changes to CI authority or required checks need review and must be recorded
  in `docs/ci.md` and this document.
- Commits are focused and use the conventional prefix style: `[feat] …`,
  `[fix] …`, `[docs] …`, `[test] …`, `[refactor] …`, `[build] …`,
  `[chore] …`, `[release] …`.

## A release

1. **Prepare** on `dev`:
   - Bump the workspace version in `Cargo.toml` and the matching entries of
     `Cargo.lock` (all workspace members inherit the version).
   - Turn the `Unreleased` section of `CHANGELOG.md` into a dated release
     section.
   - Write the release notes: what the release does, the supported platforms,
     the known limitations, and what changed since the previous release. The
     notes go into the release body on Gitea.
   - Update the documentation that the release changes (README, `docs/`).
2. **Verify**: the checklist of `docs/ci.md` and all required GitHub Actions
   checks pass on the release commit. Gitea Actions are informational while
   no functional runner is available. Test the release with real inference on
   the supported platform.
3. **Tag** the release commit with an annotated, GPG-signed tag:
   `git tag -s -a vX.Y.Z <commit>`. The tag message names the release and
   the platform support. An unsigned tag is not a release.
4. **Publish** the release on Gitea: the tag, the notes, and the source
   archive with its SHA-256 checksum as an asset. The archive is built from
   the tag (`git archive`) so that it holds exactly the released tree.
5. **Mirror**: push the tag and `main` to GitHub, and verify that both
   platforms resolve the tag to the same commit and that the published
   checksum matches the local artifact.
6. **Fast-forward `main`** to the release commit on both platforms, and
   verify that the CI is green on `main` as well.

## The platforms

- Gitea (`somegit.dev`) is the source of truth: pull requests, releases and
  the release archive live there.
- GitHub is the synchronized mirror and runs the authoritative GitHub Actions
  checks. A release is published on Gitea; the GitHub tag is a mirror of it,
  not a second release.
- The CI workflow is the same file on both platforms
  (`.github/workflows/ci.yml`). Gitea Actions are informational while its
  runner is unavailable.

## What is not published

- No packages on crates.io, and no binary distributions. PolyScribe is
  installed from source (`cargo install --path crates/polyscribe-cli`); see
  `docs/release-packaging.md` for the packaging notes of a distribution.

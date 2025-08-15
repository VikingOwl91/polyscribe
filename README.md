# PolyScribe

Local-first transcription and plugins.

## Features

- **Local-first**: Works offline with downloaded models
- **Multiple backends**: CPU, CUDA, ROCm/HIP, and Vulkan support
- **Plugin system**: Extensible via JSON-RPC plugins
- **Model management**: Automatic download and verification of Whisper models
- **Manifest caching**: Local cache for Hugging Face model manifests to reduce network requests

## Model Management

PolyScribe automatically manages Whisper models from Hugging Face:

```bash
# Download models interactively
polyscribe models download

# Update existing models
polyscribe models update

# Clear manifest cache (force fresh fetch)
polyscribe models clear-cache
```

### Manifest Caching

The Hugging Face model manifest is cached locally to avoid repeated network requests:

- **Default TTL**: 24 hours
- **Cache location**: `$XDG_CACHE_HOME/polyscribe/manifest/` (or platform equivalent)
- **Environment variables**:
  - `POLYSCRIBE_NO_CACHE_MANIFEST=1`: Disable caching
  - `POLYSCRIBE_MANIFEST_TTL_SECONDS=3600`: Set custom TTL (in seconds)

## Installation

```bash
cargo install --path .
```

## Usage

```bash
# Transcribe audio/video
polyscribe transcribe input.mp4

# Merge multiple transcripts
polyscribe transcribe --merge input1.json input2.json

# Use specific GPU backend
polyscribe transcribe --gpu-backend cuda input.mp4
```

## Development

```bash
# Build
cargo build

# Run tests
cargo test

# Run with verbose logging
cargo run -- --verbose transcribe input.mp4
```

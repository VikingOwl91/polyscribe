# AMD Radeon RX 9070 XT: HIP validation and CPU/GPU benchmarks

Status: validated on 2026-10-21 against `v0.1.0-alpha.1` (commit `3db364d`) plus the
backend-selection fix (merged in `dev` as `b5d2698`). The multi-model comparison was
measured on branch `feature/bench-multi-model`.

## 1. Environment and hardware

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 3700X, 8 cores / 16 threads, ~4.2-4.3 GHz, governor `performance` |
| GPU | AMD Radeon RX 9070 XT (Navi 48), 16304 MiB VRAM, ISA `gfx1201` (RDNA 4) |
| OS | Arch Linux, kernel `7.2.9-zen1-1.1-zen` |
| ROCm | 7.2.4 at `/opt/rocm`, HIP `7.2.53211-3d9ef42`, AMD clang `22.0.0git` |
| Profiler | `rocprofv3` (ROCm 7.2); `amd-smi` 26.2.2 for VRAM and activity counters |
| Rust | 1.89.0 (`rust-toolchain.toml`), MSRV 1.88 |
| whisper stack | `whisper-rs` 0.16.0, `whisper-rs-sys` 0.15.0, whisper.cpp 1.8.3, ggml 0.9.5 |
| Models | `ggml-tiny.en`, `ggml-base.en`, `ggml-small.en`, `ggml-medium.en` (checksums in section 5) |
| Audio | `crates/polyscribe-cli/tests/fixtures/speech.wav` (16 kHz mono s16, 5.412563 s) and `speech-12x.wav` (the same fixture repeated 12 times, 64.950750 s) |

The host is shared: the load average during the measurements was between 6 and 22 from
other tenants. This matters for the CPU numbers and is called out in the results.

## 2. Build instructions

```bash
# ROCm has to be on PATH, because whisper-rs-sys points cmake at hipcc.
export PATH=/opt/rocm/bin:$PATH

# The HIP compiler writes device code intermediates to TMPDIR. Point it at a
# filesystem with a few GB free; a small /tmp makes the build fail with
# "No space left on device".
export TMPDIR=/var/tmp

cargo build --release -p polyscribe-cli --features gpu-hip --locked
```

A cold HIP build took 2m31s on this machine (110 HIP translation units compiled for
`gfx1201`). An incremental rebuild after a source change took 40s.

No extra CMake flags are needed. `hipcc` detects the installed GPU and compiles for
`gfx1201` by default, which is what the resulting binaries use:

```
/opt/rocm/bin/hipcc ... -x hip --offload-arch=gfx1201 ...
```

## 3. Compatibility findings

1. **RDNA 4 works out of the box.** whisper.cpp 1.8.3 ships ggml 0.9.5, which has
   explicit RDNA 4 code paths (`GGML_CUDA_CC_RDNA4` in `ggml/src/ggml-cuda/common.cuh`,
   `RDNA4` guards in `mmq.cuh`). The CDNA-only MFMA path is not used on `gfx1201`, and
   the build compiles without errors or warnings that affect the result.
2. **The build needs a writable TMPDIR with a few GB free.** HIP/clang writes device
   code intermediates (`*.hipi`, `*-gfx1201-*.o`) into TMPDIR. On this host `/tmp` is a
   16 GB tmpfs that was full, and the build failed with
   `error: unable to open output file '/tmp/mmid-3c79f0.hipi': 'No space left on device'`.
   With `TMPDIR=/var/tmp` the build succeeds.
3. **A HIP build hard-depends on the ROCm runtime.** `libhipblas`, `libamdhip64`,
   `libhsa-runtime64`, `librocblas` and `librocsolver` are linked dynamically. Without
   them the binary does not start at all:
   `./polyscribe: error while loading shared libraries: libhipblas.so.3: cannot open shared object file` (exit 127).
   There is no graceful fallback to CPU, so a HIP-enabled binary is not portable to a
   machine without ROCm.
4. **Backend selection was wrong in a GPU build, and is fixed on this branch.**
   `whisper-rs` defaults `WhisperContextParameters::use_gpu` to the compiled-in GPU
   feature, so a HIP build ran every transcription on the GPU, including
   `--gpu-backend cpu`, while logging "Selected backend: Cpu". The fix passes the
   selected backend into the context parameters, so a CPU request sets `use gpu = 0`
   and the model stays in RAM.
5. **The GPU is enumerated even for CPU requests** in a HIP build
   (`ggml_cuda_init: found 1 ROCm devices` appears with `use gpu = 0`). Harmless, but it
   makes `-vv` logs look as if the GPU was used. The authoritative signal is
   `use gpu = 0/1` plus `whisper_model_load: ROCm0 total size = 77.11 MB`.
6. **`rocprof` v1 does not support gfx1201** and aborts the profiled process. Use
   `rocprofv3 --kernel-trace true` for kernel-level evidence.

## 4. Benchmark methodology

- Same model, same audio and same output configuration for every configuration.
- Audio: `speech-12x.wav` (64.950750 s). The committed 5.4 s fixture is too short to
  separate model loading from inference.
- Phases are taken from the CLI's own progress markers on stderr (`-vv`):
  `0% -> 5%` is the ffmpeg decode, `5% -> 20%` is the Whisper model load and state
  creation, `40% -> 90%` is the inference.
- Wall clock is measured around the whole process, so it includes process start,
  decode, model load, inference and writing the output files.
- Peak RSS is read from `/proc/<pid>/status` (`VmHWM`) while the process runs.
- VRAM and gfx activity are sampled with `amd-smi metric -m/-u --csv` every 400 ms.
  Sampling is expensive, so timing runs are done without it and the resource profile
  is a separate run. Disabling the sampler removed most of the CPU-side noise.
- Cold runs evict the model from the page cache with
  `posix_fadvise(POSIX_FADV_DONTNEED)` before each run.
- CPU and HIP are measured in interleaved pairs so both see the same host load.
- For the multi-model comparison: one warm-up per backend, then 5 measured
  iterations per backend, interleaved, plus a separate resource pass with 2
  iterations per backend and the sampler enabled. The host load average is recorded
  with every run, because on a shared host it is the main source of variance in the
  CPU numbers.
- Model downloads use `polyscribe models add`, which verifies the file against the
  SHA-256 in the Hugging Face manifest; the checksums in section 6 were computed
  locally and match that manifest.

## 5. Results

Paired runs, 6 pairs, 64.95 s of audio, tiny.en model:

| Configuration | Runs | Wall median | Wall min | Wall max | Inference median | Model init median | Peak RSS | Real-time factor |
|---|---|---|---|---|---|---|---|---|
| CPU only, `--gpu-backend cpu` | 6 | 5722 ms | 3778 ms | 7933 ms | 5612 ms | 48.5 ms | 305 MB | 11.4x |
| HIP, `--gpu-backend hip` | 6 | 831 ms | 820 ms | 841 ms | 594 ms | 117.1 ms | 309 MB | 78.2x |

- **Median speedup: 6.9x wall clock, 9.5x inference time.**
- The GPU path is stable (819-841 ms, 1.3% spread). The CPU path varies by 2.1x
  (3778-7933 ms) because of the shared host; the GPU was otherwise idle.
- Model initialization is slower on the GPU (117 ms vs 48 ms): the 77.11 MB of weights
  are copied over PCIe into VRAM instead of being read into RAM.
- Cold cache (model evicted from the page cache), 3 runs each: CPU 3756 ms median
  (2988-4081 ms), HIP 799 ms median (798-803 ms). The GPU path shows no measurable
  cold penalty, the model read overlaps with GPU work.

Resource profile, separate run with `amd-smi` sampling:

| Metric | CPU | HIP |
|---|---|---|
| VRAM baseline (idle, other tenants) | 1901 MiB | 1901 MiB |
| VRAM peak during the run | 2019 MiB (no model in VRAM) | 2171 MiB (about 270 MiB for model and compute buffers) |
| gfx activity | 9-12% (desktop noise) | 30-39% |
| VRAM after the process exits | 1907 MiB | 1907 MiB (allocation released) |

Evidence that the work really ran on the GPU:

1. ggml's own device log: `ggml_cuda_init: found 1 ROCm devices: Device 0: AMD Radeon
   RX 9070 XT, gfx1201 (0x1201), VMM: no, Wave Size: 32`, then
   `whisper_backend_init_gpu: found GPU device 0: ROCm0` and `using ROCm0 backend`.
2. Kernel trace with `rocprofv3 --kernel-trace true` on the 5.4 s fixture: **3653 kernel
   dispatches on Agent 1 (the GPU)**, including ggml's `im2col_kernel<__half>`, the
   rocBLAS GEMM `Cijk_Alik_Bljk_HSS_BH_MT64x32x8_SE_1LDSB0_APM1_...`,
   `unary_op_kernel<&(op_gelu(float)), float>`, `k_bin_bcast`, `norm_f32<32>` and
   `cpy_scalar_transpose<float>`, which are exactly the operations of a whisper
   transformer.
3. Process CPU time against wall clock: the HIP run used 1281 ms of CPU time in 841 ms
   of wall time (1.5 cores), the CPU run used 140,622 ms of CPU time in 10,476 ms of
   wall time (13.4 cores, whisper.cpp multithreaded).

Reliability:

- Five sequential HIP transcriptions: all exit 0, 822-853 ms, and byte-identical
  output (same transcript SHA-256).
- CPU and HIP produced **byte-identical transcripts** for the 64.95 s fixture.
- VRAM returns to the idle baseline after the process exits.

## 6. Multi-model comparison

All four English models were downloaded with the project's own verified path
(`polyscribe models add ggerganov/whisper.cpp <file>`), which checks the file against
the SHA-256 recorded in the Hugging Face manifest. The checksums below were computed
locally and match the manifest exactly.

| Model | Size | SHA-256 |
|---|---|---|
| `ggml-tiny.en.bin` | 74.1 MiB | `921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f` |
| `ggml-base.en.bin` | 141.1 MiB | `a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002` |
| `ggml-small.en.bin` | 465.0 MiB | `c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d` |
| `ggml-medium.en.bin` | 1462.7 MiB | `cc37e93478338ec7700281a7ac30a10128929eb8f427dda2e865faa8f6da4356` |

Method: 64.950750 s of audio (`speech-12x.wav`), one warm-up per backend, then 5
measured iterations per backend, CPU and HIP interleaved so both see the same host
load. A separate resource pass with `amd-smi` sampling captured VRAM and gfx activity.
Raw per-run data is in `docs/benchmarks/data/`.

| Model | Backend | Wall median | Wall min | Wall max | Inference | Model init | Peak RSS | VRAM peak | gfx peak | RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny.en | CPU | 3884 ms | 3033 ms | 9702 ms | 3778 ms | 45 ms | 305 MB | idle (no model in VRAM) | 8% | 16.7x |
| tiny.en | HIP | 822 ms | - | - | 589 ms | 113 ms | 303 MB | 2587 MiB | 40% | 79.0x |
| base.en | CPU | 5449 ms | 4990 ms | 6035 ms | 5323 ms | 54 ms | 412 MB | idle | 8% | 11.9x |
| base.en | HIP | 1195 ms | - | - | 947 ms | 123 ms | 301 MB | 2701 MiB | 84% | 54.3x |
| small.en | CPU | 46795 ms | 21912 ms | 107390 ms | 46618 ms | 120 ms | 857 MB | idle | 19% | 1.4x |
| small.en | HIP | 1806 ms | - | - | 1462 ms | 212 ms | 304 MB | 3126 MiB | 89% | 36.0x |
| medium.en | CPU | 45320 ms | 40061 ms | 115466 ms | 44953 ms | 294 ms | 2046 MB | idle | 9% | 1.4x |
| medium.en | HIP | 3330 ms | - | - | 2800 ms | 377 ms | 307 MB | 4310 MiB | 91% | 19.5x |

Speedups (CPU wall median / HIP wall median):

| Model | Wall clock | Inference only | Host load during the runs |
|---|---|---|---|
| tiny.en | 4.72x | 6.42x | 6.4 |
| base.en | 4.56x | 5.62x | 14.5 |
| small.en | 25.91x | 31.89x | 21.9 |
| medium.en | 13.61x | 16.06x | 21.0 |

**Read the speedups with the load column in mind.** The two models measured under low
load (tiny.en at 6.4, base.en at 14.5) show 4.6-4.7x, the two measured under heavy
load (small.en and medium.en at 21-22) show 13.6-25.9x. The GPU numbers barely move
with host load, the CPU numbers move by a factor of three, so the speedup on a busy
host is not comparable with the speedup on an idle one. The honest statement is: the
GPU path is 0.8-3.3 s of wall clock for 65 s of audio on every model tested, while the
CPU path needs 3.9-46.8 s on this host, and the CPU cost grows much faster with model
size than the GPU cost does.

### Memory scaling

| Model | Model file | VRAM idle | VRAM peak | VRAM delta | Peak RSS (CPU) | Peak RSS (HIP) |
|---|---|---|---|---|---|---|
| tiny.en | 74.1 MiB | 2057 MiB | 2587 MiB | 530 MiB | 305 MB | 303 MB |
| base.en | 141.1 MiB | 2033 MiB | 2701 MiB | 668 MiB | 412 MB | 301 MB |
| small.en | 465.0 MiB | 2018 MiB | 3126 MiB | 1108 MiB | 857 MB | 304 MB |
| medium.en | 1462.7 MiB | 2018 MiB | 4310 MiB | 2292 MiB | 2046 MB | 307 MB |

The VRAM delta is larger than the fp16 weights because it also contains the ggml
compute buffers, which grow with the model's layer and embedding dimensions. It is
well below the model file size, because whisper.cpp converts the weights to fp16 for
the GPU. The idle baseline of about 2 GB is other tenants on the same GPU, not
PolyScribe. On the CPU path the peak RSS grows with the model (305 MB for tiny.en to
2046 MB for medium.en) because the whole model and the decoded audio live in RAM; on
the GPU path the process stays at about 300 MB regardless of the model size, because
the weights are in VRAM.

### Transcript consistency

| Model | CPU transcript SHA-256 | HIP transcript SHA-256 | Identical |
|---|---|---|---|
| tiny.en | `618f069c235c0dc7` | `618f069c235c0dc7` | yes |
| base.en | `b618640e56e53017` | `b618640e56e53017` | yes |
| small.en | `ab22277195f40f2e` | `ab22277195f40f2e` | yes |
| medium.en | `2fa23674f7319d3b` | `2fa23674f7319d3b` | yes |

For all four models the CPU and the HIP run produced byte-identical transcripts, and
the text itself is correct: 12 segments, first segment starting at 0.0 s, last one
ending at 64.4-64.6 s, and the sentence transcribed as "This is a short test recording
for the polyscribe end-to-end transcription test."

## 7. GPU runtime behaviour

- **Model initialization overhead.** The GPU path always loads slower than the CPU
  path: 113 ms against 45 ms for tiny.en, 377 ms against 294 ms for medium.en. The
  weights are read from disk and uploaded over PCIe into VRAM. The gap grows with
  model size but stays small compared with the inference time.
- **VRAM scaling** is in the table above: 530 MiB for tiny.en up to 2292 MiB for
  medium.en, on top of whatever else the GPU is using. A 16 GB card therefore has
  room for models well beyond medium.en.
- **GPU utilization scales with model size.** gfx activity was 40% for tiny.en, 84%
  for base.en, 89% for small.en and 91% for medium.en. The tiny model does not have
  enough work to saturate the GPU, which is also why its speedup over the CPU is the
  smallest of the four.
- **ROCm runtime dependencies.** A HIP-enabled binary links 7 ROCm/HSA libraries
  (`libamdhip64`, `libhipblas`, `libhsa-runtime64`, `librocblas`, `librocsolver`,
  `libroctx64`, `librocprofiler-register`) plus `libdrm_amdgpu`, 22 dynamic
  dependencies in total, all from `/opt/rocm/lib` and `/usr/lib`. Without them the
  binary does not start (exit 127).
- **CPU-only execution from a HIP-enabled binary** works and is now correct: after the
  backend-selection fix, `--gpu-backend cpu` in a HIP build reports `use gpu = 0` and
  keeps the model in RAM, at the cost of about 60 MB more resident memory because the
  ROCm runtime is still mapped.
- **Backend error reporting.** Asking a CPU-only build for the HIP backend fails
  cleanly with "Requested ROCm/HIP backend but this binary was built without HIP
  support. How to fix: install the ROCm runtime and rebuild with --features
  gpu-hip." (exit 1). A missing model file fails with "WHISPER_MODEL points to a
  non-existing path" (exit 1).
- **Silent CPU fallback (fixed in sprint 4).** Before sprint 4, a GPU request with
  no visible device (`HIP_VISIBLE_DEVICES=`) fell through to the CPU: ggml logged
  `failed to initialize ROCm: no ROCm-capable device is detected`, the
  transcription **still succeeded on the CPU with exit 0**, and the log still said
  "Selected backend: Hip". The backend selection now checks the device list in
  ggml's backend registry, so an explicit GPU request without a device fails with
  "Requested the ROCm/HIP backend but no compatible GPU device is available" and
  exit 1, while `auto` falls back to the CPU and says so.
- **Cold versus warm model loading.** With the model evicted from the page cache,
  the GPU path costs 993 ms against 840 ms warm for tiny.en, and 3611 ms against
  3369 ms warm for medium.en. The disk read overlaps with the VRAM upload, so the
  cold penalty stays in the low hundreds of milliseconds.
- **Model context reuse (fixed in sprint 4).** Each input file in one invocation
  used to load the model again: three inputs produced three model loads and three
  ROCm backend initializations, so the GPU path uploaded the whole model over PCIe
  three times (377 ms and 2.3 GB of VRAM traffic for medium.en). The backend now
  keeps the model for the whole invocation, see section 8.

## 8. Model context reuse, before and after

Sprint 4 changed the backend so that the Whisper model is loaded once per
invocation and reused for every input file. Each file still gets its own
inference state, so the outputs and the error handling stay independent.

Measured with the same model (`ggml-tiny.en`, SHA-256 `921e4cf8...20b1f`), the
same input files and the same output configuration. 5 measured runs per
configuration, the configurations interleaved so they all see the same host load
(load average 14-15 during the run). The raw data is in
`docs/benchmarks/data/reuse-summary.json`.

| Configuration | Wall median | Wall min | Wall max | Peak RSS | Peak VRAM | Model loads | Transcript |
|---|---|---|---|---|---|---|---|
| 1 file, CPU, before | 4297 ms | 3681 ms | 17721 ms | 304 MB | 2264 MiB | 1 | `618f069c235c0dc7` |
| 1 file, CPU, after | 4601 ms | 3651 ms | 16815 ms | 304 MB | 2264 MiB | 1 | `618f069c235c0dc7` |
| 1 file, HIP, before | 827 ms | 809 ms | 834 ms | 303 MB | 2818 MiB | 1 | `618f069c235c0dc7` |
| 1 file, HIP, after | 821 ms | 804 ms | 830 ms | 310 MB | 2818 MiB | 1 | `618f069c235c0dc7` |
| 3 files, CPU, before | 12337 ms | 10985 ms | 18022 ms | 324 MB | 2264 MiB | **3** | `8095e514e0150cc1` |
| 3 files, CPU, after | 12885 ms | 12197 ms | 27165 ms | 314 MB | 2264 MiB | **1** | `8095e514e0150cc1` |
| 3 files, HIP, before | 2091 ms | 2083 ms | 2145 ms | 321 MB | 2818 MiB | **3** | `8095e514e0150cc1` |
| 3 files, HIP, after | 1993 ms | 1967 ms | 2235 ms | 328 MB | 2818 MiB | **1** | `8095e514e0150cc1` |

- **The model is loaded once instead of three times** for a three-file batch, on
  both backends. This is the reliable, directly measured effect of the change.
- **The transcripts are byte-identical** before and after in every configuration.
- The wall-clock difference of the batch is inside the noise of this host: the
  CPU runs of the same configuration span 11.0-27.2 s, while the saving is one
  model load (45 ms on the CPU path, 113 ms on the GPU path for tiny.en). For
  medium.en on the GPU the saving is one upload of 1.5 GB of weights, which is
  the case where the change matters most.
- A single file is unaffected, as expected: the model is loaded once either way.
- Peak memory is unchanged, and VRAM returns to its idle value after the run.

## 9. Recommended default configuration for this hardware

- Use a HIP build (`--features gpu-hip`) for any model from `base.en` upwards. It is
  4.6x faster than the CPU path even on a loaded host for base.en, and 13.6-25.9x for
  small.en and medium.en.
- `--gpu-backend auto` picks the GPU in a HIP build, so no flag is needed; use
  `--gpu-backend cpu` when the CPU is wanted explicitly, which now works correctly.
- `tiny.en` is the one model where the GPU advantage is modest (4.7x) and where the
  GPU is not saturated (40% activity).
- Keep the CPU-only build as the portable default for installation and packaging; the
  HIP build is an opt-in artifact that needs ROCm on the target machine.
- medium.en fits comfortably (2.3 GB of VRAM for a 16 GB card), so larger models are
  viable on this hardware. They were not downloaded and measured here, because the
  CPU cost of a `large` model on this host would have added hours of measurement time
  for a result the four tested models already make clear.

## 10. Limitations

- The CPU numbers come from a shared host with a load average between 6 and 22 and are
  therefore a range, not a single value. The GPU numbers are stable. Speedups
  measured under load are not comparable with speedups measured on an idle host.
- The VRAM idle baseline of about 2 GB is other tenants on the same GPU. Only the
  delta is attributable to PolyScribe.
- Only `tiny.en`, `base.en`, `small.en` and `medium.en` were measured. `large` models
  were not downloaded.
- Only the RX 9070 XT / gfx1201 / ROCm 7.2 combination was tested. Other RDNA 4 cards
  and other ROCm versions are untested.
- The GPU backend is not covered by CI. The GitHub Actions workflow runs the CPU
  build; there is no GPU runner in either CI system.
- A GPU request that cannot reach a device now fails instead of falling back to
  the CPU silently (section 7), and `auto` reports the fallback. What PolyScribe
  still cannot establish is whether every kernel of a transcription ran on the
  GPU: whisper-rs 0.16 does not expose that, so the guarantee is limited to "a
  matching device was present and the model was offloaded to it".
- At the time of measurement the Gitea Actions runner (`somegit-runner`, org
  `Owlibou`) was offline, so the queued Gitea runs had not executed. GitHub Actions is
  the CI of record.

## 11. Reproduction commands

```bash
# 1. Build the HIP binary
export PATH=/opt/rocm/bin:$PATH TMPDIR=/var/tmp
cargo build --release -p polyscribe-cli --features gpu-hip --locked
cp target/release/polyscribe /tmp/polyscribe-hip

# 2. Build the CPU-only binary
cargo build --release -p polyscribe-cli --locked
cp target/release/polyscribe /tmp/polyscribe-cpu

# 3. Longer input: the committed fixture repeated 12 times
python3 - << 'EOF'
import wave
src = "crates/polyscribe-cli/tests/fixtures/speech.wav"
with wave.open(src) as w:
    params = w.getparams()
    frames = w.readframes(w.getnframes())
with wave.open("/tmp/speech-12x.wav", "wb") as out:
    out.setparams(params)
    out.writeframes(frames * 12)
EOF

# 4. Download the models with the project's own verified path
export POLYSCRIBE_CACHE_DIR=/var/tmp/polyscribe-cache
export POLYSCRIBE_CONFIG_DIR=/var/tmp/polyscribe-config
for m in tiny base small medium; do
  /tmp/polyscribe-cpu models add ggerganov/whisper.cpp ggml-$m.en.bin
done
sha256sum /var/tmp/polyscribe-cache/models/*.bin   # compare with section 6

# 5. Confirm the GPU is used (look for use gpu = 1 and the ROCm backend)
WHISPER_MODEL=/var/tmp/polyscribe-cache/models/ggml-tiny.en.bin /tmp/polyscribe-hip -vv \
  --no-interaction --no-progress transcribe /tmp/speech-12x.wav --language en -o /tmp/out \
  --gpu-backend hip

# 6. Confirm the CPU path stays on the CPU
WHISPER_MODEL=/var/tmp/polyscribe-cache/models/ggml-tiny.en.bin /tmp/polyscribe-hip -vv \
  --no-interaction --no-progress transcribe /tmp/speech-12x.wav --language en -o /tmp/out \
  --gpu-backend cpu

# 7. Kernel level evidence
rocprofv3 --kernel-trace true --output-format csv --output-directory /tmp/prof -- \
  /tmp/polyscribe-hip --no-interaction --no-progress transcribe \
  crates/polyscribe-cli/tests/fixtures/speech.wav --language en -o /tmp/out --gpu-backend hip

# 8. VRAM and activity counters, before and during a run
amd-smi metric -m --csv
amd-smi metric -u --csv

# 9. What happens when no device is visible (silent CPU fallback)
HIP_VISIBLE_DEVICES= WHISPER_MODEL=/var/tmp/polyscribe-cache/models/ggml-tiny.en.bin \
  /tmp/polyscribe-hip -vv --no-interaction --no-progress transcribe /tmp/speech-12x.wav \
  --language en -o /tmp/out --gpu-backend hip
```

The raw measurement data of the runs behind section 6 is kept in
`docs/benchmarks/data/`: `multi-summary.json` (summary and the resource pass),
`ggml-*.bin-raw.json` (the per-run records of the timing pass), `vram.json` (the
before/during/after VRAM samples) and `bench.log` (the run log).

## 12. Recommended next steps

1. **Document the GPU build** in `docs/usage.md`: that a HIP build needs ROCm at
   runtime, that the ROCm runtime is a hard dependency of such a binary, and that
   an explicit GPU request without a device now fails instead of falling back to
   the CPU. The contract and the model loading behaviour are documented in
   `docs/usage.md` and `docs/design.md`.
2. **Add a `TMPDIR` note** to `docs/development.md` and the CI script, because the HIP
   build needs several GB of temporary space.
3. **Measure `large.en` on the GPU only.** The CPU path for a large model is not
   practical on this host, but the VRAM budget (2.3 GB for medium.en on a 16 GB card)
   suggests it would fit and run at a few times real time.
4. **Add a GPU CI job** if GPU support is to be claimed as tested. Until then, describe
   CUDA and HIP as experimental, as the README now does.
5. **Investigate the Gitea runner outage** separately. It is not a repository problem:
   the runner host is offline, and the same workflow builds fine in an equivalent
   minimal container.

# AMD Radeon RX 9070 XT: HIP validation and CPU/GPU benchmarks

Status: validated on 2026-10-21 against `v0.1.0-alpha.1` (commit `3db364d`) plus the
backend-selection fix on branch `feature/amd-hip-validation`.

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
| Model | `ggml-tiny.en.bin`, 77,704,715 bytes, SHA-256 `921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f` |
| Audio | `crates/polyscribe-cli/tests/fixtures/speech.wav` (16 kHz mono s16, 5.412563 s) and `speech-12x.wav` (the same fixture repeated 12 times, 64.950750 s) |

The host is shared: the load average during the measurements was between 11 and 15 from
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

## 6. Limitations

- The CPU numbers come from a shared host with a load average of 11-15 and are
  therefore a range, not a single value. The GPU numbers are stable.
- Only `tiny.en` was measured. A larger model changes the balance: the GPU offloads
  the whole model, so the VRAM budget (16 GB here) decides which models fit, and
  larger models would amortize the slower model load better.
- Only the RX 9070 XT / gfx1201 / ROCm 7.2 combination was tested. Other RDNA 4 cards
  and other ROCm versions are untested.
- The GPU backend is not covered by CI. The GitHub Actions workflow runs the CPU
  build; there is no GPU runner in either CI system.
- At the time of measurement the Gitea Actions runner (`somegit-runner`, org
  `Owlibou`) was offline, so the queued Gitea runs had not executed. GitHub Actions is
  the CI of record.

## 7. Reproduction commands

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

# 4. Confirm the GPU is used (look for use gpu = 1 and the ROCm backend)
WHISPER_MODEL=/path/to/ggml-tiny.en.bin /tmp/polyscribe-hip -vv --no-interaction \
  --no-progress transcribe /tmp/speech-12x.wav --language en -o /tmp/out --gpu-backend hip

# 5. Confirm the CPU path stays on the CPU
WHISPER_MODEL=/path/to/ggml-tiny.en.bin /tmp/polyscribe-hip -vv --no-interaction \
  --no-progress transcribe /tmp/speech-12x.wav --language en -o /tmp/out --gpu-backend cpu

# 6. Kernel level evidence
rocprofv3 --kernel-trace true --output-format csv --output-directory /tmp/prof -- \
  /tmp/polyscribe-hip --no-interaction --no-progress transcribe \
  crates/polyscribe-cli/tests/fixtures/speech.wav --language en -o /tmp/out --gpu-backend hip

# 7. VRAM and activity counters
amd-smi metric -m --csv
amd-smi metric -u --csv
```

## 8. Recommended next steps

1. **Merge the backend-selection fix** so `--gpu-backend cpu` really uses the CPU in a
   GPU build. It is a three-line change in `crates/polyscribe-core/src/backend.rs` and
   does not affect CPU-only builds.
2. **Document the GPU build** in `docs/usage.md`: that a HIP build needs ROCm at
   runtime and cannot fall back to CPU, and that the ROCm runtime is a hard
   dependency of such a binary.
3. **Add a `TMPDIR` note** to `docs/development.md` and the CI script, because the HIP
   build needs several GB of temporary space.
4. **Re-measure with a larger model** (for example `ggml-base.en.bin`) to see where the
   GPU/break-even point sits for real workloads, and check the VRAM budget on a 16 GB
   card.
5. **Add a GPU CI job** if GPU support is to be claimed as tested. Until then, describe
   CUDA and HIP as experimental, as the README now does.
6. **Investigate the Gitea runner outage** separately. It is not a repository problem:
   the runner host is offline, and the same workflow builds fine in an equivalent
   minimal container.

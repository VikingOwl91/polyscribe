#!/usr/bin/env python3
"""Boundary quality analysis for the chunked transcription pipeline.

Composes the boundary fixtures from the committed speech clips and the case
definitions, transcribes every fixture with the given binaries and compares
the result with the reference transcript.

The comparison separates the two kinds of error:

* the reference transcript is the text that was fed to the speech
  synthesizer, so the difference against it contains both the recognition
  errors of whisper and the errors of the chunk reconciliation;
* the difference between the chunked and the whole file transcription of
  the same fixture isolates the reconciliation errors, because both see
  the same audio and the same model.

Usage:
  scripts/benchmarks/boundary_quality.py \
      --chunked target/release/polyscribe \
      --whole-file /path/to/whole-file/binary \
      --model /path/to/ggml-base.en.bin \
      --out docs/benchmarks/data/boundary-quality.json
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "crates", "polyscribe-cli", "tests", "fixtures", "boundary",
)


def load_cases():
    with open(os.path.join(FIXTURE_DIR, "cases.json")) as fh:
        return json.load(fh)


def read_wav(path):
    with wave.open(path) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, path
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    return rate, frames


def compose(case, out_path, rate=16000):
    """Write the fixture: silence with the clips placed at their offsets."""
    chunks = []
    cursor = 0.0
    for entry in sorted(case["clips"], key=lambda e: e["offset"]):
        offset = entry["offset"]
        gap = int(round((offset - cursor) * rate))
        if gap < -rate // 100:
            raise SystemExit(f"overlapping clips in case {case['name']}")
        gap = max(gap, 0)
        chunks.append(b"\x00\x00" * gap)
        clip_rate, frames = read_wav(os.path.join(FIXTURE_DIR, entry["file"]))
        if clip_rate != rate:
            raise SystemExit(f"clip {entry['file']} has the wrong sample rate")
        chunks.append(frames)
        cursor = offset + len(frames) / 2 / rate
    data = b"".join(chunks)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)
    return len(data) / 2 / rate


def normalize(text):
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def edit_distance(ref, hyp):
    """Word level Levenshtein distance with the substitution, insertion and
    deletion counts, so the error rate can be broken down."""
    prev = list(range(len(hyp) + 1))
    ops = []
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            if r == h:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j - 1], prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def missing_and_duplicated(ref, hyp):
    """The reference words that do not appear in the hypothesis and the
    hypothesis words that appear more often than in the reference."""
    from collections import Counter

    ref_c, hyp_c = Counter(ref), Counter(hyp)
    missing = []
    for word, n in ref_c.items():
        if hyp_c[word] < n:
            missing.extend([word] * (n - hyp_c[word]))
    duplicated = []
    for word, n in hyp_c.items():
        if ref_c[word] < n:
            duplicated.extend([word] * (n - ref_c[word]))
    return missing, duplicated


def transcribe(binary, backend, audio, model, outdir):
    env = dict(os.environ)
    env["WHISPER_MODEL"] = model
    env["TMPDIR"] = os.environ.get("TMPDIR", "/var/tmp")
    os.makedirs(outdir, exist_ok=True)
    proc = subprocess.run(
        [binary, "-vv", "--no-interaction", "--no-progress", "transcribe", audio,
         "--language", "en", "-o", outdir, "--gpu-backend", backend],
        env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"transcribe failed: {proc.stderr[-2000:]}")
    transcript = None
    windows = None
    for name in sorted(os.listdir(outdir)):
        if name.endswith(".json"):
            with open(os.path.join(outdir, name)) as fh:
                transcript = json.load(fh)["items"]
    m = re.search(r"Transcribed (\d+) window", proc.stderr)
    if m:
        windows = int(m.group(1))
    return transcript, windows


def analyse(transcript):
    words = []
    for item in transcript:
        words.extend(normalize(item["text"]))
    starts = [item["start"] for item in transcript]
    ends = [item["end"] for item in transcript]
    monotonic = all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1))
    overlaps = [round(ends[i] - starts[i + 1], 3) for i in range(len(ends) - 1)]
    gaps = [round(starts[i + 1] - ends[i], 3) for i in range(len(ends) - 1)]
    return {
        "segments": len(transcript),
        "words": len(words),
        "text": " ".join(words),
        "timestamps_monotonic": monotonic,
        "max_overlap_s": max(overlaps) if overlaps else 0.0,
        "max_gap_s": max(gaps) if gaps else 0.0,
        "first_start_s": starts[0] if starts else None,
        "last_end_s": ends[-1] if ends else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunked", required=True)
    ap.add_argument("--whole-file", help="binary of the previous, not chunked version")
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", help="directory to keep the composed fixtures in")
    args = ap.parse_args()

    spec = load_cases()
    workdir = args.keep or tempfile.mkdtemp(prefix="boundary-", dir="/var/tmp")
    os.makedirs(workdir, exist_ok=True)

    results = {
        "model": os.path.basename(args.model),
        "backend": args.backend,
        "fixtures": [],
    }

    for case in spec["cases"]:
        audio = os.path.join(workdir, case["name"] + ".wav")
        duration = compose(case, audio)
        reference = normalize(case["reference"])

        entry = {
            "name": case["name"],
            "description": case["description"],
            "audio_seconds": round(duration, 3),
            "reference_words": len(reference),
        }

        for label, binary in (("chunked", args.chunked), ("whole_file", args.whole_file)):
            if not binary:
                continue
            outdir = os.path.join(workdir, f"{case['name']}-{label}")
            transcript, windows = transcribe(binary, args.backend, audio, args.model, outdir)
            info = analyse(transcript)
            hyp = normalize(info["text"])
            distance = edit_distance(reference, hyp)
            missing, duplicated = missing_and_duplicated(reference, hyp)
            info.update({
                "windows": windows,
                "word_errors": distance,
                "wer": round(distance / max(len(reference), 1), 4),
                "missing_reference_words": missing,
                "duplicated_words": duplicated,
            })
            entry[label] = info

        if "chunked" in entry and "whole_file" in entry:
            entry["reconciliation_delta_words"] = (
                entry["chunked"]["word_errors"] - entry["whole_file"]["word_errors"])
        results["fixtures"].append(entry)
        print(f"{case['name']:28s} ref={len(reference):3d} "
              f"chunked_wer={entry['chunked']['wer']:.3f} "
              f"missing={entry['chunked']['missing_reference_words']} "
              f"duplicated={entry['chunked']['duplicated_words']} "
              f"delta_vs_whole_file={entry.get('reconciliation_delta_words')}",
              file=sys.stderr)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

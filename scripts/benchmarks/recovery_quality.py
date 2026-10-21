#!/usr/bin/env python3
"""Compare the chunked transcription with the recovery against the previous
implementations.

The script transcribes the fixtures of the boundary suite and of the
recovery suite with two binaries and reports the word error rate against the
reference transcript, the missing and the duplicated words, and the number
of windows and recovery attempts.

Usage:
  scripts/benchmarks/recovery_quality.py \
      --chunked target/release/polyscribe \
      --previous /path/to/the/previous/binary \
      --model /path/to/ggml-base.en.bin \
      --backend cpu \
      --out docs/benchmarks/data/recovery-quality-cpu.json
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import wave

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOUNDARY_DIR = os.path.join(REPO, "crates", "polyscribe-cli", "tests", "fixtures", "boundary")
RECOVERY_DIR = os.path.join(REPO, "crates", "polyscribe-cli", "tests", "fixtures", "recovery")


def normalize(text):
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def read_wav(path):
    with wave.open(path) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return w.getframerate(), w.readframes(w.getnframes())


def compose(case, out_path, rate=16000):
    """Write the fixture: silence with the clips placed at their offsets."""
    chunks = []
    cursor = 0.0
    for entry in sorted(case["clips"], key=lambda e: e["offset"]):
        gap = int(round((entry["offset"] - cursor) * rate))
        if gap < -rate // 100:
            raise SystemExit(f"overlapping clips in case {case['name']}")
        gap = max(gap, 0)
        chunks.append(b"\x00\x00" * gap)
        clip_rate, frames = read_wav(os.path.join(BOUNDARY_DIR, entry["file"]))
        if clip_rate != rate:
            raise SystemExit(f"clip {entry['file']} has the wrong sample rate")
        chunks.append(frames)
        cursor = entry["offset"] + len(frames) / 2 / rate
    data = b"".join(chunks)
    with wave.open(out_path, "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(rate)
        o.writeframes(data)
    return len(data) / 2 / rate


def word_errors(reference, hypothesis):
    """The word level Levenshtein distance."""
    prev = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        cur = [i] + [0] * len(hypothesis)
        for j, h in enumerate(hypothesis, 1):
            cur[j] = prev[j - 1] if r == h else 1 + min(prev[j - 1], prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def missing_and_duplicated(reference, hypothesis):
    from collections import Counter

    ref_c, hyp_c = Counter(reference), Counter(hypothesis)
    missing = []
    for word, n in ref_c.items():
        missing.extend([word] * max(0, n - hyp_c[word]))
    duplicated = []
    for word, n in hyp_c.items():
        duplicated.extend([word] * max(0, n - ref_c[word]))
    return missing, duplicated


def transcribe(binary, backend, audio, model, outdir, trace=False):
    env = dict(os.environ)
    env["WHISPER_MODEL"] = model
    if trace:
        env["POLYSCRIBE_WINDOW_TRACE"] = "1"
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    cmd = [binary, "-vv", "--no-interaction", "--no-progress", "transcribe", audio,
           "--language", "en", "-o", outdir]
    if backend != "cpu":
        cmd += ["--gpu-backend", backend]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"transcribe failed: {proc.stderr[-2000:]}")
    transcript = []
    for name in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        transcript = json.load(open(name))["items"]
    transcript.sort(key=lambda s: s["start"])
    windows = None
    recoveries = 0
    accepted = 0
    m = re.search(r"Transcribed (\d+) window", proc.stderr)
    if m:
        windows = int(m.group(1))
    recoveries = proc.stderr.count("action=recover")
    # A region that carries speech and was not decoded again is speech that
    # is missing from the transcript.
    lost = 0
    for line in proc.stderr.splitlines():
        if "action=accept" in line and "speech=" in line:
            share = float(line.split("speech=")[1].split()[0])
            if share > 0.05:
                lost += 1
    return transcript, windows, recoveries, lost


def analyse(transcript):
    words = []
    for item in transcript:
        words.extend(normalize(item["text"]))
    starts = [item["start"] for item in transcript]
    ends = [item["end"] for item in transcript]
    overlaps = [round(ends[i] - starts[i + 1], 3) for i in range(len(starts) - 1)]
    gaps = [round(starts[i + 1] - ends[i], 3) for i in range(len(starts) - 1)]
    return {
        "segments": len(transcript),
        "words": len(words),
        "text": " ".join(words),
        "timestamps_monotonic": all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1)),
        "max_overlap_s": max(overlaps) if overlaps else 0.0,
        "max_gap_s": max(gaps) if gaps else 0.0,
        "first_start_s": starts[0] if starts else None,
        "last_end_s": ends[-1] if ends else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunked", required=True, help="the implementation with the recovery")
    ap.add_argument("--previous", required=True, help="the implementation without the recovery")
    ap.add_argument("--whole-file", help="the implementation that hands the whole recording to whisper")
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/var/tmp/recovery-quality")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    with open(os.path.join(BOUNDARY_DIR, "cases.json")) as fh:
        boundary = json.load(fh)
    with open(os.path.join(RECOVERY_DIR, "cases.json")) as fh:
        recovery = json.load(fh)

    results = {"model": os.path.basename(args.model), "backend": args.backend, "fixtures": []}

    clip_texts = {name: info["text"] for name, info in boundary["clips"].items()}

    cases = []
    for case in boundary["cases"]:
        cases.append(("boundary", case, None))
    # The reproducer of the sprint: a recording that whisper does not
    # report the beginning of.
    cases.append(("recovery", {"name": "skipped-22s",
                               "clips": [],
                               "reference": "this is a short test recording for the polyscribe end to end transcription test " * 8},
                  os.path.join(RECOVERY_DIR, "skipped-22s.wav")))
    for case in recovery["cases"]:
        # The reference of a composed case is the text of its clips in the
        # order they are placed.
        case = dict(case)
        case["reference"] = " ".join(clip_texts[c["file"]] for c in case["clips"])
        cases.append(("recovery", case, None))

    for kind, case, audio in cases:
        if audio is None:
            audio = os.path.join(args.work, case["name"] + ".wav")
            compose(case, audio)
        reference = normalize(case["reference"])
        entry = {"name": case["name"], "kind": kind, "reference_words": len(reference)}
        for label, binary in (("chunked", args.chunked), ("previous", args.previous),
                              ("whole_file", args.whole_file)):
            if not binary:
                continue
            transcript, windows, recoveries, accepted = transcribe(
                binary, args.backend, audio, args.model,
                os.path.join(args.work, f"{case['name']}-{label}"), trace=(label == "chunked"))
            info = analyse(transcript)
            hypothesis = normalize(info["text"])
            errors = word_errors(reference, hypothesis)
            missing, duplicated = missing_and_duplicated(reference, hypothesis)
            info.update({
                "windows": windows,
                "recovery_attempts": recoveries,
                "regions_with_speech_not_recovered": accepted,
                "word_errors": errors,
                "wer": round(errors / max(len(reference), 1), 4),
                "missing_reference_words": missing,
                "duplicated_words": duplicated,
            })
            entry[label] = info
        if "chunked" in entry and "previous" in entry:
            entry["recovery_delta_words"] = (
                entry["chunked"]["word_errors"] - entry["previous"]["word_errors"])
        results["fixtures"].append(entry)
        print(f"{case['name']:28s} ref={len(reference):3d} "
              f"chunked={entry['chunked']['wer']:.3f} "
              f"previous={entry['previous']['wer']:.3f} "
              f"whole_file={entry.get('whole_file', {}).get('wer', float('nan')):.3f} "
              f"attempts={entry['chunked']['recovery_attempts']} "
              f"speech_lost={entry['chunked']['regions_with_speech_not_recovered']} "
              f"delta_vs_previous={entry.get('recovery_delta_words')}", file=sys.stderr)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

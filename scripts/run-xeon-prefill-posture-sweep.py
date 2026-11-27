#!/usr/bin/env python3
"""Thread-posture sweep: LAIR prefill-profile vs llama-bench pp at matched thread postures.

Same-host arms alternate in one session (warmup rounds first, measured rounds rotate on odd
rounds) so host drift cancels; each arm optionally pins CPUs with taskset. Throughput parsing
follows the repo's established conventions: LAIR batchedTok/s rows from prefill-profile output
(as in tune-prefill-chain-groups.ps1) and llama-bench pp token columns (as in
benchmark-prefill-vs-llama.ps1).

Arm syntax: lair:<threads>[@cpu-list] or llama:<threads>. The --arms option is comma-separated,
so a cpu list containing commas must be re-joined after splitting; tokens that do not start with
lair:/llama: are appended to the preceding arm. Examples of valid specs:

  lair:8,lair:16,llama:8                       (default posture arms)
  lair:32@0-15                                  (pinned to node0 physical cores)
  lair:32@0-15,32-47                            (node0 physical + HT siblings; the comma inside
                                                 the cpu list is re-joined back into one arm)

llama arms run prompt-processing-only tests mirroring benchmark-prefill-vs-llama.ps1's flag shape
(-m -ngl 0 -t N -p <lengths> -r 1) plus -n 0 to disable generation (tg) tests. CPU pinning via
taskset is only supported for lair arms; llama-bench has its own affinity flags but this sweep
deliberately leaves llama unpinned so taskset effects stay isolated to LAIR.
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import subprocess


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LAIR_KERNEL_RE = re.compile(r"prefill-profile kernel=(\w+)")
# prefill-profile throughput table row: length subBatches batchedMs batchedTok/s [more columns]
LAIR_ROW_RE = re.compile(r"^(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)")
LLAMA_PP_RE = re.compile(r"\|\s*(pp\d+)\s*\|\s*([\d.]+)")  # mean tok/s; stdev column intentionally ignored (locale-safe)
# model|size GiB|params|backend CPU ...|threads|test — same shape benchmark-prefill-vs-llama.ps1 uses
LLAMA_FACTS_RE = re.compile(
    r"\|\s*([^|]+?)\s*\|\s*([^|]+?GiB)\s*\|\s*([^|]+?)\s*\|\s*(CPU[^|]*?)\s*\|\s*(\d+)\s*\|\s*pp\d+\s*\|"
)
MEASURED_LOG_RE = re.compile(r"^posture-r(\d+)-(.+)\.txt$")
ARM_PREFIXES = ("lair:", "llama:")


def parse_args():
    parser = argparse.ArgumentParser(description="Run interleaved same-host LAIR-vs-llama thread-posture sweep.")
    parser.add_argument("--root", default="/root/lair-measurements/qwen25-small-cpu")
    parser.add_argument("--app")
    parser.add_argument("--model")
    parser.add_argument("--dotnet", default="/usr/local/dotnet/dotnet")
    parser.add_argument("--llama", required=True, help="path to llama-bench binary")
    parser.add_argument("--lengths", default="128,256")
    parser.add_argument("--widths", default="64")
    parser.add_argument("--kernels", default="gemv,tiled")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--warmup-rounds", type=int, default=2, help="arm runs executed before the measured rounds and excluded from parsing (cold-start/page-cache handling)")
    parser.add_argument("--arms", default="lair:8,lair:16,lair:32,llama:8,llama:16,llama:32")
    parser.add_argument("--no-rotate", action="store_true", help="keep a fixed arm order in every measured round instead of reversing it on odd rounds")
    parser.add_argument("--report-name", default="optimize-prefill-posture-sweep-xeon")
    parser.add_argument("--dry-run", action="store_true", help="print the planned command schedule and exit without running or writing anything")
    return parser.parse_args()


def median(values):
    values = sorted(values)
    n = len(values)
    if not n:
        raise ValueError("no samples")
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2.0


def spread_of(values, med):
    if med:
        return (max(values) - min(values)) / med
    return 0.0


def add(samples, key, value):
    samples.setdefault(key, []).append(value)


def sanitize(arm_raw):
    """Filesystem-safe arm token used in log filenames; must round-trip through the raw->sanitized map."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", arm_raw).strip("_") or "unnamed"


def parse_arms(spec):
    """Split a comma-separated arms spec, re-joining fragments that continue a cpu list.

    lair:32@0-15,32-47 survives the split because `32-47` does not start with an arm prefix and is
    appended back to the previous token with a comma separator.
    """
    tokens = [x.strip() for x in spec.split(",")]
    joined = []
    for tok in tokens:
        if not tok:
            continue
        if any(tok.startswith(prefix) for prefix in ARM_PREFIXES):
            joined.append(tok)
        elif joined:
            joined[-1] += "," + tok
        else:
            raise SystemExit(f"invalid arm token (must start with lair:/llama:): {tok!r}")

    arms = []  # dicts: raw, kind, threads, cpulist (None or str)
    for entry in joined:
        match = re.match(r"^(lair|llama):(\d+)(?:@(.+))?$", entry)
        if not match:
            raise SystemExit(f"invalid arm spec: {entry!r} (expected lair:<threads>[@<cpu-list>] or llama:<threads>)")
        kind = match.group(1)
        threads = int(match.group(2))
        cpulist_raw = match.group(3)

        if kind == "llama" and cpulist_raw is not None:
            raise SystemExit(f"cpu pinning on llama arms is not supported by this script (taskset only applies to lair arms): {entry!r}")
        if threads <= 0:
            raise SystemExit(f"invalid thread count in arm {entry!r}: must be > 0")

        cpulist = cpulist_raw.strip() if cpulist_raw else None
        if kind == "lair" and cpulist is not None:
            if not re.match(r"[0-9]+(?:[-,][0-9]+)*$", cpulist):
                raise SystemExit(f"invalid cpu list in arm {entry!r}: expected digits with - ranges or , separators")

        arms.append({"raw": entry, "kind": kind, "threads": threads, "cpulist": cpulist})

    if not arms:
        raise SystemExit("--arms must list at least one arm")

    sanitized_to_raw = {}
    for arm in arms:
        name = sanitize(arm["raw"])
        if name in sanitized_to_raw and sanitized_to_raw[name] != arm["raw"]:
            raise SystemExit(f"arms {sanitized_to_raw[name]!r} and {arm['raw']!r} collide after sanitizing to {name!r}")
        sanitized_to_raw[name] = arm["raw"]

    by_identity = {}
    for arm in arms:
        key = (arm["kind"], arm["threads"], arm["cpulist"])
        if key in by_identity:
            raise SystemExit(f"duplicate posture {key[0]}:{key[1]}@{key[2] or 'unpinned'} ({by_identity[key]!r} and {arm['raw']!r})")
        by_identity[key] = arm["raw"]

    return arms, sanitized_to_raw


def run(cmd, label, out_path):
    print(f"RUN[{label}]: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = proc.stdout.decode("utf-8", errors="replace")  # locale-independent decode; logs stay UTF-8 on disk
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}: {text}")
    out_path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    app = Path(args.app) if args.app else root / "app" / "Lair.Bench.dll"
    model = Path(args.model) if args.model else root / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

    arms, sanitized_to_raw = parse_arms(args.arms)
    raw_order = [arm["raw"] for arm in arms]
    lengths_values = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    kernel_order = [x.strip() for x in args.kernels.split(",") if x.strip()]

    started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp_dir = datetime.strptime(started_utc, "%Y-%m-%dT%H:%M:%SZ").strftime(f"reports/%Y-%m-%dT%H-%M-%SZ-{args.report_name}")
    report_dir = root / Path(stamp_dir)
    raw_dir = report_dir / "logs"

    def bench_cmd(arm):
        if arm["kind"] == "lair":
            cmd = []
            if arm["cpulist"]:
                cmd += ["taskset", "-c", arm["cpulist"]]
            return cmd + [
                str(args.dotnet), str(app), "prefill-profile", str(model),
                "--threads", str(arm["threads"]),
                "--lengths", args.lengths,
                "--widths", args.widths,
                "--kernel", args.kernels,
                "--reps", "1",
            ]

        return [
            str(args.llama), "-m", str(model),
            "-ngl", "0",
            "-t", str(arm["threads"]),
            "-p", args.lengths,
            "-n", "0",  # disable generation (tg) tests: prompt-processing-only measurement
            "-r", "1",
        ]

    def measured_order(round_index):
        return list(reversed(raw_order)) if (round_index % 2 and not args.no_rotate) else raw_order

    def arm_by_raw(raw):
        return next(a for a in arms if a["raw"] == raw)

    schedule = [("warmup", arm, f"posture-warm{w}-{sanitize(arm['raw'])}.txt") for w in range(args.warmup_rounds) for arm in arms]
    measured_schedule = [
        (f"measured-round{r}", arm_by_raw(raw), f"posture-r{r}-{sanitize(raw)}.txt")
        for r in range(args.rounds) for raw in measured_order(r)
    ]

    all_schedule = schedule + measured_schedule

    if args.dry_run:
        print(f"dry-run report_dir={report_dir} warmup_rounds={args.warmup_rounds} rounds={args.rounds} rotate={not args.no_rotate}", flush=True)
        for label, arm, out_name in all_schedule:
            print(f"RUN[{label}-arm-{arm['raw']}]: {' '.join(bench_cmd(arm))} -> {raw_dir / out_name}", flush=True)
        return

    report_dir.mkdir(parents=True, exist_ok=False)
    raw_dir.mkdir(parents=True, exist_ok=False)

    for label, arm, out_name in all_schedule:
        run(bench_cmd(arm), f"{label}-arm-{arm['raw']}", raw_dir / out_name)

    samples = {}            # lair: arm|kernel|length|tok_s ; llama: arm|length|tok_s
    batched_ms_samples = {}  # secondary metric parsed from the same LAIR rows (never used for comparisons)
    llama_facts_by_arm = {}

    def parse_lair_log(path, arm_raw):
        text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
        current_kernel = None
        for line in text.splitlines():
            kernel_match = LAIR_KERNEL_RE.search(line)
            if kernel_match:
                current_kernel = kernel_match.group(1)
                continue
            if not current_kernel:
                continue
            row_match = LAIR_ROW_RE.match(line.strip())
            if not row_match:
                continue
            length_str, _sub_batches, batched_ms_str, tok_s_str = row_match.groups()
            try:
                length = int(length_str)
                add(samples, f"{arm_raw}|{current_kernel}|{length}|tok_s", float(tok_s_str))
                add(batched_ms_samples, f"{arm_raw}|{current_kernel}|{length}|batched_ms", float(batched_ms_str))
            except ValueError:
                continue

    def parse_llama_log(path, arm_raw):
        text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
        if arm_raw not in llama_facts_by_arm:
            for line in text.splitlines():
                facts_match = LLAMA_FACTS_RE.search(line)
                if facts_match:
                    try:
                        threads_fact = int(facts_match.group(5))
                    except ValueError:
                        continue
                    llama_facts_by_arm[arm_raw] = {
                        "model": facts_match.group(1).strip(),
                        "size": facts_match.group(2).strip(),
                        "params": facts_match.group(3).strip(),
                        "backend": facts_match.group(4).strip(),
                        "threads": threads_fact,
                    }
                    break
        for line in text.splitlines():
            match = LLAMA_PP_RE.search(line)
            if not match:
                continue
            test_name, mean_str = match.groups()
            length = int(test_name[2:])  # pp128 -> 128
            add(samples, f"{arm_raw}|{length}|tok_s", float(mean_str))

    for path in sorted(raw_dir.glob("posture-r*.txt")):
        name_match = MEASURED_LOG_RE.match(path.name)
        if not name_match:
            continue
        arm_raw = sanitized_to_raw.get(name_match.group(2), name_match.group(2))
        try:
            arm = arm_by_raw(arm_raw)
        except StopIteration:
            print(f"skipping unrecognized measured log {path.name}", flush=True)
            continue
        if arm["kind"] == "lair":
            parse_lair_log(path, arm_raw)
        else:
            parse_llama_log(path, arm_raw)

    lair_arms_seen = [raw for raw in raw_order if any(len(k.split("|")) == 4 and k.startswith(f"{raw}|") for k in samples)]
    llama_arms_seen = [raw for raw in raw_order if any(len(k.split("|")) == 3 and k.startswith(f"{raw}|") for k in samples)]

    def kernel_rank(kernel):
        return (kernel_order.index(kernel), kernel) if kernel in kernel_order else (99, kernel)

    lair_combos = sorted(
        {(k.split("|")[1], int(k.split("|")[2])) for k in samples if len(k.split("|")) == 4},
        key=lambda item: (kernel_rank(item[0]), item[1]),
    )
    llama_lengths_seen = sorted({int(k.split("|")[1]) for k in samples if len(k.split("|")) == 3})

    if not lair_combos or not any(k.endswith("|tok_s") and len(k.split("|")) == 4 for k in samples):
        raise SystemExit("no parseable LAIR batchedTok/s rows found in measured logs (check prefill-profile throughput table output)")
    if not llama_arms_seen or not llama_lengths_seen:
        raise SystemExit("no parseable llama-bench pp token rows found in measured logs (check markdown table output)")

    rows = []
    lookup = {}          # (arm, kernel_or_empty, length) -> median tok/s
    spread_lookup = {}   # same key -> intra-arm spread fraction

    def append_row(arm_raw, kernel, length, metric, values):
        kind = arm_by_raw(arm_raw)["kind"]
        med = median(values)
        rows.append({
            "key": f"{kind}|{arm_raw}|{kernel or ''}|{length}|{metric}",
            "kind": kind,
            "arm": arm_raw,
            "threads": arm_by_raw(arm_raw)["threads"],
            "cpulist": arm_by_raw(arm_raw)["cpulist"] or "",
            "kernel": kernel or "",
            "length": length,
            "metric": metric,
            "samples": [f"{v:.3f}" for v in values],
            "median": med,
            "min": min(values),
            "max": max(values),
            "count": len(values),
            "spread": spread_of(values, med),
        })

    for arm_raw in lair_arms_seen:
        for kernel, length in lair_combos:
            key = f"{arm_raw}|{kernel}|{length}|tok_s"
            if key not in samples:
                continue
            append_row(arm_raw, kernel, length, "tok_s", samples[key])

    for arm_raw in llama_arms_seen:
        for length in llama_lengths_seen:
            key = f"{arm_raw}|{length}|tok_s"
            if key not in samples:
                continue
            append_row(arm_raw, "", length, "tok_s", samples[key])

    # secondary batched_ms rows (parsed alongside tok_s; reported but never used for comparisons)
    for arm_raw in lair_arms_seen:
        for kernel, length in lair_combos:
            ms_key = f"{arm_raw}|{kernel}|{length}|batched_ms"
            tok_key = f"{arm_raw}|{kernel}|{length}|tok_s"
            if ms_key not in batched_ms_samples or len(samples.get(tok_key, [])) != len(batched_ms_samples[ms_key]):
                continue
            append_row(arm_raw, kernel, length, "batched_ms", batched_ms_samples[ms_key])

    for row in rows:
        if row["metric"] == "tok_s":
            lookup[(row["arm"], row["kernel"], row["length"])] = float(row["median"])
            spread_lookup[(row["arm"], row["kernel"], row["length"])] = float(row["spread"])

    combo_index = {(kernel, length): i for i, (kernel, length) in enumerate(lair_combos)}

    def compare_against_llama(lair_arm):
        llama_arms_same_threads = [a for a in arms if a["kind"] == "llama" and a["threads"] == lair_arm["threads"]]
        entries = []
        all_separate = True
        worst_band_pct = 0.0

        def compare_with(llama_arm):
            nonlocal all_separate, worst_band_pct
            for kernel, length in lair_combos:
                l_med = lookup.get((lair_arm["raw"], kernel, length))
                g_med = lookup.get((llama_arm["raw"], "", length))
                if l_med is None or g_med is None:
                    continue

                delta_pct = (l_med - g_med) / g_med * 100.0 if g_med else 0.0
                band_pct = max(
                    spread_lookup.get((lair_arm["raw"], kernel, length), 0.0),
                    spread_lookup.get((llama_arm["raw"], "", length), 0.0),
                ) * 100.0
                ratio_pct = l_med / g_med * 100.0 if g_med else None
                separates = abs(delta_pct) > band_pct

                worst_band_pct = max(worst_band_pct, band_pct)
                if not separates:
                    all_separate = False
                entries.append({
                    "kernel": kernel,
                    "length": length,
                    "lair_median_tok_s": round(l_med, 3),
                    "llama_arm": llama_arm["raw"],
                    "llama_median_tok_s": round(g_med, 3),
                    "ratio_lair_over_llama_pct": round(ratio_pct, 2) if ratio_pct is not None else None,
                    "delta_vs_matched_pp_pct": round(delta_pct, 2),
                    "lair_spread_at_shape_pct": round(spread_lookup.get((lair_arm["raw"], kernel, length), 0.0) * 100.0, 2),
                    "llama_spread_at_length_pct": round(spread_lookup.get((llama_arm["raw"], "", length), 0.0) * 100.0, 2),
                    "noise_band_pct": round(band_pct, 2),
                    "separates_beyond_noise": separates,
                })

        for llama_arm in llama_arms_same_threads:
            compare_with(llama_arm)

        entries.sort(key=lambda e: combo_index.get((e["kernel"], e["length"]), 99))
        if not entries:
            return None

        verdict = (
            f"{'yes' if all_separate else 'no'}: LAIR medians at posture {lair_arm['raw']} differ from matched llama pp on every shape beyond the noise band."
            if all_separate else
            f"no: some LAIR/llama deltas at posture {lair_arm['raw']} fall inside their own noise bands; per-shape verdicts below are authoritative."
        )
        return {
            "threads": lair_arm["threads"],
            "lair_arm": lair_arm["raw"],
            "cpulist": lair_arm["cpulist"] or "",
            "llama_arms": [a["raw"] for a in llama_arms_same_threads],
            "shapes": entries,
            "worst_case_noise_band_pct": round(worst_band_pct, 2),
            "verdict": verdict,
        }

    pairs = []
    pinned_pairings = []
    unpaired_lair_raws = [a["raw"] for a in arms if a["kind"] == "lair" and not a["cpulist"]]
    pinned_lair_arms = [a for a in arms if a["kind"] == "lair" and a.get("cpulist")]

    for arm in arms:
        if arm["kind"] != "lair":
            continue
        pairing = compare_against_llama(arm)
        if not pairing:
            continue
        if arm["raw"] in unpaired_lair_raws:
            pairs.append(pairing)
        else:
            pinned_pairings.append(pairing)

    # posture scaling data (LAIR tok/s by threads for each kernel/length, unpinned arms only)
    posture_scaling_rows = []
    for arm_raw in unpaired_lair_raws:
        if arm_raw not in lair_arms_seen:
            continue
        threads = arm_by_raw(arm_raw)["threads"]
        for kernel, length in lair_combos:
            med = lookup.get((arm_raw, kernel, length))
            if med is None:
                continue
            posture_scaling_rows.append({
                "kernel": kernel,
                "length": length,
                "thread_posture": f"{threads} threads",
                "lair_arm": arm_raw,
                "median_tok_s": round(med, 3),
                "per_thread_efficiency_tok_s_per_thread": round(med / threads, 4) if threads else None,
            })

    data = {
        "report_started_utc": started_utc,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lane": f"Xeon CPU-only lair-lab (root@192.168.33.53), rowWidth={args.widths}",
        "model_file": str(model),
        "lair_binary": str(app),
        "dotnet_runtime": str(args.dotnet),
        "llama_binary": str(args.llama),
        "method": (
            f"{args.warmup_rounds} warmup rounds then {args.rounds} measured rounds x arms [{', '.join(raw_order)}] interleaved back-to-back in one session so host drift cancels; "
            f"measured arm order rotates on odd rounds to cancel positional bias while warmups run each arm once per round for cold-start/page-cache handling; "
            f"reps=1 per command with kernels {', '.join(kernel_order)} and lengths {args.lengths}; LAIR batchedTok/s medians parsed from prefill-profile throughput rows, "
            f"llama pp token medians parsed from llama-bench markdown columns (-ngl 0 -t N -p <lengths> -n 0 -r 1); taskset pins CPUs only for lair arms."
        ),
        "rounds": args.rounds,
        "warmup_rounds": args.warmup_rounds,
        "rotate_arm_order": not args.no_rotate,
        "arms": [{"raw": a["raw"], "kind": a["kind"], "threads": a["threads"], "cpulist": a["cpulist"] or ""} for a in arms],
        "llama_model_facts_by_arm": {k: v for k, v in llama_facts_by_arm.items() if k in llama_arms_seen},
        "rows": rows,
        "posture_scaling_rows": posture_scaling_rows,
        "matched_posture_pairs": pairs,
        "pinned_pairings": pinned_pairings,
    }

    (report_dir / "data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    summary = [f"# Prefill thread-posture sweep {Path(report_dir).name}", "", f"Lane: {data['lane']}. Method: {data['method']}", ""]
    for arm in arms:
        posture_note = f"taskset -c {arm['cpulist']}" if arm["cpulist"] else ("unpinned (llama)" if arm["kind"] == "llama" else "unpinned")
        summary.append(f"- `{arm['raw']}` ({posture_note})")

    if llama_facts_by_arm:
        summary.extend(["", "## llama-bench model facts"])
        for arm_raw in llama_arms_seen:
            facts = llama_facts_by_arm.get(arm_raw)
            if facts:
                summary.append(f"- `{arm_raw}`: {facts['model']} | {facts['size']} | {facts['params']} | {facts['backend']} | threads={facts['threads']}")

    lair_arms_for_tables = [a for a in raw_order if a in lair_arms_seen]
    llama_arms_for_tables = [a for a in raw_order if a in llama_arms_seen]

    summary.append("")
    header_cells = ["kernel", "length"] + lair_arms_for_tables
    summary.extend(["## LAIR batchedTok/s medians", "| " + " | ".join(header_cells) + " |", "| " + " | ".join(["---:"] * len(header_cells)) + " |"])
    for kernel, length in lair_combos:
        cells = [kernel, str(length)]
        for arm_raw in lair_arms_for_tables:
            med = lookup.get((arm_raw, kernel, length))
            cells.append(f"{med:.3f}" if med is not None else "-")
        summary.append("| " + " | ".join(cells) + " |")

    header_cells = ["kernel", "length"] + [f"{a} %" for a in lair_arms_for_tables]
    summary.extend(["", "## LAIR intra-arm tok/s spread (= (max-min)/median)", "| " + " | ".join(header_cells) + " |", "| " + " | ".join(["---:"] * len(header_cells)) + " |"])
    for kernel, length in lair_combos:
        cells = [kernel, str(length)]
        for arm_raw in lair_arms_for_tables:
            sp = spread_lookup.get((arm_raw, kernel, length))
            cells.append(f"{sp * 100.0:.2f}" if sp is not None else "-")
        summary.append("| " + " | ".join(cells) + " |")

    header_cells = ["length"] + [f"{a} %" for a in llama_arms_for_tables]
    summary.extend(["", "## llama-bench intra-arm pp tok/s spread (= (max-min)/median)", "| " + " | ".join(header_cells) + " |", "| " + " | ".join(["---:"] * len(header_cells)) + " |"])
    for length in llama_lengths_seen:
        cells = [str(length)]
        for arm_raw in llama_arms_for_tables:
            sp = spread_lookup.get((arm_raw, "", length))
            cells.append(f"{sp * 100.0:.2f}" if sp is not None else "-")
        summary.append("| " + " | ".join(cells) + " |")

    header_cells = ["length"] + llama_arms_for_tables
    summary.extend(["", "## llama-bench pp tok/s medians", "| " + " | ".join(header_cells) + " |", "| " + " | ".join(["---:"] * len(header_cells)) + " |"])
    for length in llama_lengths_seen:
        cells = [str(length)]
        for arm_raw in llama_arms_for_tables:
            med = lookup.get((arm_raw, "", length))
            cells.append(f"{med:.3f}" if med is not None else "-")
        summary.append("| " + " | ".join(cells) + " |")

    # posture scaling table (LAIR unpinned arms): tok/s by thread posture plus per-thread efficiency
    unpaired_seen = [a for a in raw_order if a in lair_arms_seen and not arm_by_raw(a)["cpulist"]]
    summary.extend(["", "## Posture scaling: LAIR batchedTok/s by thread posture (unpinned arms)"])
    if unpaired_seen:
        header_cells = ["kernel", "length"] + [f"{arm_by_raw(a)['threads']} threads tok/s / per-thread eff" for a in unpaired_seen]
        summary.append("| " + " | ".join(header_cells) + " |")
        summary.append("| " + " | ".join(["---:"] * len(header_cells)) + " |")
        for kernel, length in lair_combos:
            cells = [kernel, str(length)]
            for arm_raw in unpaired_seen:
                med = lookup.get((arm_raw, kernel, length))
                threads = arm_by_raw(arm_raw)["threads"]
                cells.append(f"{med:.3f} / {med / threads:.4f}" if med is not None else "-")
            summary.append("| " + " | ".join(cells) + " |")

        # per-thread efficiency trend across thread counts for each shape
        for kernel, length in lair_combos:
            parts = []
            prev_eff = None
            for arm_raw in unpaired_seen:
                med = lookup.get((arm_raw, kernel, length))
                if med is None:
                    continue
                threads = arm_by_raw(arm_raw)["threads"]
                eff = med / threads
                trend = ""
                if prev_eff and prev_eff > 0:
                    change_pct = (eff - prev_eff) / prev_eff * 100.0
                    trend = f" ({change_pct:+.1f}% vs previous posture)"
                parts.append(f"{threads}t eff {eff:.4f}{trend}")
                prev_eff = eff
            if len(parts) >= 2:
                summary.append(f"- `{kernel}` length {length}: " + "; ".join(parts))

        # pinned-vs-unpinned same-threads deltas (phase-3 style arms, when present)
        for arm in pinned_lair_arms:
            base_raws = [a["raw"] for a in arms if a["kind"] == "lair" and not a["cpulist"] and a["threads"] == arm["threads"]]
            if not base_raws:
                continue
            summary.append(f"- pinned `{arm['raw']}` (taskset -c {arm['cpulist']}) vs unpinned same-threads:")
            for kernel, length in lair_combos:
                med = lookup.get((arm["raw"], kernel, length))
                base_med = next((lookup.get((b, kernel, length)) for b in base_raws), None)
                if med is None or base_med is None:
                    continue
                delta_pct = (med - base_med) / base_med * 100.0 if base_med else 0.0
                band_pct = max(spread_lookup.get((arm["raw"], kernel, length), 0.0), spread_lookup.get((base_raws[0], kernel, length), 0.0)) * 100.0
                separates = abs(delta_pct) > band_pct
                summary.append(
                    f"  - `{kernel}` length {length}: pinned {med:.3f} vs unpinned {base_med:.3f}, delta {delta_pct:+.2f}% "
                    f"vs noise band ±{band_pct:.2f}% -> separates beyond own spread: {'yes' if separates else 'no'}"
                )

    # matched-posture ratios (unpinned lair arms vs same-threads llama arm)
    summary.extend(["", "## Matched-posture comparison (LAIR vs llama at the same -t value)"])
    for pair in pairs:
        llama_label = ", ".join(pair["llama_arms"]) or "(no same-threads llama arm)"
        summary.append(f"### threads={pair['threads']}: `{pair['lair_arm']}` vs {llama_label}")
        header_cells = ["kernel", "length", "LAIR tok/s", "llama pp tok/s", "ratio %", "delta vs matched pp %", "noise band %", "separates"]
        summary.append("| " + " | ".join(header_cells) + " |")
        summary.append("| " + " | ".join(["---:"] * len(header_cells)) + " |")
        for entry in pair["shapes"]:
            ratio = f"{entry['ratio_lair_over_llama_pct']:.2f}" if entry["ratio_lair_over_llama_pct"] is not None else "-"
            separates_flag = "yes" if entry["separates_beyond_noise"] else "no"
            summary.append(
                f"| {entry['kernel']} | {entry['length']} | {entry['lair_median_tok_s']:.3f} | {entry['llama_median_tok_s']:.3f} | "
                f"{ratio} | {entry['delta_vs_matched_pp_pct']:+.2f} | {entry['noise_band_pct']:.2f} | {separates_flag} |"
            )
        summary.append(f"- verdict: {pair['verdict']}")

    # pinned arm comparisons vs matched llama (when present)
    for pair in pinned_pairings:
        llama_label = ", ".join(pair["llama_arms"]) or "(no same-threads llama arm)"
        summary.extend(["", f"## Pinned posture comparison: `{pair['lair_arm']}` vs {llama_label}"])
        header_cells = ["kernel", "length", "LAIR tok/s", "llama pp tok/s", "ratio %", "noise band %", "separates"]
        summary.append("| " + " | ".join(header_cells) + " |")
        summary.append("| " + " | ".join(["---:"] * len(header_cells)) + " |")
        for entry in pair["shapes"]:
            ratio = f"{entry['ratio_lair_over_llama_pct']:.2f}" if entry["ratio_lair_over_llama_pct"] is not None else "-"
            separates_flag = "yes" if entry["separates_beyond_noise"] else "no"
            summary.append(
                f"| {entry['kernel']} | {entry['length']} | {entry['lair_median_tok_s']:.3f} | {entry['llama_median_tok_s']:.3f} | "
                f"{ratio} | {entry['noise_band_pct']:.2f} | {separates_flag} |"
            )

    summary.extend(["", ""])
    (report_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    labels = [f"{kernel} {length}" for kernel, length in lair_combos]
    colors = ["#3b6fb6", "#f2a65e", "#2a7f62", "#c0603f", "#8a5b00", "#9aa5b1", "#4d3f8a", "#116466"]
    bar_options = {
        "scales": {"y": {"title": {"display": True, "text": "batchedTok/s median"}, "beginAtZero": True}},
        "plugins": {"legend": {"position": "bottom"}},
    }

    lair_datasets = []
    for idx, arm_raw in enumerate(lair_arms_for_tables):
        vals = [lookup.get((arm_raw, kernel, length)) for kernel, length in lair_combos]
        if all(v is not None for v in vals):
            lair_datasets.append({"label": f"{arm_raw} batchedTok/s", "data": [float(v) for v in vals], "backgroundColor": colors[idx % len(colors)]})

    llama_datasets = []
    for idx, arm_raw in enumerate(llama_arms_for_tables):
        data_vals = [lookup.get((arm_raw, "", length)) for kernel, length in lair_combos]  # repeat per-shape so bars align with labels
        if all(v is not None for v in data_vals):
            llama_datasets.append({"label": f"{arm_raw} pp tok/s", "data": [float(v) for v in data_vals], "backgroundColor": colors[(len(lair_arms_for_tables) + idx) % len(colors)]})

    ratio_datasets = []
    for idx, pair in enumerate(pairs):
        vals = [entry["ratio_lair_over_llama_pct"] if entry.get("kernel") else None for entry in pair["shapes"]]
        if all(v is not None for v in vals) and len(vals) == len(lair_combos):
            ratio_datasets.append({
                "label": f"threads={pair['threads']} LAIR/llama %",
                "data": [float(v) for v in vals],
                "backgroundColor": colors[idx % len(colors)],
            })

    bar_config = {"type": "bar", "data": {"labels": labels, "datasets": lair_datasets}, "options": bar_options}

    paired_config = {
        "type": "bar",
        "data": {"labels": labels, "datasets": lair_datasets + llama_datasets},
        "options": bar_options,
    }

    ratio_options = {
        "scales": {"y": {"title": {"display": True, "text": "LAIR tok/s as % of matched llama pp tok/s"}, "beginAtZero": True}},
        "plugins": {"legend": {"position": "bottom"}},
    }
    ratio_config = {"type": "bar", "data": {"labels": labels, "datasets": ratio_datasets}, "options": ratio_options}

    html = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8" />',
        f"<title>{Path(report_dir).name} - LAIR Xeon thread-posture sweep</title>",
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>',
        "<style>body { font-family: 'Segoe UI', sans-serif; margin: 2rem auto; max-width: 1080px; color: #1c1c1c; } h1 { font-size: 1.45rem; } h2 { font-size: 1.1rem; margin-top: 2.2rem; } .note { color: #444; font-size: 0.9rem; max-width: 76ch; } canvas { max-width: 1020px; }</style>",
        "</head>",
        "<body>",
        f"<h1>{Path(report_dir).name}</h1>",
        f"<p class='note'>Generated at {data['generated_utc']}. Lane: <code>{data['lane']}</code>. Method: {data['method']}</p>",
        "<h2>LAIR batchedTok/s medians by arm, kernel and length</h2>",
        '<canvas id="postureBar" height="420"></canvas>',
        "<script>",
        f"new Chart(document.getElementById('postureBar'), {json.dumps(bar_config)});",
        "</script>",
    ]

    charts = 1
    if llama_datasets:
        html.extend([
            "<h2>LAIR arms beside matched-posture llama pp medians (same shape labels)</h2>",
            '<canvas id="pairedBar" height="460"></canvas>',
            "<script>",
            f"new Chart(document.getElementById('pairedBar'), {json.dumps(paired_config)});",
            "</script>",
        ])
        charts += 1

    if ratio_datasets:
        html.extend([
            "<h2>LAIR/llama matched-posture ratios (%) per shape</h2>",
            '<canvas id="ratioBar" height="400"></canvas>',
            "<script>",
            f"new Chart(document.getElementById('ratioBar'), {json.dumps(ratio_config)});",
            "</script>",
        ])
        charts += 1

    html.extend([
        "<p class='note'>See <a href='summary.md'><code>summary.md</code></a> for posture-scaling tables, noise bands and separation verdicts and <a href='data.json'><code>data.json</code></a> for machine-readable medians.</p>",
        "</body>",
        "</html>",
    ])

    (report_dir / "report.html").write_text("\n".join(html) + "\n", encoding="utf-8")
    log_count = len(list(raw_dir.glob("posture-r*.txt")))
    warm_count = len(list(raw_dir.glob("posture-warm*.txt")))
    print(f"wrote {report_dir} logs={log_count} warmup_logs={warm_count} rows={len(rows)} charts={charts}", flush=True)


if __name__ == "__main__":
    main()

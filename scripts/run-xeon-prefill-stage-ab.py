#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import subprocess


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(r"([a-z_]+)=([\d.]+)")
IDENT_RE = re.compile(r"\b(kernel|rowWidth|length)=([A-Za-z0-9._-]+)")
LOG_NAME_RE = re.compile(r"^stage-ab-r(\d+)-(.+)\.txt$")

BASELINE_ARM = "off"


def parse_args():
    parser = argparse.ArgumentParser(description="Run interleaved same-session LAIR prefill stage arm A/B measurement.")
    parser.add_argument("--root", default="/root/lair-measurements/qwen25-small-cpu")
    parser.add_argument("--app")
    parser.add_argument("--model")
    parser.add_argument("--dotnet", default="/usr/local/dotnet/dotnet")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--widths", default="64")
    parser.add_argument("--lengths", default="128,256")
    parser.add_argument("--kernels", default="gemv,tiled")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--arms", default="off,on,16")
    parser.add_argument("--warmup-rounds", type=int, default=2, help="alternating arm runs executed before the measured rounds and excluded from parsing (cold-start/page-cache handling)")
    parser.add_argument("--no-rotate", action="store_true", help="keep a fixed arm order in every measured round instead of reversing it on odd rounds")
    parser.add_argument("--report-name", default="optimize-prefill-q4k-route-ab-interleaved-xeon-t8")
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


def run(cmd, label, out_path):
    print(f"RUN[{label}]: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}: {proc.stdout}")
    out_path.write_text(proc.stdout, encoding="utf-8")


def parse_samples(raw_dir):
    samples = {}
    widths_observed = set()

    for raw in sorted(raw_dir.glob("stage-ab-r*.txt")):
        name_match = LOG_NAME_RE.match(raw.name)
        if not name_match:
            continue
        arm = name_match.group(2)

        text = ANSI_RE.sub("", raw.read_text(encoding="utf-8", errors="replace"))
        for line in text.splitlines():
            if "total_ms" not in line or "prefill-profile stage" not in line:
                continue

            ident = dict(IDENT_RE.findall(line))
            metrics = {k: v for k, v in METRIC_RE.findall(line) if k.endswith("_ms")}
            kernel = ident.get("kernel")
            length_str = ident.get("length")
            row_width = ident.get("rowWidth")
            if not kernel or not metrics:
                continue

            try:
                length = int(length_str)
            except ValueError:
                continue

            if row_width is not None:
                widths_observed.add(row_width)

            for metric, value in metrics.items():
                add(samples, f"{arm}|{kernel}|{length}|{metric}", float(value))

    return samples, sorted(widths_observed)


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    app = Path(args.app) if args.app else root / "app" / "Lair.Bench.dll"
    model = Path(args.model) if args.model else root / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    if not arms:
        raise SystemExit("--arms must list at least one arm")

    started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp_dir = datetime.strptime(started_utc, "%Y-%m-%dT%H:%M:%SZ").strftime(f"reports/%Y-%m-%dT%H-%M-%SZ-{args.report_name}")
    report_dir = root / Path(stamp_dir)
    raw_dir = report_dir / "logs"

    def bench_cmd(arm):
        return [
            str(args.dotnet), str(app), "prefill-profile", str(model),
            "--threads", str(args.threads),
            "--lengths", args.lengths,
            "--widths", args.widths,
            "--kernel", args.kernels,
            "--reps", "1",
            "--stage-profile",
            "--attention-batch-dots", "on",
            "--matmul-route", "on",
            "--q4k-matmul-route", arm,
        ]

    def measured_order(round_index):
        return list(reversed(arms)) if (round_index % 2 and not args.no_rotate) else arms

    schedule = [("warmup", arm, f"stage-ab-warm{w}-{arm}.txt") for w in range(args.warmup_rounds) for arm in arms]
    schedule += [(f"ab-round{r}", arm, f"stage-ab-r{r}-{arm}.txt") for r in range(args.rounds) for arm in measured_order(r)]

    if args.dry_run:
        print(f"dry-run report_dir={report_dir} warmup_rounds={args.warmup_rounds} rounds={args.rounds} rotate={not args.no_rotate}", flush=True)
        for label, arm, out_name in schedule:
            print(f"RUN[{label}-arm-{arm}]: {' '.join(bench_cmd(arm))} -> {raw_dir / out_name}", flush=True)
        return

    report_dir.mkdir(parents=True, exist_ok=False)
    raw_dir.mkdir(parents=True, exist_ok=False)

    for label, arm, out_name in schedule:
        run(bench_cmd(arm), f"{label}-arm-{arm}", raw_dir / out_name)


    samples, widths_observed = parse_samples(raw_dir)

    parsed = []
    combo_set = set()
    for key in samples:
        arm, kernel, length_str, metric = key.split("|", 3)
        length = int(length_str)
        parsed.append((arm, kernel, length, metric))
        if metric == "total_ms":
            combo_set.add((kernel, length))

    if not combo_set:
        raise SystemExit("no stage-profile total_ms rows found in interleaved logs")

    kernel_order = [x.strip() for x in args.kernels.split(",") if x.strip()]

    def kernel_rank(kernel):
        return (kernel_order.index(kernel), kernel) if kernel in kernel_order else (99, kernel)

    all_kernels = sorted({kernel for _, kernel, _, _ in parsed}, key=kernel_rank)
    length_values = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    extra_lengths = sorted(length for length in {length for _, _, length, _ in parsed} if length not in length_values)
    all_lengths = [*length_values, *extra_lengths]

    arms_seen = [arm for arm in arms if any(a == arm for a, *_ in parsed)]
    baseline_arm = BASELINE_ARM if BASELINE_ARM in arms_seen else (arms_seen[0] if arms_seen else None)

    combos = [(kernel, length) for kernel in all_kernels for length in all_lengths if (kernel, length) in combo_set]
    combo_index = {(kernel, length): idx for idx, (kernel, length) in enumerate(combos)}

    rows = []
    lookup = {}
    spread_lookup = {}
    for arm, kernel, length, metric in sorted(parsed, key=lambda item: (item[0], combo_index.get((item[1], item[2]), 99), item[3])):
        if (kernel, length) not in combo_set:
            continue

        values = samples[f"{arm}|{kernel}|{length}|{metric}"]
        med = median(values)
        row_key = f"stage|{arm}|{kernel}|{length}|{metric}"
        rows.append({
            "key": row_key,
            "arm": arm,
            "kernel": kernel,
            "length": length,
            "metric": metric,
            "samples": [f"{v:.3f}" for v in values],
            "median": med,
            "min": min(values),
            "max": max(values),
            "count": len(values),
            "spread": spread_of(values, med),
        })

    for row in rows:
        lookup[(row["arm"], row["kernel"], row["length"], row["metric"])] = float(row["median"])
        spread_lookup[(row["arm"], row["kernel"], row["length"], row["metric"])] = float(row["spread"])

    def value(arm, kernel, length, metric):
        return lookup.get((arm, kernel, length, metric))

    comparison = {}
    if baseline_arm is not None:
        for arm in arms_seen:
            if arm == baseline_arm:
                continue

            shape_entries = []
            candidate_spreads_total = []
            all_separate = True
            any_entry = False
            for kernel, length in combos:
                base_med = value(baseline_arm, kernel, length, "total_ms")
                cand_med = value(arm, kernel, length, "total_ms")
                if base_med is None or cand_med is None:
                    continue

                any_entry = True
                delta_pct = (cand_med - base_med) / base_med * 100.0 if base_med else 0.0
                cand_spread = spread_lookup.get((arm, kernel, length, "total_ms"), 0.0)
                base_spread = spread_lookup.get((baseline_arm, kernel, length, "total_ms"), 0.0)
                candidate_spreads_total.append(cand_spread)
                band_pct = max(cand_spread, base_spread) * 100.0
                separates = abs(delta_pct) > band_pct
                if not separates:
                    all_separate = False
                shape_entries.append({
                    "kernel": kernel,
                    "length": length,
                    "baseline_median_ms": round(base_med, 3),
                    "arm_median_ms": round(cand_med, 3),
                    "delta_pct": round(delta_pct, 2),
                    "arm_spread_at_shape_pct": round(cand_spread * 100.0, 2),
                    "baseline_spread_at_shape_pct": round(base_spread * 100.0, 2),
                    "noise_band_pct": round(band_pct, 2),
                    "separates_beyond_noise": separates,
                })

            pooled_worst = max(candidate_spreads_total) if candidate_spreads_total else None
            verdict = (
                f"{'yes' if all_separate and any_entry else 'no'}: arm `{arm}` total_ms deltas vs baseline `{baseline_arm}` "
                + ("exceed the noise band on every shape." if all_separate and any_entry else "do not exceed the noise band on every shape.")
            )
            comparison[arm] = {
                "shapes": shape_entries,
                "pooled_worst_case_total_ms_spread_pct": round(pooled_worst * 100.0, 2) if pooled_worst is not None else None,
                "verdict": verdict,
            }

    labels = [f"{kernel} {length}" for kernel, length in combos]
    colors = ["#3b6fb6", "#f2a65e", "#2a7f62", "#c0603f", "#8a5b00", "#9aa5b1"]

    bar_datasets = []
    for idx, arm in enumerate(arms_seen):
        vals = [value(arm, kernel, length, "total_ms") for kernel, length in combos]
        if all(v is not None for v in vals):
            bar_datasets.append({
                "label": f"arm {arm} total_ms",
                "data": [float(v) for v in vals],
                "backgroundColor": colors[idx % len(colors)],
            })

    spread_datasets = []
    for idx, arm in enumerate(arms_seen):
        vals = [spread_lookup.get((arm, kernel, length, "total_ms")) for kernel, length in combos]
        if all(v is not None for v in vals):
            spread_datasets.append({
                "label": f"arm {arm} intra-arm total_ms spread %",
                "data": [float(v) * 100.0 for v in vals],
                "backgroundColor": colors[idx % len(colors)],
            })

    bar_config = {
        "type": "bar",
        "data": {"labels": labels, "datasets": bar_datasets},
        "options": {
            "scales": {"y": {"title": {"display": True, "text": "total_ms median (lower is better)"}, "beginAtZero": True}},
            "plugins": {"legend": {"position": "bottom"}},
        },
    }

    spread_config = {
        "type": "bar",
        "data": {"labels": labels, "datasets": spread_datasets},
        "options": {
            "scales": {"y": {"title": {"display": True, "text": "(max-min)/median % per arm"}, "beginAtZero": True}},
            "plugins": {"legend": {"position": "bottom"}},
        },
    }

    data = {
        "report_started_utc": started_utc,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lane": f"Xeon CPU-only lair-lab (root@192.168.33.53), threads={args.threads}, rowWidth={args.widths}",
        "model_file": str(model),
        "lair_binary": str(app),
        "method": (
            f"{args.warmup_rounds} warmup rounds then {args.rounds} measured rounds x arms [{', '.join(arms)}] interleaved back-to-back in one session so host drift cancels; "
            f"measured arm order rotates on odd rounds to cancel positional bias while warmups run each arm once per round for cold-start/page-cache handling; "
            f"reps=1 per command with kernels {', '.join(kernel_order)}; medians/min/max/spread parsed from raw measured logs after ANSI stripping."
        ),
        "rounds": args.rounds,
        "warmup_rounds": args.warmup_rounds,
        "rotate_arm_order": not args.no_rotate,
        "arms": arms_seen,
        "baseline_arm": baseline_arm,
        "row_widths_observed": widths_observed,
        "rows": rows,
        "comparison": comparison,
    }

    (report_dir / "data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    summary = [f"# Prefill q4k route A/B interleaved Xeon t8 {Path(report_dir).name}", "", f"Lane: {data['lane']}. Method: {data['method']}", ""]
    for length in all_lengths:
        summary.append(f"## Length {length} medians (ms)")
        for kernel, combo_length in combos:
            if combo_length != length:
                continue

            required = ["total_ms", "feed_forward_gate_up_ms", "feed_forward_down_ms", "attention_weight_ms"]
            table_rows = []
            for arm in arms_seen:
                vals = [value(arm, kernel, length, metric) for metric in required]
                if any(v is None for v in vals):
                    continue

                table_rows.append("| {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} |".format(arm, *[float(v) for v in vals]))

            if not table_rows:
                continue

            summary.append(f"### {kernel}")
            summary.append("| arm | total | gate/up | down | attention_weight |")
            summary.append("| --- | ---: | ---: | ---: | ---: |")
            summary.extend(table_rows)
            summary.append("")

    summary.append("## Noise bands (intra-arm total_ms spread = (max-min)/median)")
    header_cells = ["kernel", "length"] + [f"arm {arm} %" for arm in arms_seen]
    summary.append("| " + " | ".join(header_cells) + " |")
    summary.append("| " + " | ".join(["---:"] * len(header_cells)) + " |")
    for kernel, length in combos:
        cells = [kernel, str(length)]
        for arm in arms_seen:
            sp = spread_lookup.get((arm, kernel, length, "total_ms"))
            cells.append(f"{sp * 100.0:.2f}" if sp is not None else "-")
        summary.append("| " + " | ".join(cells) + " |")
    for arm in arms_seen:
        pooled = max((spread_lookup.get((arm, kernel, length, "total_ms"), 0.0) for kernel, length in combos)) if combos else None
        if pooled is not None:
            summary.append(f"- pooled worst-case total_ms spread for arm `{arm}`: {pooled * 100.0:.2f}%")
    summary.append("")

    summary.append("## Interpretation (total_ms medians vs baseline)")
    summary.append(f"Baseline arm: `{baseline_arm}`.")
    if baseline_arm is not None:
        for kernel, length in combos:
            base_med = value(baseline_arm, kernel, length, "total_ms")
            if base_med is None:
                continue

            parts = []
            for arm in arms_seen:
                if arm == baseline_arm:
                    continue

                cand_med = value(arm, kernel, length, "total_ms")
                if cand_med is None:
                    continue

                delta_pct = (cand_med - base_med) / base_med * 100.0 if base_med else 0.0
                band_pct = max(spread_lookup.get((arm, kernel, length, "total_ms"), 0.0), spread_lookup.get((baseline_arm, kernel, length, "total_ms"), 0.0)) * 100.0
                separates = abs(delta_pct) > band_pct
                parts.append(f"`{arm}` delta {delta_pct:+.2f}% vs noise band ±{band_pct:.2f}% -> separates beyond own spread: {'yes' if separates else 'no'}")

            summary.append(f"- `{kernel}` length {length}: " + "; ".join(parts))
    summary.extend(["", ""])
    (report_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    html = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8" />',
        f"<title>{Path(report_dir).name} - LAIR Xeon q4k route A/B interleaved</title>",
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>',
        "<style>body { font-family: 'Segoe UI', sans-serif; margin: 2rem auto; max-width: 1080px; color: #1c1c1c; } h1 { font-size: 1.45rem; } h2 { font-size: 1.1rem; margin-top: 2.2rem; } .note { color: #444; font-size: 0.9rem; max-width: 76ch; } canvas { max-width: 1020px; }</style>",
        "</head>",
        "<body>",
        f"<h1>{Path(report_dir).name}</h1>",
        f"<p class='note'>Generated at {data['generated_utc']}. Lane: <code>{data['lane']}</code>. Method: {data['method']}</p>",
        "<h2>total_ms medians by arm, kernel and length</h2>",
        '<canvas id="abBar" height="420"></canvas>',
        "<script>",
        f"new Chart(document.getElementById('abBar'), {json.dumps(bar_config)});",
        "</script>",
    ]

    if spread_datasets:
        html.extend([
            "<h2>Intra-arm total_ms spread per shape</h2>",
            '<canvas id="spreadBar" height="380"></canvas>',
            "<script>",
            f"new Chart(document.getElementById('spreadBar'), {json.dumps(spread_config)});",
            "</script>",
        ])

    html.extend([
        "<p class='note'>See <a href='summary.md'><code>summary.md</code></a> for tables and noise bands and <a href='data.json'><code>data.json</code></a> for machine-readable medians.</p>",
        "</body>",
        "</html>",
    ])

    (report_dir / "report.html").write_text("\n".join(html) + "\n", encoding="utf-8")
    log_count = len(list(raw_dir.glob("stage-ab-r*.txt")))
    warm_count = len(list(raw_dir.glob("stage-ab-warm*.txt")))
    print(f"wrote {report_dir} logs={log_count} warmup_logs={warm_count} rows={len(rows)} charts={'2' if spread_datasets else '1'}", flush=True)


if __name__ == "__main__":
    main()

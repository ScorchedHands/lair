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

KERNELS_DEFAULT = "gemv,tiled,gemm"
LENGTHS_DEFAULT = "128,256"


def parse_args():
    parser = argparse.ArgumentParser(description="Run and parse a bounded LAIR prefill stage profile.")
    parser.add_argument("--root", default="/root/lair-measurements/qwen25-small-cpu")
    parser.add_argument("--app")
    parser.add_argument("--model")
    parser.add_argument("--dotnet", default="/usr/local/dotnet/dotnet")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--widths", default="64")
    parser.add_argument("--lengths", default=LENGTHS_DEFAULT)
    parser.add_argument("--kernels", default=KERNELS_DEFAULT)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--attention-batch-dots", default="on")
    parser.add_argument("--matmul-route", default="on")
    parser.add_argument("--q4k-matmul-route")
    return parser.parse_args()


def median(values):
    values = sorted(values)
    n = len(values)
    if not n:
        raise ValueError("no samples")
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2.0


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

    for raw in sorted(raw_dir.glob("stage-t8-round*.txt")):
        text = ANSI_RE.sub("", raw.read_text(encoding="utf-8", errors="replace"))
        for line in text.splitlines():
            if "total_ms" not in line or "prefill-profile stage" not in line:
                continue

            ident = dict(IDENT_RE.findall(line))
            metrics = {k: v for k, v in METRIC_RE.findall(line) if k.endswith("_ms")}
            kernel = ident.get("kernel")
            length_str = ident.get("length")
            if not kernel or not metrics:
                continue

            try:
                length = int(length_str)
            except ValueError:
                continue

            for metric, value in metrics.items():
                add(samples, f"{kernel}|{length}|{metric}", float(value))

    return samples


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    app = Path(args.app) if args.app else root / "app" / "Lair.Bench.dll"
    model = Path(args.model) if args.model else root / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

    started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp_dir = datetime.strptime(started_utc, "%Y-%m-%dT%H:%M:%SZ").strftime("reports/%Y-%m-%dT%H-%M-%SZ-optimize-prefill-stage-profile-xeon-t8")
    report_dir = root / Path(stamp_dir)
    raw_dir = report_dir / "logs"
    report_dir.mkdir(parents=True, exist_ok=False)
    raw_dir.mkdir(parents=True, exist_ok=False)

    cmd_extra = [
        "--attention-batch-dots", args.attention_batch_dots,
        "--matmul-route", args.matmul_route,
    ]
    if args.q4k_matmul_route:
        cmd_extra.extend(["--q4k-matmul-route", args.q4k_matmul_route])

    for round_index in range(args.rounds):
        out_path = raw_dir / f"stage-t8-round{round_index}.txt"
        run([str(args.dotnet), str(app), "prefill-profile", str(model), "--threads", str(args.threads), "--lengths", args.lengths, "--widths", args.widths, "--kernel", args.kernels, "--reps", "1", "--stage-profile"] + cmd_extra, f"stage-round{round_index}", out_path)

    samples = parse_samples(raw_dir)
    parsed = []
    combo_set = set()
    for key in samples:
        kernel, length_str, metric = key.split("|", 2)
        length = int(length_str)
        parsed.append((kernel, length, metric))
        if metric == "total_ms":
            combo_set.add((kernel, length))

    if not combo_set:
        raise SystemExit("no stage-profile total_ms rows found")

    kernel_order = [x.strip() for x in args.kernels.split(",") if x.strip()]

    def kernel_rank(kernel):
        return (kernel_order.index(kernel), kernel) if kernel in kernel_order else (99, kernel)

    all_kernels = sorted({kernel for kernel, _, _ in parsed}, key=kernel_rank)
    length_values = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    extra_lengths = sorted(length for length in {length for _, length, _ in parsed} if length not in length_values)
    all_lengths = [*length_values, *extra_lengths]

    combos = [(kernel, length) for kernel in all_kernels for length in all_lengths if (kernel, length) in combo_set]
    combo_index = {(kernel, length): idx for idx, (kernel, length) in enumerate(combos)}

    rows = []
    lookup = {}
    for kernel, length, metric in sorted(parsed, key=lambda item: (combo_index.get((item[0], item[1]), 99), item[2])):
        if (kernel, length) not in combo_set:
            continue

        values = samples[f"{kernel}|{length}|{metric}"]
        row_key = f"stage|{kernel}|{length}|{metric}"
        rows.append({
            "key": row_key,
            "kernel": kernel,
            "length": length,
            "metric": metric,
            "samples": [f"{v:.3f}" for v in values],
            "median": median(values),
            "min": min(values),
            "max": max(values),
        })

    for row in rows:
        lookup[row["key"]] = float(row["median"])

    def value(kernel, length, metric):
        return lookup.get(f"stage|{kernel}|{length}|{metric}")

    labels = [f"{kernel} {length}" for kernel, length in combos]
    chart_metric_names = ["feed_forward_gate_up_ms", "feed_forward_down_ms", "attention_weight_ms", "attention_projection_ms", "attention_output_ms", "attention_kernel_ms", "logits_head_ms"]
    metric_labels = {
        "total_ms": "Total window ms (lower is better)",
        "attention_weight_ms": "Attention weight traversal ms",
        "attention_projection_ms": "Attention projection ms",
        "attention_output_ms": "Attention output ms",
        "attention_kernel_ms": "Attention kernel ms",
        "feed_forward_gate_up_ms": "Feed-forward gate/up ms",
        "feed_forward_down_ms": "Feed-forward down ms",
        "logits_head_ms": "Logit head ms",
    }

    datasets = []
    colors = ["#3b6fb6", "#f2a65e", "#2a7f62", "#c0603f", "#8a5b00", "#9aa5b1", "#4d4d4d", "#6b4c9a", "#3d8b9e"]
    for idx, metric in enumerate(chart_metric_names):
        vals = [value(kernel, length, metric) for kernel, length in combos]
        if all(v is not None for v in vals):
            datasets.append({
                "label": metric_labels.get(metric, metric),
                "data": [float(v) for v in vals],
                "backgroundColor": colors[idx % len(colors)],
            })

    if not datasets:
        total_vals = []
        for kernel, length in combos:
            total_vals.append(value(kernel, length, "total_ms") or 0.0)
        datasets.append({
            "label": metric_labels["total_ms"],
            "data": [float(v) for v in total_vals],
            "backgroundColor": colors[0],
        })

    bar_config = {
        "type": "bar",
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "scales": {"y": {"title": {"display": True, "text": "ms median (lower is better)"}, "beginAtZero": True}},
            "plugins": {"legend": {"position": "bottom"}},
        },
    }

    pie_metric_keys = ["attention_weight_ms", "feed_forward_gate_up_ms", "feed_forward_down_ms"]
    preferred_pie_combo = ("gemv", 256) if value("gemv", 256, "total_ms") is not None else combos[0]

    def default_value(kernel, length, metric):
        return value(kernel, length, metric) or 0.0

    pie_has_metrics = all(value(preferred_pie_combo[0], preferred_pie_combo[1], m) is not None for m in pie_metric_keys) and value(preferred_pie_combo[0], preferred_pie_combo[1], "total_ms") is not None
    other_slices = {}

    if pie_has_metrics:
        kernel, length = preferred_pie_combo
        components = [
            "attention_weight_ms",
            "feed_forward_gate_up_ms",
            "feed_forward_down_ms",
            "attention_projection_ms",
            "attention_output_ms",
            "attention_kernel_ms",
            "logits_head_ms",
        ]
        other_slices["seed/cache/kv/logits/other"] = default_value(kernel, length, "logits_head_ms") + (default_value(kernel, length, "total_ms") - sum(default_value(kernel, length, m) for m in components))

    pie_config = None
    if pie_has_metrics:
        kernel, length = preferred_pie_combo
        pie_labels = [*pie_metric_keys, *other_slices.keys()]
        pie_values = [default_value(kernel, length, metric) for metric in pie_metric_keys] + list(other_slices.values())
        pie_config = {
            "type": "pie",
            "data": {
                "labels": pie_labels,
                "datasets": [{
                    "label": f"{kernel} length {length} stage medians (ms)",
                    "data": [float(v) for v in pie_values],
                    "backgroundColor": ["#3b6fb6", "#f2a65e", "#2a7f62", "#c9d4dd"][:len(pie_labels)],
                }],
            },
        }

    data = {
        "report_started_utc": started_utc,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lane": f"Xeon CPU-only lair-lab (root@192.168.33.53), threads={args.threads}, rowWidth={args.widths}",
        "model_file": str(model),
        "lair_binary": str(app),
        "method": f"{args.rounds} alternating same-session stage-profile reps=1 commands with kernels {', '.join(kernel_order)}; medians/min/max parsed from raw logs after ANSI stripping.",
        "rounds": args.rounds,
        "rows": rows,
    }

    (report_dir / "data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    summary = [f"# Prefill stage profile Xeon t8 {Path(report_dir).name}", "", f"Lane: {data['lane']}. Method: {data['method']}", ""]
    for length in all_lengths:
        summary.append(f"## Length {length} medians (ms)")
        summary.append("| kernel | total | gate/up | down | attention_weight | projection | output | kernel | logits |")
        summary.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        for kernel, combo_length in combos:
            if combo_length != length:
                continue

            required = ["total_ms", "feed_forward_gate_up_ms", "feed_forward_down_ms", "attention_weight_ms", "attention_projection_ms", "attention_output_ms", "attention_kernel_ms", "logits_head_ms"]
            vals = [value(kernel, length, metric) for metric in required]
            if any(v is None for v in vals):
                continue

            summary.append("| {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} |".format(kernel, *[float(v) for v in vals]))
        summary.append("")

    summary.append("Interpretation by total_ms medians:")
    for length in all_lengths:
        rows_for_length = [(kernel, value(kernel, length, "total_ms") or 0.0) for kernel, combo_length in combos if combo_length == length and value(kernel, length, "total_ms") is not None]
        if not rows_for_length:
            continue

        fastest_kernel = min(rows_for_length, key=lambda item: item[1])[0]
        slowest_kernel = max(rows_for_length, key=lambda item: item[1])[0]
        summary.append(f"- length {length}: fastest total_ms kernel is `{fastest_kernel}`; slowest is `{slowest_kernel}`.")

    if pie_has_metrics:
        kernel, length = preferred_pie_combo
        stage_values = [(metric, value(kernel, length, metric) or 0.0) for metric in chart_metric_names if value(kernel, length, metric) is not None]
        top_stages = sorted(stage_values, key=lambda item: item[1], reverse=True)[:3]
        summary.append(f"- dominant stages around `{kernel}` length {length}: " + ", ".join(f"`{metric}`" for metric, _ in top_stages))

    summary.extend([
        "",
    ])
    (report_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    html = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8" />',
        f"<title>{Path(report_dir).name} - LAIR Xeon stage profile</title>",
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>',
        "<style>body { font-family: 'Segoe UI', sans-serif; margin: 2rem auto; max-width: 1080px; color: #1c1c1c; } h1 { font-size: 1.45rem; } h2 { font-size: 1.1rem; margin-top: 2.2rem; } .note { color: #444; font-size: 0.9rem; max-width: 76ch; } canvas { max-width: 1020px; }</style>",
        "</head>",
        "<body>",
        f"<h1>{Path(report_dir).name}</h1>",
        f"<p class='note'>Generated at {data['generated_utc']}. Lane: <code>{data['lane']}</code>. Method: {data['method']}</p>",
        "<h2>Stage medians by kernel and length</h2>",
        '<canvas id="stageBar" height="420"></canvas>',
        "<script>",
        f"new Chart(document.getElementById('stageBar'), {json.dumps(bar_config)});",
        "</script>",
    ]

    if pie_config is not None:
        html.extend([
            "<h2>Stage shares</h2>",
            '<canvas id="gemvPie" height="380"></canvas>',
            "<script>",
            f"new Chart(document.getElementById('gemvPie'), {json.dumps(pie_config)});",
            "</script>",
        ])

    html.extend([
        "<p class='note'>See <a href='summary.md'><code>summary.md</code></a> for tables and <a href='data.json'><code>data.json</code></a> for machine-readable medians.</p>",
        "</body>",
        "</html>",
    ])

    (report_dir / "report.html").write_text("\n".join(html) + "\n", encoding="utf-8")
    print(f"wrote {report_dir} rows={len(rows)} charts={'2' if pie_config is not None else '1'}", flush=True)


if __name__ == "__main__":
    main()

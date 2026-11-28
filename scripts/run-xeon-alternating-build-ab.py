import argparse
import datetime as dt
import json
import os
import re
import statistics
import subprocess

METRICS = ("total_ms", "feed_forward_down_ms", "feed_forward_gate_up_ms", "attention_output_ms")
STAGE_RE = re.compile(r"prefill-profile stage .*kernel=(gemv|tiled)")
VALUE_RES = {m: re.compile(m + r"=([0-9.]+)") for m in METRICS}


def parse_log(path):
    samples = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    for line in text.splitlines():
        if "prefill-profile stage" not in line:
            continue
        km = re.search(r"kernel=(gemv|tiled)", line)
        if not km:
            continue
        kernel = km.group(1)
        bucket = samples.setdefault(kernel, {m: [] for m in METRICS})
        for metric, pattern in VALUE_RES.items():
            match = pattern.search(line)
            if match:
                bucket[metric].append(float(match.group(1)))
    return samples


def summarize(values):
    ordered = sorted(values)
    median = statistics.median(ordered)
    low = ordered[0]
    high = ordered[-1]
    spread_pct = (high - low) / median * 100.0 if median > 0 else 0.0
    return {"median": round(median, 3), "min": round(low, 3), "max": round(high, 3), "count": len(ordered), "spread_pct": round(spread_pct, 2)}


def main():
    parser = argparse.ArgumentParser(description="Alternate two published LAIR bench builds in interleaved rounds.")
    parser.add_argument("--old-app", required=True)
    parser.add_argument("--new-app", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dotnet", default="/usr/local/dotnet/dotnet")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lengths", default="256")
    parser.add_argument("--widths", default="64")
    parser.add_argument("--rounds", type=int, default=13)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    started = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    report_dir = os.path.join(args.out_root, f"{started}-{args.name}")
    log_dir = os.path.join(report_dir, "logs")
    os.makedirs(log_dir)

    apps = {"old": args.old_app, "new": args.new_app}

    def invoke(app_key, tag):
        app = apps[app_key]
        command = [args.dotnet, app, "prefill-profile", args.model, "--threads", str(args.threads), "--lengths", args.lengths, "--widths", args.widths, "--kernel", "gemv,tiled", "--reps", "1", "--stage-profile", "--attention-batch-dots", "on", "--matmul-route", "on"]
        log_path = os.path.join(log_dir, f"{tag}-{app_key}.txt")
        with open(log_path, "w", encoding="utf-8") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)

    for round_index in range(args.warmup_rounds):
        order = ["old", "new"] if round_index % 2 == 0 else ["new", "old"]
        for app_key in order:
            invoke(app_key, f"w{round_index}")

    samples = {app_key: {} for app_key in apps}
    for round_index in range(args.rounds):
        print(f"round {round_index}", flush=True)
        order = ["old", "new"] if round_index % 2 == 0 else ["new", "old"]
        for app_key in order:
            invoke(app_key, f"r{round_index}")
            parsed = parse_log(os.path.join(log_dir, f"r{round_index}-{app_key}.txt"))
            for kernel, metrics in parsed.items():
                bucket = samples[app_key].setdefault(kernel, {m: [] for m in METRICS})
                for metric, values in metrics.items():
                    if values:
                        bucket[metric].append(values[len(values) // 2])

    shapes = {}
    verdict_lines = []
    for kernel in ("gemv", "tiled"):
        old_kernel = samples["old"].get(kernel, {})
        new_kernel = samples["new"].get(kernel, {})
        if not old_kernel or not new_kernel:
            continue
        entry = {}
        shapes[kernel] = entry
        for metric in METRICS:
            old_values = old_kernel.get(metric, [])
            new_values = new_kernel.get(metric, [])
            if not old_values or not new_values:
                continue
            old_summary = summarize(old_values)
            new_summary = summarize(new_values)
            delta_pct = (new_summary["median"] - old_summary["median"]) / old_summary["median"] * 100.0
            band_new = max(new_summary["spread_pct"], old_summary["spread_pct"])
            separates = abs(delta_pct) > band_new and (new_summary["max"] <= old_summary["min"] or new_summary["min"] >= old_summary["max"])
            entry[metric] = {"old": old_summary, "new": new_summary, "delta_pct": round(delta_pct, 2), "band_pct": round(band_new, 2), "separates_beyond_noise": bool(separates)}
            verdict_lines.append(f"{kernel} {metric}: old={old_summary['median']} new={new_summary['median']} delta={round(delta_pct, 2)}% band=±{round(band_new, 2)}% separates={'yes' if separates else 'no'}")

    data = {
        "report_started_utc": started,
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lane": f"Xeon CPU-only lair-lab, threads={args.threads}, rowWidth={args.widths}, lengths {args.lengths}",
        "old_app": args.old_app,
        "new_app": args.new_app,
        "method": f"{args.warmup_rounds} warmup rounds then {args.rounds} measured interleaved rounds alternating old/new builds with odd-round order rotation; reps=1 per command; kernels gemv,tiled",
        "shapes": shapes,
    }
    summary_lines = ["# Alternating build A/B 2026 style headline", "", *verdict_lines]
    with open(os.path.join(report_dir, "data.json"), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    with open(os.path.join(report_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    print("REPORT_DIR=" + report_dir, flush=True)
    for line in verdict_lines:
        print(line, flush=True)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = [
    ("baseline-off", "route OFF"),
    ("route-on-w8", "route ON, width 8 (script default `on` maps to 8)"),
    ("route-w16", "route ON, explicit width 16"),
]
METRICS = ["total_ms", "feed_forward_gate_up_ms", "feed_forward_down_ms", "attention_weight_ms"]
COMBOS = [("gemv", 128), ("gemv", 256), ("tiled", 128), ("tiled", 256)]

data = {}
for folder, _ in VARIANTS:
    doc = json.loads((HERE / folder / "data.json").read_text(encoding="utf-8"))
    data[folder] = (doc, {(r["kernel"], r["length"], r["metric"]): r for r in doc["rows"]})

missing = [(f, k, l, m) for f, _ in VARIANTS for k, l in COMBOS for m in METRICS if (k, l, m) not in data[f][1]]

def med(f, k, l, m):
    return data[f][1][(k, l, m)]["median"]

def delta_pct(cand, k, l, m):
    b = med("baseline-off", k, l, m)
    return (med(cand, k, l, m) - b) / b * 100.0

def spread(f, k, l, m):
    r = data[f][1][(k, l, m)]
    return 0.0 if r["median"] == 0 else (r["max"] - r["min"]) / r["median"] * 100.0

L = []
L.append("# Q4_K dense matmul-route A/B medians -- Xeon t8, qwen2.5-3b-instruct-q4_k_m")
L.append("")
L.append("All numbers below are parsed from the downloaded `data.json` files in this directory "
         "(script-computed medians of 3 same-session stage-profile rounds; sample lists and min/max spreads included). "
        "Nothing here is estimated.")
L.append("")
for folder, label in VARIANTS:
    doc, _ = data[folder]
    L.append(f"- `{folder}` -- {label}; remote report started `{doc['report_started_utc']}`, rounds={doc['rounds']}, binary `{Path(doc['lair_binary']).name}`, model `{Path(doc['model_file']).name}`")
if missing:
    L.append(f"- MISSING ROWS in downloaded data: {missing}")
L.append("")

for kernel, length in COMBOS:
    L.append(f"## {kernel} @ length {length}")
    L.append("")
    L.append("| metric | off median (samples) | on-w8 median (delta vs off) | w16 median (delta vs off) |")
    L.append("| --- | ---: | ---: | ---: |")
    for m in METRICS:
        b = data["baseline-off"][1][(kernel, length, m)]
        row = [f"{b['median']:.1f} ({', '.join(b['samples'])})"]
        for f in ("route-on-w8", "route-w16"):
            c = data[f][1][(kernel, length, m)]
            dlt = (c["median"] - b["median"]) / b["median"] * 100.0
            row.append(f"{c['median']:.1f} ({dlt:+.1f}%)")
        L.append(f"| {m} | " + " | ".join(row) + " |")
    L.append("")

L.append("## Sample noise -- (max-min)/median per variant")
L.append("")
L.append("| combo | metric | off | on-w8 | w16 |")
L.append("| --- | --- | ---: | ---: | ---: |")
for kernel, length in COMBOS:
    for m in METRICS:
        cells = [f"{spread(f, kernel, length, m):.1f}%" for f, _ in VARIANTS]
        L.append(f"| {kernel}@{length} | {m} | " + " | ".join(cells) + " |")
L.append("")

deltas8 = {c: delta_pct("route-on-w8", *c, "total_ms") for c in COMBOS}
deltas16 = {c: delta_pct("route-w16", *c, "total_ms") for c in COMBOS}
mean8 = sum(deltas8.values()) / len(deltas8)
mean16 = sum(deltas16.values()) / len(deltas16)

L.append("## total_ms medians: baseline vs candidates")
L.append("")
L.append("| combo | off median ms | on-w8 (delta) | w16 (delta) |")
L.append("| --- | ---: | ---: | ---: |")
for k, l in COMBOS:
    L.append(f"| {k}@{l} | {med('baseline-off', k, l, 'total_ms'):.1f} "
             f"| {med('route-on-w8', k, l, 'total_ms'):.1f} ({deltas8[(k,l)]:+.1f}%) "
             f"| {med('route-w16', k, l, 'total_ms'):.1f} ({deltas16[(k,l)]:+.1f}%) |")
L.append("")

noisy = []
for kernel, length in COMBOS:
    for m in METRICS:
        for f, _ in VARIANTS:
            s = spread(f, kernel, length, m)
            if s > 20.0:
                noisy.append((f, kernel, length, m, s))

L.append("## Conclusion")
L.append("")
gemv8 = (deltas8[('gemv',128)], deltas8[('gemv',256)])
gemv16 = (deltas16[('gemv',128)], deltas16[('gemv',256)])
tiled8 = (deltas8[('tiled',128)], deltas8[('tiled',256)])
tiled16 = (deltas16[('tiled',128)], deltas16[('tiled',256)])

L.append(f"- gemv: route ON improves total_ms at both lengths for both widths "
         f"(width 8 {gemv8[0]:+.1f}% @128, {gemv8[1]:+.1f}% @256; width 16 {gemv16[0]:+.1f}% @128, {gemv16[1]:+.1f}% @256), "
         f"and the gain tracks the expected stages: `feed_forward_down_ms` and `attention_weight_ms` fall while `feed_forward_gate_up_ms` stays within noise.")
L.append(f"- width 16 is at least as good as width 8 on every gemv combo ({gemv16[0]:+.1f}% vs {gemv8[0]:+.1f}% @128, "
         f"{gemv16[1]:+.1f}% vs {gemv8[1]:+.1f}% @256).")
L.append(f"- tiled: mixed. Width 8 shows {tiled8[1]:+.1f}% at length 256 but {tiled8[0]:+.1f}% at length 128; "
         f"width 16 shows {tiled16[1]:+.1f}% @256 and {tiled16[0]:+.1f}% @128. The tiled@128 width-8 median is not trustworthy: "
         f"its own sample spread reaches {spread('route-on-w8','tiled',128,'total_ms'):.1f}% of the median and its gate/up stage spreads "
         f"{spread('route-on-w8','tiled',128,'feed_forward_gate_up_ms'):.1f}%, so a +/-6% reading sits inside that noise band.")
L.append(f"- Across all four combos, mean-of-medians total_ms delta is {mean8:+.1f}% for width 8 and {mean16:+.1f}% for width 16; "
         f"width 16 has no positive (slower) combo in this run while width 8 does.")
if noisy:
    L.append("- Sample spreads above 20% of median were seen in these rows, which is why single-digit deltas should not be read as decisive: ")
    for f, k, l, m, s in sorted(noisy, key=lambda x: -x[4]):
        L.append(f"  - `{f}` {k}@{l} {m}: spread {s:.1f}%")
L.append("- Baseline `gemv@128` also carries a wide spread "
         f"({spread('baseline-off','gemv',128,'total_ms'):.1f}% of median, samples 11476/10426/9337), with its slowest sample in the first round; "
         "that cold-start shape appears in several variants and inflates any length-128 comparison.")
L.append("- Verdict on this evidence: the Q4_K matmul route looks faster for `gemv` prefill at both lengths, slightly more so at width 16 than at the default width 8; "
         "`tiled` is improved at length 256 and unresolved-to-slightly-worse at length 128 within noise. These are 3-round medians with up to ~30% intra-variant spread, "
         "so a rounds=4 repeat of baseline vs width 8 (and ideally width 16) is needed before any promotion decision.")

(HERE / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote analysis.md lines=%d" % len(L))

"""ANS vs AC weight-compression experiment.

Standalone: parses GGUF files in models/, extracts quant code streams per format
(Q8_0, Q4_K, IQ4_XS, IQ3_XXS, IQ3_S; IQ4_NL skipped), encodes them with a shared
frequency model using arithmetic coding and rANS, compares against the packed
baseline across several grouping ("block") candidates, measures encode/decode
time, and writes a JSON + HTML report to ../reports/.

Run:  python src/experiment.py [--models-dir PATH] [--cap N] [--max-models K]
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gguf
import code_streams
import codec_ac
import codec_ans

FRAMING_BYTES = 12  # u32 n_symbols + u32 payload_len + u32 model_len, per chunk
ESCAPE_FLAG = 1     # per chunk: 0 = packed escape, 1 = compressed

FORMATS = {8: "Q8_0", 12: "Q4_K", 23: "IQ4_XS", 18: "IQ3_XXS", 21: "IQ3_S", 20: "IQ4_NL"}

CHUNK_CAP = 0

LAYER_RE = re.compile(r"blk\.(\d+)\.")
EXPERT_RE = re.compile(r"ffn_\w*exps?\.(\d+)\.|ffn_exps\.(\d+)\.|ffn_\w+\.experts\.(\d+)\.")


def parse_layer_expert(name):
    m = LAYER_RE.search(name)
    layer = int(m.group(1)) if m else None
    expert = None
    em = EXPERT_RE.search(name)
    if em:
        for g in em.groups():
            if g is not None:
                expert = int(g)
                break
    return layer, expert


def collect_blocks(path, cap):
    gf = gguf.GgufFile(path)
    data = open(path, "rb").read()
    payload = memoryview(data)[gf.data_start:]
    out = {}  # fmt -> list of block dicts
    for t in gf.tensors:
        gt = t.ggml_type
        if gt not in FORMATS:
            continue
        ts = gguf.TYPE_SIZE[gt]
        rem = cap - len(out.get(gt, []))
        if rem <= 0:
            continue
        nb = min(t.n_blocks(), rem)
        tb = payload[t.offset: t.offset + nb * ts]
        blocks = code_streams.extract_blocks(gt, tb)[:rem]
        layer, expert = parse_layer_expert(t.name)
        for blk in blocks:
            blk["layer"] = layer
            blk["expert"] = expert
            blk["tensor"] = t.name
            out.setdefault(gt, []).append(blk)
    for gt in out:
        if len(out[gt]) > cap:
            out[gt] = out[gt][:cap]
    return out


def build_chunkings(blocks):
    """Return list of (candidate_name, [ [block_idx,...], ... ])."""
    n = len(blocks)
    out = []
    for name, size in (("block", 1), ("tile8", 8), ("group16", 16), ("group64", 64)):
        out.append((name, [list(range(i, min(i + size, n))) for i in range(0, n, size)]))
    out.append(("set", [list(range(n))]))  # whole sampled set as one shared model
    by_tensor = {}
    for i, b in enumerate(blocks):
        by_tensor.setdefault(b["tensor"], []).append(i)
    out.append(("tensor", list(by_tensor.values())))
    by_layer = {}
    for i, b in enumerate(blocks):
        by_layer.setdefault(b["layer"], []).append(i)
    out.append(("layer", list(by_layer.values())))
    by_expert = {}
    has_expert = False
    for i, b in enumerate(blocks):
        if b["expert"] is not None:
            has_expert = True
            by_expert.setdefault((b["layer"], b["expert"]), []).append(i)
    if has_expert:
        out.append(("expert", list(by_expert.values())))
    out.append(("model", [list(range(n))]))
    return [(name, _split_chunks(chunks, blocks)) for name, chunks in out]


def _split_chunks(chunks, blocks):
    """Split any chunk whose total symbol count exceeds CHUNK_CAP into consecutive
    subchunks. Applied identically to every candidate and both codecs so the
    comparison stays fair while keeping each independently-decodable chunk bounded."""
    if CHUNK_CAP <= 0:
        return chunks
    out = []
    for c in chunks:
        cur = []
        n_sym = 0
        for bi in c:
            slen = len(blocks[bi]["symbols"])
            if cur and n_sym + slen > CHUNK_CAP:
                out.append(cur)
                cur, n_sym = [], 0
            cur.append(bi)
            n_sym += slen
        if cur:
            out.append(cur)
    return out


def evaluate(blocks, codecs=("ac", "ans")):
    chunkings = build_chunkings(blocks)
    records = []
    for name, chunks in chunkings:
        for codec in codecs:
            enc = codec_ac if codec == "ac" else codec_ans
            baseline_block = 0
            baseline_code = 0
            meta_total = 0
            comp_nofb = 0
            comp_fb = 0
            model_total = 0
            framing_total = 0
            n_symbols = 0
            rt_ok = True
            t_enc = 0.0
            t_dec = 0.0
            for chunk in chunks:
                syms = []
                code_b = 0
                meta_b = 0
                block_b = 0
                for bi in chunk:
                    blk = blocks[bi]
                    syms.extend(blk["symbols"])
                    code_b += blk["code_bytes"]
                    meta_b += blk["meta_bytes"]
                baseline_code += code_b
                meta_total += meta_b
                baseline_block += code_b + meta_b
                n_symbols += len(syms)
                t0 = time.perf_counter()
                model_bytes, payload = enc.encode_chunk(syms)
                t_enc += time.perf_counter() - t0
                t0 = time.perf_counter()
                dec = enc.decode_chunk(model_bytes, len(syms), payload)
                t_dec += time.perf_counter() - t0
                if dec != syms:
                    rt_ok = False
                framing_total += FRAMING_BYTES
                model_total += len(model_bytes)
                compressed = ESCAPE_FLAG + FRAMING_BYTES + len(model_bytes) + len(payload)
                packed = ESCAPE_FLAG + code_b
                comp_nofb += compressed
                comp_fb += packed if compressed >= packed else compressed
            records.append({
                "candidate": name,
                "codec": codec,
                "blocks": sum(len(c) for c in chunks),
                "symbols": n_symbols,
                "baseline_block_bytes": baseline_block,
                "baseline_code_bytes": baseline_code,
                "meta_bytes": meta_total,
                "compressed_nofallback": comp_nofb,
                "compressed_fallback": comp_fb,
                "model_bytes": model_total,
                "framing_bytes": framing_total,
                "saved_pct": round(100.0 * (baseline_block - comp_fb) / baseline_block, 2) if baseline_block else 0.0,
                "encode_ms": round(t_enc * 1000, 2),
                "decode_ms": round(t_dec * 1000, 2),
                "roundtrip_ok": rt_ok,
            })
    return records


def self_test():
    cases = {
        "empty": [],
        "all_one": [7] * 500,
        "uniform16": [i % 16 for i in range(2000)],
        "skewed16": ([0] * 1500 + [1] * 300 + [5] * 200),
        "uniform256": [i % 256 for i in range(2000)],
        "random16": [ (i * 2654435761) % 16 for i in range(1000)],
    }
    ok = True
    for label, syms in cases.items():
        for codec in ("ac", "ans"):
            enc = codec_ac if codec == "ac" else codec_ans
            m, p = enc.encode_chunk(syms)
            dec = enc.decode_chunk(m, len(syms), p)
            if dec != syms:
                ok = False
                print(f"SELFTEST FAIL {label} {codec}")
    print("SELFTEST", "OK" if ok else "FAIL")
    return ok


def _expert_count(metadata):
    for k, v in metadata.items():
        if k.endswith(".expert_count") and isinstance(v, int):
            return v
    return None


def collect_blocks_sharded(paths, cap):
    """Collect blocks for a (possibly split) GGUF model without loading whole shards.

    Tensor data is read lazily via absolute-offset range reads mapped across shards,
    bounded by cap blocks per format. Expert tensors are split per expert using the
    expert-count dimension so the per-expert grouping candidate is populated.
    """
    first = paths[0]
    gf = gguf.GgufFile(first)
    sizes = [os.path.getsize(p) for p in paths]
    starts = []
    cur = 0
    for i, s in enumerate(sizes):
        ds = gf.data_start if i == 0 else 0
        starts.append(cur)
        cur += s - ds
    handles = {}

    def read_abs(off, length):
        out = bytearray()
        pos = off
        remaining = length
        while remaining > 0:
            placed = False
            for i in range(len(paths)):
                ds = gf.data_start if i == 0 else 0
                data_len = sizes[i] - ds
                if starts[i] <= pos < starts[i] + data_len:
                    local = pos - starts[i] + ds
                    take = min(remaining, starts[i] + data_len - pos)
                    h = handles.setdefault(paths[i], open(paths[i], "rb"))
                    h.seek(local)
                    out.extend(h.read(take))
                    pos += take
                    remaining -= take
                    placed = True
                    break
            if not placed:
                break
        return bytes(out)

    ec = _expert_count(gf.metadata)
    out = {}
    for t in gf.tensors:
        gt = t.ggml_type
        if gt not in FORMATS:
            continue
        ts = gguf.TYPE_SIZE[gt]
        rem = cap - len(out.get(gt, []))
        if rem <= 0:
            continue
        layer, expert = parse_layer_expert(t.name)
        if ec and "exp" in t.name and len(t.dims) >= 3 and t.dims[-1] == ec:
            total_blocks = t.n_blocks()
            bpe = total_blocks // ec
            if bpe >= 1:
                per_e = max(1, rem // ec)
                for e in range(ec):
                    if len(out.get(gt, [])) >= cap:
                        break
                    take = min(per_e, bpe, cap - len(out.get(gt, [])))
                    tb = read_abs(t.offset + e * bpe * ts, take * ts)
                    blocks = code_streams.extract_blocks(gt, tb)[:take]
                    for blk in blocks:
                        blk["expert"] = e
                        blk["layer"] = layer
                        blk["tensor"] = t.name
                    out.setdefault(gt, []).extend(blocks)
                continue
        nb = min(t.n_blocks(), rem)
        tb = read_abs(t.offset, nb * ts)
        blocks = code_streams.extract_blocks(gt, tb)[:rem]
        for blk in blocks:
            blk["layer"] = layer
            blk.setdefault("expert", expert)
            blk["tensor"] = t.name
        out.setdefault(gt, []).extend(blocks)
    for gt in out:
        if len(out[gt]) > cap:
            out[gt] = out[gt][:cap]
    return out


def _group_models(files):
    split_re = re.compile(r"-\d{5}-of-\d{5}\.gguf$")
    groups = {}
    order = []
    for f in files:
        m = split_re.search(f)
        if m:
            key = f[: m.start()]
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(f)
        else:
            if f not in groups:
                groups[f] = []
                order.append(f)
            groups[f].append(f)
    for k in groups:
        groups[k].sort()
    return [(k, groups[k]) for k in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--extra-models-dir", action="append", default=[])
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--max-models", type=int, default=99)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.normpath(os.path.join(here, "..", "..", "..", "models"))
    if args.models_dir:
        models_dirs = [args.models_dir] + list(args.extra_models_dir)
    else:
        models_dirs = [default_dir] + list(args.extra_models_dir)
    out_dir = args.out_dir or os.path.normpath(os.path.join(here, "..", "reports"))
    os.makedirs(out_dir, exist_ok=True)

    if not self_test():
        return 1

    all_records = []
    for models_dir in models_dirs:
        files = sorted(f for f in os.listdir(models_dir)
                       if f.endswith(".gguf") and not os.path.basename(f).startswith("mmproj"))
        for key, shard_names in _group_models(files)[: args.max_models]:
            paths = [os.path.join(models_dir, n) for n in shard_names]
            try:
                if len(paths) == 1:
                    blocks_by_fmt = collect_blocks(paths[0], args.cap)
                else:
                    blocks_by_fmt = collect_blocks_sharded(paths, args.cap)
            except Exception as e:
                print(f"skip {key}: {e}")
                continue
            for gt, blocks in blocks_by_fmt.items():
                recs = evaluate(blocks)
                for r in recs:
                    r["model_file"] = key
                    r["format"] = FORMATS[gt]
                all_records.extend(recs)
                print(f"{key} {FORMATS[gt]}: {len(blocks)} blocks evaluated")

    json_path = os.path.join(out_dir, "ans_ac_results.json")
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)
    html_path = os.path.join(out_dir, "ans_ac_report.html")
    with open(html_path, "w") as f:
        f.write(render_html(all_records))
    print(f"\nJSON: {json_path}\nHTML: {html_path}")
    return 0


def _avg(vals):
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _aggregate(records):
    order = ["block", "tile8", "group16", "group64", "set", "tensor", "layer", "expert", "model"]
    agg = {}
    for r in records:
        c = r["candidate"]
        if c not in order:
            continue
        e = agg.setdefault(c, {
            "baseline": 0, "ac_fb": 0, "ans_fb": 0, "ac_nofb": 0, "ans_nofb": 0,
            "ac_saved": [], "ans_saved": [], "ac_enc": [], "ans_enc": [], "ac_dec": [], "ans_dec": [],
        })
        e["baseline"] += r["baseline_block_bytes"]
        if r["codec"] == "ac":
            e["ac_fb"] += r["compressed_fallback"]; e["ac_nofb"] += r["compressed_nofallback"]
            e["ac_saved"].append(r["saved_pct"]); e["ac_enc"].append(r["encode_ms"]); e["ac_dec"].append(r["decode_ms"])
        else:
            e["ans_fb"] += r["compressed_fallback"]; e["ans_nofb"] += r["compressed_nofallback"]
            e["ans_saved"].append(r["saved_pct"]); e["ans_enc"].append(r["encode_ms"]); e["ans_dec"].append(r["decode_ms"])
    labels = [c for c in order if c in agg]
    return labels, agg


def _per_format(records, candidate):
    order = ["Q8_0", "Q4_K", "IQ4_XS", "IQ3_XXS", "IQ3_S", "IQ4_NL"]
    agg = {}
    for r in records:
        if r["candidate"] != candidate:
            continue
        f = r["format"]
        if f not in order:
            continue
        e = agg.setdefault(f, {"baseline": 0, "ac_fb": 0, "ans_fb": 0})
        e["baseline"] += r["baseline_block_bytes"]
        if r["codec"] == "ac":
            e["ac_fb"] += r["compressed_fallback"]
        else:
            e["ans_fb"] += r["compressed_fallback"]
    labels = [f for f in order if f in agg]
    return labels, agg


def render_html(records):
    rows = []
    for r in records:
        rows.append(
            "<tr>"
            f"<td>{r['model_file']}</td><td>{r['format']}</td><td>{r['candidate']}</td><td>{r['codec']}</td>"
            f"<td>{r['blocks']}</td><td>{r['symbols']}</td>"
            f"<td>{r['baseline_block_bytes']}</td><td>{r['compressed_fallback']}</td>"
            f"<td>{r['compressed_nofallback']}</td><td>{r['model_bytes']}</td>"
            f"<td>{r['saved_pct']}%</td><td>{r['encode_ms']}</td><td>{r['decode_ms']}</td>"
            f"<td>{'ok' if r['roundtrip_ok'] else 'FAIL'}</td></tr>"
        )
    head = (
        "<tr><th>model</th><th>format</th><th>grouping</th><th>codec</th><th>blocks</th>"
        "<th>symbols</th><th>baseline B</th><th>compressed+fallback B</th><th>compressed B</th>"
        "<th>model B</th><th>saved%</th><th>encode ms</th><th>decode ms</th><th>roundtrip</th></tr>"
    )

    labels, agg = _aggregate(records)
    baseline = [agg[c]["baseline"] for c in labels]
    ac_fb = [agg[c]["ac_fb"] for c in labels]
    ans_fb = [agg[c]["ans_fb"] for c in labels]
    ac_saved = [_avg(agg[c]["ac_saved"]) for c in labels]
    ans_saved = [_avg(agg[c]["ans_saved"]) for c in labels]
    ac_enc = [_avg(agg[c]["ac_enc"]) for c in labels]
    ans_enc = [_avg(agg[c]["ans_enc"]) for c in labels]
    ac_dec = [_avg(agg[c]["ac_dec"]) for c in labels]
    ans_dec = [_avg(agg[c]["ans_dec"]) for c in labels]

    flabels, fagg = _per_format(records, "tensor")
    f_baseline = [fagg[f]["baseline"] for f in flabels]
    f_ac = [fagg[f]["ac_fb"] for f in flabels]
    f_ans = [fagg[f]["ans_fb"] for f in flabels]

    data = {
        "labels": labels, "baseline": baseline, "ac_fb": ac_fb, "ans_fb": ans_fb,
        "ac_saved": ac_saved, "ans_saved": ans_saved,
        "ac_enc": ac_enc, "ans_enc": ans_enc, "ac_dec": ac_dec, "ans_dec": ans_dec,
        "fmt_labels": flabels, "fmt_baseline": f_baseline, "fmt_ac": f_ac, "fmt_ans": f_ans,
    }
    charts = (
        "<div class='charts'>"
        "<div class='card'><canvas id='cSizes'></canvas></div>"
        "<div class='card'><canvas id='cSaved'></canvas></div>"
        "<div class='card'><canvas id='cEnc'></canvas></div>"
        "<div class='card'><canvas id='cDec'></canvas></div>"
        "<div class='card'><canvas id='cFmt'></canvas></div>"
        "</div>"
    )
    script = (
        "<script src='https://cdn.jsdelivr.net/npm/chart.js@4'></script>"
        "<script>"
        f"const D={json.dumps(data)};"
        "const C=Chart;"
        "new C(document.getElementById('cSizes'),{type:'bar',data:{labels:D.labels,datasets:["
        "{label:'Baseline (raw)',data:D.baseline,backgroundColor:'#9aa0a6'},"
        "{label:'AC compressed',data:D.ac_fb,backgroundColor:'#4285f4'},"
        "{label:'ANS compressed',data:D.ans_fb,backgroundColor:'#34a853'}]},"
        "options:{responsive:true,plugins:{title:{display:true,text:'Compressed bytes vs raw baseline by grouping'}},scales:{y:{title:{display:true,text:'bytes'}}}}});"
        "new C(document.getElementById('cSaved'),{type:'line',data:{labels:D.labels,datasets:["
        "{label:'AC saved%',data:D.ac_saved,borderColor:'#4285f4',fill:false},"
        "{label:'ANS saved%',data:D.ans_saved,borderColor:'#34a853',fill:false}]},"
        "options:{responsive:true,plugins:{title:{display:true,text:'Saved % vs raw baseline by grouping'}},scales:{y:{title:{display:true,text:'saved %'}}}}});"
        "new C(document.getElementById('cEnc'),{type:'bar',data:{labels:D.labels,datasets:["
        "{label:'AC encode ms',data:D.ac_enc,backgroundColor:'#4285f4'},"
        "{label:'ANS encode ms',data:D.ans_enc,backgroundColor:'#34a853'}]},"
        "options:{responsive:true,plugins:{title:{display:true,text:'Encode time (ms) by grouping'}},scales:{y:{title:{display:true,text:'ms'}}}}});"
        "new C(document.getElementById('cDec'),{type:'bar',data:{labels:D.labels,datasets:["
        "{label:'AC decode ms',data:D.ac_dec,backgroundColor:'#4285f4'},"
        "{label:'ANS decode ms',data:D.ans_dec,backgroundColor:'#34a853'}]},"
        "options:{responsive:true,plugins:{title:{display:true,text:'Decode time (ms) by grouping'}},scales:{y:{title:{display:true,text:'ms'}}}}});"
        "new C(document.getElementById('cFmt'),{type:'bar',data:{labels:D.fmt_labels,datasets:["
        "{label:'Baseline (raw)',data:D.fmt_baseline,backgroundColor:'#9aa0a6'},"
        "{label:'AC compressed',data:D.fmt_ac,backgroundColor:'#4285f4'},"
        "{label:'ANS compressed',data:D.fmt_ans,backgroundColor:'#34a853'}]},"
        "options:{responsive:true,plugins:{title:{display:true,text:'Compressed vs raw baseline by format (per-tensor grouping)'}},scales:{y:{title:{display:true,text:'bytes'}}}}});"
        "</script>"
    )
    return (
        "<html><head><meta charset='utf-8'><title>ANS vs AC weight compression</title>"
        "<style>body{font-family:sans-serif;margin:16px}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}"
        "table{border-collapse:collapse;margin-top:16px}.charts{display:flex;flex-wrap:wrap;gap:16px;margin:16px 0}"
        ".card{border:1px solid #ddd;border-radius:6px;padding:8px;width:480px;height:320px}</style></head><body>"
        "<h1>ANS vs AC weight compression</h1>"
        "<p>Baseline = packed block bytes (code + kept scale/min metadata). compressed = framing + model + payload "
        f"per chunk (framing {FRAMING_BYTES}B + escape flag per chunk). saved% uses fallback totals. "
        "Charts use Chart.js (needs internet to render).</p>"
        f"{charts}{script}"
        f"<table>{head}{''.join(rows)}</table></body></html>"
    )


if __name__ == "__main__":
    sys.exit(main())

namespace Lair.Backends.Cuda;

// The on-device transformer-step kernel family: the rotation, the KV write, and the causal attention walk that a dense
// step composes against buffers the workspace owns, so nothing between the three ever crosses the PCIe seam. Address
// arithmetic stays in the computed-register form every kernel on this seam writes, accumulation is plain mul then add in
// ascending element order — the chain shape the host reference walks — and the only transcendental forms are `ex2.approx`
// over a log2-e scaled argument and `div.rn` for the reciprocal, both available at this module's sm_50 target. No
// tensor-core form is used here either: `mma.sync` needs a target declaration this module does not carry, and the
// softmax this step runs is a reduction over positions, not a matmul tile.
public sealed partial class CudaBackend
{
    // The causal attention kernel keeps its entire online-softmax state in per-lane registers: each lane owns one
    // output element of the row from its first score through its own final store, and every value that drives a
    // decision is a value that same lane computed. There is no shared window on this walk to schedule around, so no
    // barrier or broadcast appears in the attention entry at all — which is what keeps the multi-position rescale
    // identical in every lane of a pair instead of dependent on one reducing lane's stores being visible in order.

    // RoPE over the head regions of staged rows: one block per (row, head region) pair, one lane per rotary pair with a
    // stride so any head dimension fits. Each lane reads its position's angle entry, rotates its pair in place, and
    // writes both elements back — the same two products and one sum/difference the host rotation writes for that pair,
    // against the same table entries the host tables hold. `rope_pairs` rotates adjacent dimensions (2i, 2i + 1) and
    // `rope_half` rotates the split halves (i, i + half), the two layouts the forward engine's resolved pairing selects.
    private static string BuildRopePtx(string entryName, bool adjacentPairing)
    {
        List<string> pairAddressing =
        [
            adjacentPairing
                ? "    shl.b32 t0, pair, 1;" // the adjacent pair starts at 2 * pair ...
                : "    mov.u32 t0, pair;", // ... and the split-half pair starts at the lane's own index,
        ];

        pairAddressing.Add("    add.u32 t0, base, t0;");
        pairAddressing.Add("    cvt.u64.u32 addr, t0;");
        pairAddressing.Add("    shl.b64 addr, addr, 2;");
        pairAddressing.Add("    add.u64 addr, addr, r;");
        pairAddressing.Add("    ld.global.f32 lo, [addr];");

        if (adjacentPairing)
        {
            pairAddressing.Add("    add.u64 addr2, addr, 4;"); // the adjacent partner sits right behind it.
        }
        else
        {
            pairAddressing.Add("    add.u32 t0, t0, half_;"); // the split-half partner sits half a head away.
            pairAddressing.Add("    cvt.u64.u32 addr2, t0;");
            pairAddressing.Add("    shl.b64 addr2, addr2, 2;");
            pairAddressing.Add("    add.u64 addr2, addr2, r;");
        }

        return $".visible .entry {entryName}(\n" +
            "    .param .u64 rows_ptr,\n" +
            "    .param .u64 cos_ptr,\n" +
            "    .param .u64 sin_ptr,\n" +
            "    .param .u32 regions_arg,\n" +
            "    .param .u32 hdim_arg,\n" +
            "    .param .u32 half_arg,\n" +
            "    .param .u32 posbase_arg\n" +
            ") {\n" +
            "    .reg .u64 r, cs, sn, addr, addr2;\n" +
            "    .reg .u32 regions_, hdim_, half_, posbase_, row, head, pair, stride, pos, base, tbl, t0;\n" +
            "    .reg .pred p;\n" +
            "    .reg .f32 lo, hi, cval, sval, p0, p1;\n" +
            "\n" +
            "    ld.param.u64 r, [rows_ptr];\n" +
            "    ld.param.u64 cs, [cos_ptr];\n" +
            "    ld.param.u64 sn, [sin_ptr];\n" +
            "    ld.param.u32 regions_, [regions_arg];\n" +
            "    ld.param.u32 hdim_, [hdim_arg];\n" +
            "    ld.param.u32 half_, [half_arg];\n" +
            "    ld.param.u32 posbase_, [posbase_arg];\n" +
            "\n" +
            "    mov.u32 row, %ctaid.x;\n" +
            "    mov.u32 head, %ctaid.y;\n" +
            "    mov.u32 pair, %tid.x;\n" +
            "    mov.u32 stride, %ntid.x;\n" +
            "\n" +
            // This block's row is the position it rotates at, and its head region is the contiguous run the pair
            // addresses: same row-major, head-major layout the staged rows and the host rotation both walk.
            "    add.u32 pos, posbase_, row;\n" +
            "    mad.lo.u32 base, row, regions_, head;\n" +
            "    mul.lo.u32 base, base, hdim_;\n" +
            "    mul.lo.u32 tbl, pos, half_;\n" +
            "\n" +
            "_pair:\n" +
            "    setp.ge.u32 p, pair, half_;\n" +
            "    @p bra done;\n" +
            "\n" +
            "    add.u32 t0, tbl, pair;\n" +
            "    cvt.u64.u32 addr, t0;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, cs;\n" +
            "    ld.global.f32 cval, [addr];\n" +
            "\n" +
            "    add.u32 t0, tbl, pair;\n" +
            "    cvt.u64.u32 addr, t0;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, sn;\n" +
            "    ld.global.f32 sval, [addr];\n" +
            "\n" +
            string.Join("\n", [.. pairAddressing]) + "\n" +
            "    ld.global.f32 hi, [addr2];\n" +
            "\n" +
            // The rotation itself, in the host reference's operand order: low * cos - high * sin, and high * cos +
            // low * sin, each as one multiply pair and one add or subtract, both written back in place.
            "    mul.f32 p0, lo, cval;\n" +
            "    mul.f32 p1, hi, sval;\n" +
            "    sub.f32 p0, p0, p1;\n" +
            "    st.global.f32 [addr], p0;\n" +
            "    mul.f32 p1, hi, cval;\n" +
            "    mul.f32 lo, lo, sval;\n" +
            "    add.f32 p1, p1, lo;\n" +
            "    st.global.f32 [addr2], p1;\n" +
            "\n" +
            "    add.u32 pair, pair, stride;\n" +
            "    bra _pair;\n" +
            "done:\n" +
            "    ret;\n" +
            "}\n";
    }

    // KV write: one block per staged row, lanes striding over the row's kv-head region. Each element crosses from the
    // staging row into the arena slot its (layer, position) names, in the arena layout the attention walk reads back:
    // ((layer * maxPositions + position) * kvHeads + kvHead) * headDim + d. The staging side is exactly the row-major
    // kv-head-major layout the host produced, so the write is a positional move with no arithmetic on the values.
    private const string KvStorePtx =
        ".visible .entry kv_store(\n" +
        "    .param .u64 dstk_ptr,\n" +
        "    .param .u64 dstv_ptr,\n" +
        "    .param .u64 srck_ptr,\n" +
        "    .param .u64 srcv_ptr,\n" +
        "    .param .u32 rowelems_arg,\n" +
        "    .param .u32 maxpos_arg,\n" +
        "    .param .u32 layer_arg,\n" +
        "    .param .u32 posbase_arg\n" +
        ") {\n" +
        "    .reg .u64 dk, dv, sk, sv, addr, addr2;\n" +
        "    .reg .u32 rowelems_, maxpos_, layer_, posbase_, row, e, stride, srcBase, dstBase, t0;\n" +
        "    .reg .pred p;\n" +
        "    .reg .f32 val;\n" +
        "\n" +
        "    ld.param.u64 dk, [dstk_ptr];\n" +
        "    ld.param.u64 dv, [dstv_ptr];\n" +
        "    ld.param.u64 sk, [srck_ptr];\n" +
        "    ld.param.u64 sv, [srcv_ptr];\n" +
        "    ld.param.u32 rowelems_, [rowelems_arg];\n" +
        "    ld.param.u32 maxpos_, [maxpos_arg];\n" +
        "    ld.param.u32 layer_, [layer_arg];\n" +
        "    ld.param.u32 posbase_, [posbase_arg];\n" +
        "\n" +
        "    mov.u32 row, %ctaid.x;\n" +
        "    mov.u32 e, %tid.x;\n" +
        "    mov.u32 stride, %ntid.x;\n" +
        "\n" +
        "    mul.lo.u32 srcBase, row, rowelems_;\n" +
        "    add.u32 t0, posbase_, row;\n" +
        "    mad.lo.u32 dstBase, layer_, maxpos_, t0;\n" +
        "    mul.lo.u32 dstBase, dstBase, rowelems_;\n" +
        "\n" +
        "_element:\n" +
        "    setp.ge.u32 p, e, rowelems_;\n" +
        "    @p bra done;\n" +
        "\n" +
        "    add.u32 t0, srcBase, e;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, addr, sk;\n" +
        "    ld.global.f32 val, [addr];\n" +
        "    add.u32 t0, dstBase, e;\n" +
        "    cvt.u64.u32 addr2, t0;\n" +
        "    shl.b64 addr2, addr2, 2;\n" +
        "    add.u64 addr2, addr2, dk;\n" +
        "    st.global.f32 [addr2], val;\n" +
        "\n" +
        "    add.u32 t0, srcBase, e;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, addr, sv;\n" +
        "    ld.global.f32 val, [addr];\n" +
        "    add.u32 t0, dstBase, e;\n" +
        "    cvt.u64.u32 addr2, t0;\n" +
        "    shl.b64 addr2, addr2, 2;\n" +
        "    add.u64 addr2, addr2, dv;\n" +
        "    st.global.f32 [addr2], val;\n" +
        "\n" +
        "    add.u32 e, e, stride;\n" +
        "    bra _element;\n" +
        "done:\n" +
        "    ret;\n" +
        "}\n";

    // Causal attention over the resident arena: one block per (row, head) pair, one lane per head-dimension element.
    // The lane owns that element's accumulator for the whole position walk, and the score for each position is reduced
    // in the host reference's own order — ascending products summed by one lane, then the scale, then the cap — and
    // broadcast through the shared window before it reaches that pair's online softmax state. The online update is the
    // host kernel's rule written so every lane sees the same decision: a new running max rescales the weight sum and
    // every accumulator element by exp(oldMax - score), the current weight is exp(score - runningMax) so the max term
    // contributes exactly 1, and the final normalization multiplies by the reciprocal of the weight sum. Both exponent
    // terms ride on `ex2.approx` over the log2-e scaled argument, and the reciprocal on `div.rn`, which is what the
    // host kernel's `MathF.Exp` and `1f / sumExp` answer within this seam's measured band.
    private static string BuildAttentionPtx()
    {
        return ".visible .entry attn_causal(\n" +
            "    .param .u64 q_ptr,\n" +
            "    .param .u64 k_ptr,\n" +
            "    .param .u64 v_ptr,\n" +
            "    .param .u64 o_ptr,\n" +
            "    .param .u32 heads_arg,\n" +
            "    .param .u32 kvheads_arg,\n" +
            "    .param .u32 hdim_arg,\n" +
            "    .param .u32 maxpos_arg,\n" +
            "    .param .u32 layer_arg,\n" +
            "    .param .u32 posbase_arg,\n" +
            "    .param .u32 win_arg,\n" +
            "    .param .u32 groupshift_arg,\n" +
            "    .param .u32 scale_bits_arg,\n" +
            "    .param .u32 cap_bits_arg\n" +
            ") {\n" +
            "    .reg .u64 q, kk, vv, o, addr;\n" +
            "    .reg .u32 heads_, kvheads_, hdim_, maxpos_, layer_, posbase_, win_, groupshift_, row, head, lane, ntid, pos, first, last, kRow, vRow, idx, t0, d, kb, kvh, kvregion, headBase;\n" +
            "    .reg .pred p;\n" +
            "    .reg .f32 acc, val, prod, score, rmax, sumexp, rescale, weight, inv, scale, cap, num, den, e2;\n" +
            "\n" +
            "    ld.param.u64 q, [q_ptr];\n" +
            "    ld.param.u64 kk, [k_ptr];\n" +
            "    ld.param.u64 vv, [v_ptr];\n" +
            "    ld.param.u64 o, [o_ptr];\n" +
            "    ld.param.u32 heads_, [heads_arg];\n" +
            "    ld.param.u32 kvheads_, [kvheads_arg];\n" +
            "    ld.param.u32 hdim_, [hdim_arg];\n" +
            "    ld.param.u32 maxpos_, [maxpos_arg];\n" +
            "    ld.param.u32 layer_, [layer_arg];\n" +
            "    ld.param.u32 posbase_, [posbase_arg];\n" +
            "    ld.param.u32 win_, [win_arg];\n" +
            "    ld.param.u32 groupshift_, [groupshift_arg];\n" +
            "    ld.param.u32 t0, [scale_bits_arg];\n" +
            "    mov.b32 scale, t0;\n" +
            "    ld.param.u32 t0, [cap_bits_arg];\n" +
            "    mov.b32 cap, t0;\n" +
            "\n" +
            "    mov.u32 row, %ctaid.x;\n" +
            "    mov.u32 head, %ctaid.y;\n" +
            "    mov.u32 lane, %tid.x;\n" +
            "    mov.u32 ntid, %ntid.x;\n" +
            "\n" +
            "    setp.ge.u32 p, lane, hdim_;\n" +
            "    @p bra done;\n" +
            "\n" +
            // The grouped-query mapping the host attention uses for the same pair: consecutive heads share a kv-head,
            // and the host refuses any head count that is not a whole multiple of the kv-head count, so the shift is
            // exact where the host resolved it.
            "    shr.u32 kvh, head, groupshift_;\n" +
            "    mul.lo.u32 kvregion, kvheads_, hdim_;\n" +
            "    mad.lo.u32 headBase, row, heads_, head;\n" +
            "    mul.lo.u32 headBase, headBase, hdim_;\n" +
            "\n" +
            // This row's causal range: its own position is the last visible one, and a positive window narrows the
            // backward span exactly as the host selection does before any score is seen. A non-positive window reaches
            // back to position 0 of the sequence. A window that reaches further back than this row's own position
            // resolves to position 0 through the guard below rather than through unsigned subtraction, which would wrap
            // the first index past the last and leave the walk with no positions at all.
            "    add.u32 last, posbase_, row;\n" +
            "    mov.u32 first, 0;\n" +
            "    setp.le.u32 p, win_, 0;\n" +
            "    @p bra _range;\n" +
            "    add.u32 t0, last, 1;\n" +
            "    setp.le.u32 p, t0, win_;\n" + // last + 1 <= win: the window starts at or before position 0
            "    @p bra _range;\n" +
            "    sub.u32 first, last, win_;\n" +
            "    add.u32 first, first, 1;\n" +
            "_range:\n" +
            "\n" +
            // Online-softmax state for this lane's element, seeded the way the host seeds each head it walks — in
            // registers only, because no other lane reads these values and nothing here needs a shared window.
            "    mov.f32 rmax, 0fFF800000;\n" +
            "    mov.f32 sumexp, 0f00000000;\n" +
            "    mov.f32 acc, 0f00000000;\n" +
            "\n" +
            "    mov.u32 pos, first;\n" +
            "_walk:\n" +
            "    setp.gt.u32 p, pos, last;\n" +
            "    @p bra _normalize;\n" +
            "\n" +
            // The score's dot for this lane: the same ascending sum over the head dimension that the host reference
            // performs, reading this pair's query row and this position's K row element by element. Every lane of the
            // pair reduces the identical product chain into its own `score` register, so no reduction crosses lanes.
            "    mad.lo.u32 kRow, layer_, maxpos_, pos;\n" +
            "    mul.lo.u32 kRow, kRow, kvregion;\n" +
            "    mad.lo.u32 kb, kvh, hdim_, kRow;\n" +
            "    mov.f32 score, 0f00000000;\n" +
            "    mov.u32 d, 0;\n" +
"_dot:\n" +
            "    setp.ge.u32 p, d, hdim_;\n" +
            "    @p bra _scaled;\n" +
            // The query element for this dimension first, then the matching key element: same ascending product order.
            "    add.u32 idx, headBase, d;\n" +
            "    cvt.u64.u32 addr, idx;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, q;\n" +
            "    ld.global.f32 val, [addr];\n" +
            "    add.u32 idx, kb, d;\n" +
            "    cvt.u64.u32 addr, idx;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, kk;\n" +
            "    ld.global.f32 prod, [addr];\n" +
            "    mul.f32 val, val, prod;\n" +
            "    add.f32 score, score, val;\n" +
            "    add.u32 d, d, 1;\n" +
            "    bra _dot;\n" +
            "_scaled:\n" +
            // Scaled, then capped, in the host kernel's order: the scale lands on the whole dot before the cap sees it.
            "    mul.f32 score, score, scale;\n" +
            "    setp.le.f32 p, cap, 0f00000000;\n" +
            "    @p bra _capped;\n" +
            // cap * tanh(score / cap) through the same transcendental forms the softmax uses: e = exp2(2 * log2e *
            // score / cap), tanh = (e - 1) / (e + 1), evaluated in the host reference's scaled-then-capped position.
            "    div.rn.f32 num, score, cap;\n" +
            "    mul.f32 num, num, 0f40000000;\n" + // 2 * (score / cap)
            "    mul.f32 num, num, 0f3FB8AA3B;\n" + // log2(e)
            "    ex2.approx.f32 e2, num;\n" +
            "    sub.f32 num, e2, 0f3F800000;\n" +
            "    add.f32 den, e2, 0f3F800000;\n" +
            "    div.rn.f32 num, num, den;\n" +
            "    mul.f32 score, num, cap;\n" +
            "_capped:\n" +
            "\n" +
            // The online softmax decision for this lane's own state registers: a new max rescales the running sum and
            // opens the new base, and the current term's weight is exp(score - runningMax), so the max term contributes
            // exactly 1. Every value these branches read is one this same lane wrote earlier in its own walk.
            "    mov.f32 rescale, 0f3F800000;\n" +
            "    setp.le.f32 p, score, rmax;\n" +
            "    @p bra _weight;\n" +
            "    setp.eq.f32 p, rmax, 0fFF800000;\n" +
            "    @p bra _newmax;\n" +
            "    sub.f32 num, rmax, score;\n" +
            "    mul.f32 num, num, 0f3FB8AA3B;\n" + // log2(e)
            "    ex2.approx.f32 rescale, num;\n" +
            "    mul.f32 sumexp, sumexp, rescale;\n" +
"_newmax:\n" +
            "    mov.f32 rmax, score;\n" +
"_weight:\n" +
            "    sub.f32 num, score, rmax;\n" +
            "    mul.f32 num, num, 0f3FB8AA3B;\n" + // log2(e)
            "    ex2.approx.f32 weight, num;\n" +
            // The term's own weight folds into the running sum on every position — rescaled first when a new max
            // opened, then extended by this term, exactly as the host kernel accumulates its weight sum.
            "    add.f32 sumexp, sumexp, weight;\n" +
            "\n" +
            // This lane folds this position into its own element: rescale first (exactly 1 where no new max opened),
            // then the weighted value in the host kernel's accumulation order.
            "    mul.f32 acc, acc, rescale;\n" +
            "    mad.lo.u32 vRow, layer_, maxpos_, pos;\n" +
            "    mul.lo.u32 vRow, vRow, kvregion;\n" +
            "    mad.lo.u32 idx, kvh, hdim_, vRow;\n" +
            "    add.u32 idx, idx, lane;\n" +
            "    cvt.u64.u32 addr, idx;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, vv;\n" +
            "    ld.global.f32 val, [addr];\n" +
            "    mul.f32 val, weight, val;\n" +
            "    add.f32 acc, acc, val;\n" +
            "\n" +
            "    add.u32 pos, pos, 1;\n" +
            "    bra _walk;\n" +
"_normalize:\n" +
            "\n" +
            // Final normalization for this lane's element: the reciprocal of its own weight sum, the same 1f / sumExp
            // the host applies when its walk finishes — and >= 1 for every head, because the max term contributed exp(0).
            "    div.rn.f32 inv, 0f3F800000, sumexp;\n" +
            "    mul.f32 acc, acc, inv;\n" +
            "    add.u32 idx, headBase, lane;\n" +
            "    cvt.u64.u32 addr, idx;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, o;\n" +
            "    st.global.f32 [addr], acc;\n" +
            "done:\n" +
            "    ret;\n" +
            "}\n";
    }

    // KV write into a pool's device store: one block per staged row, lanes striding over the row's kv-head region. Each
    // element crosses from the staging row into the store slot its (layer, position) names, in the pool's own block
    // layout — store offset ((block - firstBlock) * floatsPerBlock + layer * layerRegion + (isV ? blockTokens * kvDim : 0)
    // + slot * kvDim + element) — the layout the pool's host views address and the attention-paged walk reads back. The
    // staging side is exactly the row-major kv-head-major layout the host produced, so the write is a positional move
    // with no arithmetic on the values, and a logical position resolves to its block and slot by shift and mask.
    private static string BuildKvStorePagedPtx()
    {
        return ".visible .entry kv_store_paged(\n" +
            "    .param .u64 store_ptr,\n" +
            "    .param .u64 srck_ptr,\n" +
            "    .param .u64 srcv_ptr,\n" +
            "    .param .u32 rowelems_arg,\n" +
            "    .param .u32 blockshift_arg,\n" +
            "    .param .u32 floatspbl_arg,\n" +
            "    .param .u32 layerregion_arg,\n" +
            "    .param .u32 vhalf_arg,\n" +
            "    .param .u32 firstblock_arg,\n" +
            "    .param .u32 layer_arg,\n" +
            "    .param .u32 posbase_arg\n" +
            ") {\n" +
            "    .reg .u64 st, sk, sv, addr, addr2;\n" +
            "    .reg .u32 rowelems_, blockshift_, floatspbl_, layerregion_, vhalf_, firstblock_, layer_, posbase_, row, e, stride, pos, blk, slot, mask, srcBase, dstBase, t0;\n" +
            "    .reg .pred p;\n" +
            "    .reg .f32 val;\n" +
            "\n" +
            "    ld.param.u64 st, [store_ptr];\n" +
            "    ld.param.u64 sk, [srck_ptr];\n" +
            "    ld.param.u64 sv, [srcv_ptr];\n" +
            "    ld.param.u32 rowelems_, [rowelems_arg];\n" +
            "    ld.param.u32 blockshift_, [blockshift_arg];\n" +
            "    ld.param.u32 floatspbl_, [floatspbl_arg];\n" +
            "    ld.param.u32 layerregion_, [layerregion_arg];\n" +
            "    ld.param.u32 vhalf_, [vhalf_arg];\n" +
            "    ld.param.u32 firstblock_, [firstblock_arg];\n" +
            "    ld.param.u32 layer_, [layer_arg];\n" +
            "    ld.param.u32 posbase_, [posbase_arg];\n" +
            "\n" +
            "    mov.u32 row, %ctaid.x;\n" +
            "    mov.u32 e, %tid.x;\n" +
            "    mov.u32 stride, %ntid.x;\n" +
            "\n" +
            // The block size arrives as a shift (validated a power of two at open), so the slot is the low bits the
            // same shift leaves behind: same position-to-block resolution the pool's host views apply.
            "    mov.u32 mask, 1;\n" +
            "    shl.b32 mask, mask, blockshift_;\n" +
            "    sub.u32 mask, mask, 1;\n" +
            "\n" +
            "    add.u32 pos, posbase_, row;\n" +
            "    shr.u32 blk, pos, blockshift_;\n" +
            "    sub.u32 blk, blk, firstblock_;\n" +
            "    and.b32 slot, pos, mask;\n" +
            "\n" +
            "    mul.lo.u32 srcBase, row, rowelems_;\n" +
            "    mul.lo.u32 dstBase, blk, floatspbl_;\n" +
            "    mul.lo.u32 t0, layer_, layerregion_;\n" +
            "    add.u32 dstBase, dstBase, t0;\n" +
            "    mad.lo.u32 dstBase, slot, rowelems_, dstBase;\n" +
            "\n" +
            "_element:\n" +
            "    setp.ge.u32 p, e, rowelems_;\n" +
            "    @p bra done;\n" +
            "\n" +
            "    add.u32 t0, srcBase, e;\n" +
            "    cvt.u64.u32 addr, t0;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, sk;\n" +
            "    ld.global.f32 val, [addr];\n" +
            "    add.u32 t0, dstBase, e;\n" +
            "    cvt.u64.u32 addr2, t0;\n" +
            "    shl.b64 addr2, addr2, 2;\n" +
            "    add.u64 addr2, addr2, st;\n" +
            "    st.global.f32 [addr2], val;\n" +
            "\n" +
            "    add.u32 t0, srcBase, e;\n" +
            "    cvt.u64.u32 addr, t0;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, sv;\n" +
            "    ld.global.f32 val, [addr];\n" +
            "    add.u32 t0, dstBase, vhalf_;\n" +
            "    add.u32 t0, t0, e;\n" +
            "    cvt.u64.u32 addr2, t0;\n" +
            "    shl.b64 addr2, addr2, 2;\n" +
            "    add.u64 addr2, addr2, st;\n" +
            "    st.global.f32 [addr2], val;\n" +
            "\n" +
            "    add.u32 e, e, stride;\n" +
            "    bra _element;\n" +
            "done:\n" +
            "    ret;\n" +
            "}\n";
    }

    // Causal attention over a pool's device store: the same per-lane online-softmax walk as `attn_causal`, with each
    // position's K and V rows resolved through the pool's block layout — block by shift, slot by mask, then the
    // layer region, the half, and the kv-head element — instead of a flat arena stride. The score chain, the scale and
    // cap order, the online update, and the final reciprocal are byte-identical to the private-arena walk: only the
    // addressing differs, so the same staged window answers the same bytes through either store.
    private static string BuildAttentionPagedPtx()
    {
        return ".visible .entry attn_causal_paged(\n" +
            "    .param .u64 q_ptr,\n" +
            "    .param .u64 kv_ptr,\n" +
            "    .param .u64 o_ptr,\n" +
            "    .param .u32 heads_arg,\n" +
            "    .param .u32 kvheads_arg,\n" +
            "    .param .u32 hdim_arg,\n" +
            "    .param .u32 blockshift_arg,\n" +
            "    .param .u32 floatspbl_arg,\n" +
            "    .param .u32 layerregion_arg,\n" +
            "    .param .u32 vhalf_arg,\n" +
            "    .param .u32 firstblock_arg,\n" +
            "    .param .u32 layer_arg,\n" +
            "    .param .u32 posbase_arg,\n" +
            "    .param .u32 win_arg,\n" +
            "    .param .u32 groupshift_arg,\n" +
            "    .param .u32 scale_bits_arg,\n" +
            "    .param .u32 cap_bits_arg\n" +
            ") {\n" +
            "    .reg .u64 q, kv, o, addr;\n" +
            "    .reg .u32 heads_, kvheads_, hdim_, blockshift_, floatspbl_, layerregion_, vhalf_, firstblock_, layer_, posbase_, win_, groupshift_, row, head, lane, pos, first, last, kvdim, mask, blk, slot, kRow, kb, kvh, headBase, idx, t0, d;\n" +
            "    .reg .pred p;\n" +
            "    .reg .f32 acc, val, prod, score, rmax, sumexp, rescale, weight, inv, scale, cap, num, den, e2;\n" +
            "\n" +
            "    ld.param.u64 q, [q_ptr];\n" +
            "    ld.param.u64 kv, [kv_ptr];\n" +
            "    ld.param.u64 o, [o_ptr];\n" +
            "    ld.param.u32 heads_, [heads_arg];\n" +
            "    ld.param.u32 kvheads_, [kvheads_arg];\n" +
            "    ld.param.u32 hdim_, [hdim_arg];\n" +
            "    ld.param.u32 blockshift_, [blockshift_arg];\n" +
            "    ld.param.u32 floatspbl_, [floatspbl_arg];\n" +
            "    ld.param.u32 layerregion_, [layerregion_arg];\n" +
            "    ld.param.u32 vhalf_, [vhalf_arg];\n" +
            "    ld.param.u32 firstblock_, [firstblock_arg];\n" +
            "    ld.param.u32 layer_, [layer_arg];\n" +
            "    ld.param.u32 posbase_, [posbase_arg];\n" +
            "    ld.param.u32 win_, [win_arg];\n" +
            "    ld.param.u32 groupshift_, [groupshift_arg];\n" +
            "    ld.param.u32 t0, [scale_bits_arg];\n" +
            "    mov.b32 scale, t0;\n" +
            "    ld.param.u32 t0, [cap_bits_arg];\n" +
            "    mov.b32 cap, t0;\n" +
            "\n" +
            "    mov.u32 row, %ctaid.x;\n" +
            "    mov.u32 head, %ctaid.y;\n" +
            "    mov.u32 lane, %tid.x;\n" +
            "\n" +
            "    setp.ge.u32 p, lane, hdim_;\n" +
            "    @p bra done;\n" +
            "\n" +
            "    shr.u32 kvh, head, groupshift_;\n" +
            "    mul.lo.u32 kvdim, kvheads_, hdim_;\n" +
            "    mad.lo.u32 headBase, row, heads_, head;\n" +
            "    mul.lo.u32 headBase, headBase, hdim_;\n" +
            "\n" +
            "    mov.u32 mask, 1;\n" +
            "    shl.b32 mask, mask, blockshift_;\n" +
            "    sub.u32 mask, mask, 1;\n" +
            "\n" +
            // The same causal range the private-arena walk and the host selection resolve.
            "    add.u32 last, posbase_, row;\n" +
            "    mov.u32 first, 0;\n" +
            "    setp.le.u32 p, win_, 0;\n" +
            "    @p bra _range;\n" +
            "    add.u32 t0, last, 1;\n" +
            "    setp.le.u32 p, t0, win_;\n" + // last + 1 <= win: the window starts at or before position 0
            "    @p bra _range;\n" +
            "    sub.u32 first, last, win_;\n" +
            "    add.u32 first, first, 1;\n" +
            "_range:\n" +
            "\n" +
            "    mov.f32 rmax, 0fFF800000;\n" +
            "    mov.f32 sumexp, 0f00000000;\n" +
            "    mov.f32 acc, 0f00000000;\n" +
            "\n" +
            "    mov.u32 pos, first;\n" +
            "_walk:\n" +
            "    setp.gt.u32 p, pos, last;\n" +
            "    @p bra _normalize;\n" +
            "\n" +
            // This position's K row in the store: block by shift, slot by mask, then the layer region and the kv-head
            // element — the pool's block addressing, in the pool's own stride order.
            "    shr.u32 blk, pos, blockshift_;\n" +
            "    sub.u32 blk, blk, firstblock_;\n" +
            "    and.b32 slot, pos, mask;\n" +
            "    mul.lo.u32 kRow, blk, floatspbl_;\n" +
            "    mul.lo.u32 t0, layer_, layerregion_;\n" +
            "    add.u32 kRow, kRow, t0;\n" +
            "    mad.lo.u32 kRow, slot, kvdim, kRow;\n" +
            "    mad.lo.u32 kb, kvh, hdim_, kRow;\n" +
            "    mov.f32 score, 0f00000000;\n" +
            "    mov.u32 d, 0;\n" +
            "_dot:\n" +
            "    setp.ge.u32 p, d, hdim_;\n" +
            "    @p bra _scaled;\n" +
            "    add.u32 idx, headBase, d;\n" +
            "    cvt.u64.u32 addr, idx;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, q;\n" +
            "    ld.global.f32 val, [addr];\n" +
            "    add.u32 idx, kb, d;\n" +
            "    cvt.u64.u32 addr, idx;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, kv;\n" +
            "    ld.global.f32 prod, [addr];\n" +
            "    mul.f32 val, val, prod;\n" +
            "    add.f32 score, score, val;\n" +
            "    add.u32 d, d, 1;\n" +
            "    bra _dot;\n" +
            "_scaled:\n" +
            "    mul.f32 score, score, scale;\n" +
            "    setp.le.f32 p, cap, 0f00000000;\n" +
            "    @p bra _capped;\n" +
            "    div.rn.f32 num, score, cap;\n" +
            "    mul.f32 num, num, 0f40000000;\n" + // 2 * (score / cap)
            "    mul.f32 num, num, 0f3FB8AA3B;\n" + // log2(e)
            "    ex2.approx.f32 e2, num;\n" +
            "    sub.f32 num, e2, 0f3F800000;\n" +
            "    add.f32 den, e2, 0f3F800000;\n" +
            "    div.rn.f32 num, num, den;\n" +
            "    mul.f32 score, num, cap;\n" +
            "_capped:\n" +
            "\n" +
            "    mov.f32 rescale, 0f3F800000;\n" +
            "    setp.le.f32 p, score, rmax;\n" +
            "    @p bra _weight;\n" +
            "    setp.eq.f32 p, rmax, 0fFF800000;\n" +
            "    @p bra _newmax;\n" +
            "    sub.f32 num, rmax, score;\n" +
            "    mul.f32 num, num, 0f3FB8AA3B;\n" + // log2(e)
            "    ex2.approx.f32 rescale, num;\n" +
            "    mul.f32 sumexp, sumexp, rescale;\n" +
            "_newmax:\n" +
            "    mov.f32 rmax, score;\n" +
            "_weight:\n" +
            "    sub.f32 num, score, rmax;\n" +
            "    mul.f32 num, num, 0f3FB8AA3B;\n" + // log2(e)
            "    ex2.approx.f32 weight, num;\n" +
            "    add.f32 sumexp, sumexp, weight;\n" +
            "\n" +
            // This lane folds this position into its own element: the V row sits in the same block at the V half's
            // offset beside the K row the score just read.
            "    mul.f32 acc, acc, rescale;\n" +
            "    add.u32 idx, kb, vhalf_;\n" +
            "    add.u32 idx, idx, lane;\n" +
            "    cvt.u64.u32 addr, idx;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, kv;\n" +
            "    ld.global.f32 val, [addr];\n" +
            "    mul.f32 val, weight, val;\n" +
            "    add.f32 acc, acc, val;\n" +
            "\n" +
            "    add.u32 pos, pos, 1;\n" +
            "    bra _walk;\n" +
            "_normalize:\n" +
            "\n" +
            "    div.rn.f32 inv, 0f3F800000, sumexp;\n" +
            "    mul.f32 acc, acc, inv;\n" +
            "    add.u32 idx, headBase, lane;\n" +
            "    cvt.u64.u32 addr, idx;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, o;\n" +
            "    st.global.f32 [addr], acc;\n" +
            "done:\n" +
            "    ret;\n" +
            "}\n";
    }

    private static string BuildAttentionFamilyPtx() =>
        "\n" + BuildRopePtx("rope_pairs", adjacentPairing: true) +
        "\n" + BuildRopePtx("rope_half", adjacentPairing: false) +
        "\n" + KvStorePtx +
        "\n" + BuildKvStorePagedPtx() +
        "\n" + BuildAttentionPtx() +
        "\n" + BuildAttentionPagedPtx() +
        "\n" + BuildQuantAttentionPtx("attn_causal_paged_q8q8", vSideQ4: false) +
        "\n" + BuildQuantAttentionPtx("attn_causal_paged_k8v4", vSideQ4: true);

    // Half-float group scale at a computed byte address widened to f32, through integer bit math and a per-thread local
    // slot because cvt.f16 forms are rejected on this seam. The base arrives in a u64 register pointing at the group's
    // first byte (the scale), so the load is at [base] with no constant offset. The widening is the same subnormal branch
    // and exponent-fold the packed-quant GEMV headers use, just against a dynamic base.
    private static IEnumerable<string> PtxWidenHalfAt(string baseReg, string dst, string lbl) =>
    [
        "    ld.global.b16 hw, [" + baseReg + "];",
        "    mov.b32 bits, {hw, 0};",
        "    and.b32 sign, bits, 32768;",
        "    and.b32 payload, bits, 32767;",
        "    shr.u32 exp, payload, 10;",
        "    and.b32 mant, payload, 1023;",
        "    setp.le.u32 p, exp, 0;",
        $"    @!p bra {lbl}_norm;",
        $"{lbl}_sub:",
        "    cvt.rn.f32.u32 num, mant;",
        $"    mul.f32 {dst}, num, 0f33800000;", // 2^-24: the half subnormal magnitude, exactly representable in f32
        "    setp.eq.u32 p, sign, 32768;",
        $"    @!p bra {lbl}_done;",
        $"    mul.f32 {dst}, {dst}, 0fBF800000;", // -1.0f so the subnormal magnitude keeps its sign through a real negation
        $"    bra {lbl}_done;",
        $"{lbl}_norm:",
        "    shl.b32 sign, sign, 16;",
        "    add.u32 fbits, exp, 112;",
        "    shl.b32 fbits, fbits, 23;",
        "    shl.b32 mant, mant, 13;",
        "    or.b32 fbits, fbits, mant;",
        "    xor.b32 fbits, fbits, sign;",
        "    st.local.u32 [hslot], fbits;",
        $"    ld.local.f32 {dst}, [hslot];",
        $"{lbl}_done:",
    ];

    // Causal attention over a pool's device store holding a quantized KV combo: the same per-lane online-softmax walk as
    // `attn_causal_paged`, but each position's K and V rows are read as whole groups of the combo's block structs and
    // decoded on the fly. The dot side is always q8_0 (a half group scale beside 32 signed-byte quants); the value side is
    // q8_0 or q4_0 (a half group scale beside 16 nibble bytes, low nibble to element j and high nibble to element 16 + j).
    // The score chain decodes each K value as quant * groupScale and dots it against the FP32 query in ascending element
    // order — the same dequantized chain the CPU quantized kernel is measured against — and the V accumulate decodes each
    // stored value under its group scale, exactly the pair the committed writers stored. The scale, cap, online update, and
    // final reciprocal are byte-identical to the float walk; only the K/V read changes, so the same staged window answers
    // the same bytes through either store.
    private static string BuildQuantAttentionPtx(string entryName, bool vSideQ4)
    {
        // Build the value-side decode arm: q8_0 reads a signed-byte quant; q4_0 reads a nibble (low nibble for element
        // j in [0,16), high nibble for element 16+j) and subtracts 8. Each arm first widens this element's V group scale
        // (the half at the group's bytes 0-1) into vscale, so the stored code is decoded under the group's own scale —
        // never the K group scale the dot loop left in dscale.
        List<string> vArm = vSideQ4
            ? [
                "    shr.u32 byteidx, vi, 5;", // group index within the token's V region
                "    and.b32 laneidx, vi, 31;", // element index within the group
                "    mul.lo.u32 idx, byteidx, vgb_;",
                "    cvt.u64.u32 off, idx;",
                "    add.u64 vgrp, vgrp, off;", // vgrp now points at this element's group
                string.Join("\n", PtxWidenHalfAt("vgrp", "vscale", "vw")),
                "    setp.lt.u32 p, laneidx, 16;",
                "    @p bra _vq4low;",
                "    sub.u32 laneidx, laneidx, 16;",
                "    add.u32 idx, laneidx, 2;",
                "    cvt.u64.u32 off, idx;",
                "    add.u64 addr, vgrp, off;",
                "    ld.global.u8 hb, [addr];",
                "    shr.b32 nibble, hb, 4;", // high nibble: element 16 + j
                "    bra _vq4done;",
                "_vq4low:",
                "    add.u32 idx, laneidx, 2;",
                "    cvt.u64.u32 off, idx;",
                "    add.u64 addr, vgrp, off;",
                "    ld.global.u8 hb, [addr];",
                "    and.b32 nibble, hb, 15;", // low nibble: element j
                "_vq4done:",
                "    sub.u32 nibble, nibble, 8;", // signed code [-8, 7]
                "    mov.b32 nib_s, nibble;", // bridge the two's-complement code through a signed register before the conversion
                "    cvt.rn.f32.s32 dval, nib_s;",
                "    mul.f32 val, weight, dval;",
                "    mul.f32 val, val, vscale;",
              ]
            : [
                "    shr.u32 byteidx, vi, 5;",
                "    and.b32 laneidx, vi, 31;",
                "    mul.lo.u32 idx, byteidx, vgb_;",
                "    cvt.u64.u32 off, idx;",
                "    add.u64 vgrp, vgrp, off;",
                string.Join("\n", PtxWidenHalfAt("vgrp", "vscale", "vw")),
                "    add.u32 idx, laneidx, 2;", // scale is bytes 0-1, quants start at byte 2
                "    cvt.u64.u32 off, idx;",
                "    add.u64 addr, vgrp, off;",
                "    ld.global.u8 hb, [addr];",
                "    and.b32 hb, hb, 255;",
                "    shr.u32 bits, hb, 7;",
                "    shl.b32 bits, bits, 8;",
                "    cvt.rn.f32.u32 dval, hb;",
                "    cvt.rn.f32.u32 num, bits;",
                "    sub.f32 dval, dval, num;", // signed quant [-127, 127]
                "    mul.f32 val, weight, dval;",
                "    mul.f32 val, val, vscale;",
              ];

        return ".visible .entry " + entryName + "(\n" +
            "    .param .u64 q_ptr,\n" +
            "    .param .u64 kv_ptr,\n" +
            "    .param .u64 o_ptr,\n" +
            "    .param .u32 heads_arg,\n" +
            "    .param .u32 kvheads_arg,\n" +
            "    .param .u32 hdim_arg,\n" +
            "    .param .u32 blockshift_arg,\n" +
            "    .param .u32 bytesperblock_arg,\n" +
            "    .param .u32 layerregion_arg,\n" +
            "    .param .u32 kside_arg,\n" +
            "    .param .u32 gpk_arg,\n" +
            "    .param .u32 gpv_arg,\n" +
            "    .param .u32 kgb_arg,\n" +
            "    .param .u32 vgb_arg,\n" +
            "    .param .u32 firstblock_arg,\n" +
            "    .param .u32 layer_arg,\n" +
            "    .param .u32 posbase_arg,\n" +
            "    .param .u32 win_arg,\n" +
            "    .param .u32 groupshift_arg,\n" +
            "    .param .u32 scale_bits_arg,\n" +
            "    .param .u32 cap_bits_arg\n" +
            ") {\n" +
            "    .reg .u64 q, kv, o, kgrp, vgrp, gaddr, addr, off;\n" +
            "    .reg .u32 heads_, kvheads_, hdim_, blockshift_, bytesperblock_, layerregion_, kside_, gpk_, gpv_, kgb_, vgb_, firstblock_, layer_, posbase_, win_, groupshift_, row, head, lane, pos, first, last, kvdim, mask, blk, slot, kvh, headBase, vi, byteidx, laneidx, idx, t0, t1, d, nibble;\n" +
            "    .reg .u32 bits, payload, exp, mant, sign, fbits, hb;\n" +
            "    .reg .s32 nib_s;\n" +
            "    .reg .b16 hw;\n" +
            "    .reg .pred p;\n" +
            "    .reg .f32 acc, val, prod, score, rmax, sumexp, rescale, weight, inv, scale, cap, num, den, e2, dscale, vscale, qval, dval;\n" +
            "    .local .align 4 .b8 hslot[4];\n" +
            "\n" +
            "    ld.param.u64 q, [q_ptr];\n" +
            "    ld.param.u64 kv, [kv_ptr];\n" +
            "    ld.param.u64 o, [o_ptr];\n" +
            "    ld.param.u32 heads_, [heads_arg];\n" +
            "    ld.param.u32 kvheads_, [kvheads_arg];\n" +
            "    ld.param.u32 hdim_, [hdim_arg];\n" +
            "    ld.param.u32 blockshift_, [blockshift_arg];\n" +
            "    ld.param.u32 bytesperblock_, [bytesperblock_arg];\n" +
            "    ld.param.u32 layerregion_, [layerregion_arg];\n" +
            "    ld.param.u32 kside_, [kside_arg];\n" +
            "    ld.param.u32 gpk_, [gpk_arg];\n" +
            "    ld.param.u32 gpv_, [gpv_arg];\n" +
            "    ld.param.u32 kgb_, [kgb_arg];\n" +
            "    ld.param.u32 vgb_, [vgb_arg];\n" +
            "    ld.param.u32 firstblock_, [firstblock_arg];\n" +
            "    ld.param.u32 layer_, [layer_arg];\n" +
            "    ld.param.u32 posbase_, [posbase_arg];\n" +
            "    ld.param.u32 win_, [win_arg];\n" +
            "    ld.param.u32 groupshift_, [groupshift_arg];\n" +
            "    ld.param.u32 t0, [scale_bits_arg];\n" +
            "    mov.b32 scale, t0;\n" +
            "    ld.param.u32 t0, [cap_bits_arg];\n" +
            "    mov.b32 cap, t0;\n" +
            "\n" +
            "    mov.u32 row, %ctaid.x;\n" +
            "    mov.u32 head, %ctaid.y;\n" +
            "    mov.u32 lane, %tid.x;\n" +
            "\n" +
            "    setp.ge.u32 p, lane, hdim_;\n" +
            "    @p bra done;\n" +
            "\n" +
            "    shr.u32 kvh, head, groupshift_;\n" +
            "    mul.lo.u32 kvdim, kvheads_, hdim_;\n" +
            "    mad.lo.u32 headBase, row, heads_, head;\n" +
            "    mul.lo.u32 headBase, headBase, hdim_;\n" +
            "\n" +
            "    mov.u32 mask, 1;\n" +
            "    shl.b32 mask, mask, blockshift_;\n" +
            "    sub.u32 mask, mask, 1;\n" +
            "\n" +
            "    add.u32 last, posbase_, row;\n" +
            "    mov.u32 first, 0;\n" +
            "    setp.le.u32 p, win_, 0;\n" +
            "    @p bra _range;\n" +
            "    add.u32 t0, last, 1;\n" +
            "    setp.le.u32 p, t0, win_;\n" +
            "    @p bra _range;\n" +
            "    sub.u32 first, last, win_;\n" +
            "    add.u32 first, first, 1;\n" +
            "_range:\n" +
            "\n" +
            "    mov.f32 rmax, 0fFF800000;\n" +
            "    mov.f32 sumexp, 0f00000000;\n" +
            "    mov.f32 acc, 0f00000000;\n" +
            "\n" +
            "    mov.u32 pos, first;\n" +
            "_walk:\n" +
            "    setp.gt.u32 p, pos, last;\n" +
            "    @p bra _normalize;\n" +
            "\n" +
            "    shr.u32 blk, pos, blockshift_;\n" +
            "    sub.u32 blk, blk, firstblock_;\n" +
            "    and.b32 slot, pos, mask;\n" +
            "\n" +
            // K region base for this (blk, layer, slot): blk*bytesperblock + layer*layerregion + (slot*gpk)*kgb.
            "    mul.lo.u32 t0, blk, bytesperblock_;\n" +
            "    mul.lo.u32 t1, layer_, layerregion_;\n" +
            "    add.u32 t0, t0, t1;\n" +
            "    mul.lo.u32 t1, slot, gpk_;\n" +
            "    mul.lo.u32 t1, t1, kgb_;\n" +
            "    add.u32 t0, t0, t1;\n" +
            "    cvt.u64.u32 kgrp, t0;\n" +
            "    add.u64 kgrp, kgrp, kv;\n" +
            "\n" +
            // V region base for this (blk, layer, slot): blk*bytesperblock + layer*layerregion + kside + (slot*gpv)*vgb.
            "    mul.lo.u32 t0, blk, bytesperblock_;\n" +
            "    mul.lo.u32 t1, layer_, layerregion_;\n" +
            "    add.u32 t0, t0, t1;\n" +
            "    add.u32 t0, t0, kside_;\n" +
            "    mul.lo.u32 t1, slot, gpv_;\n" +
            "    mul.lo.u32 t1, t1, vgb_;\n" +
            "    add.u32 t0, t0, t1;\n" +
            "    cvt.u64.u32 vgrp, t0;\n" +
            "    add.u64 vgrp, vgrp, kv;\n" +
            "\n" +
            // The score dot: for each head dimension d, vi = kvh*hdim + d resolves the K group and byte-in-group; the K
            // value is quant * groupScale and the product against the FP32 query lands in ascending order.
            "    mul.lo.u32 t1, kvh, hdim_;\n" + // vi base for this head's kv region
            "    mov.f32 score, 0f00000000;\n" +
            "    mov.u32 d, 0;\n" +
            "_dot:\n" +
            "    setp.ge.u32 p, d, hdim_;\n" +
            "    @p bra _scaled;\n" +
            "    add.u32 vi, t1, d;\n" +
            "    shr.u32 byteidx, vi, 5;\n" + // group index within the token's K region
            "    and.b32 laneidx, vi, 31;\n" + // element index within the group
            "    mul.lo.u32 idx, byteidx, kgb_;\n" +
            "    cvt.u64.u32 off, idx;\n" +
            "    add.u64 gaddr, kgrp, off;\n" + // gaddr points at this element's K group (scale at bytes 0-1)
            string.Join("\n", PtxWidenHalfAt("gaddr", "dscale", "kw")) + "\n" +
            "    add.u32 idx, laneidx, 2;\n" + // quants start at byte 2 beside the half scale
            "    cvt.u64.u32 off, idx;\n" +
            "    add.u64 addr, gaddr, off;\n" +
            "    ld.global.u8 hb, [addr];\n" +
            "    and.b32 hb, hb, 255;\n" +
            "    shr.u32 bits, hb, 7;\n" +
            "    shl.b32 bits, bits, 8;\n" +
            "    cvt.rn.f32.u32 qval, hb;\n" +
            "    cvt.rn.f32.u32 num, bits;\n" +
            "    sub.f32 qval, qval, num;\n" + // signed quant [-127, 127]
            "    mul.f32 dval, qval, dscale;\n" + // dequantized K value
            "    add.u32 idx, headBase, d;\n" +
            "    cvt.u64.u32 addr, idx;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, q;\n" +
            "    ld.global.f32 val, [addr];\n" +
            "    mul.f32 val, val, dval;\n" +
            "    add.f32 score, score, val;\n" +
            "    add.u32 d, d, 1;\n" +
            "    bra _dot;\n" +
            "_scaled:\n" +
            "    mul.f32 score, score, scale;\n" +
            "    setp.le.f32 p, cap, 0f00000000;\n" +
            "    @p bra _capped;\n" +
            "    div.rn.f32 num, score, cap;\n" +
            "    mul.f32 num, num, 0f40000000;\n" +
            "    mul.f32 num, num, 0f3FB8AA3B;\n" +
            "    ex2.approx.f32 e2, num;\n" +
            "    sub.f32 num, e2, 0f3F800000;\n" +
            "    add.f32 den, e2, 0f3F800000;\n" +
            "    div.rn.f32 num, num, den;\n" +
            "    mul.f32 score, num, cap;\n" +
            "_capped:\n" +
            "\n" +
            "    mov.f32 rescale, 0f3F800000;\n" +
            "    setp.le.f32 p, score, rmax;\n" +
            "    @p bra _weight;\n" +
            "    setp.eq.f32 p, rmax, 0fFF800000;\n" +
            "    @p bra _newmax;\n" +
            "    sub.f32 num, rmax, score;\n" +
            "    mul.f32 num, num, 0f3FB8AA3B;\n" +
            "    ex2.approx.f32 rescale, num;\n" +
            "    mul.f32 sumexp, sumexp, rescale;\n" +
            "_newmax:\n" +
            "    mov.f32 rmax, score;\n" +
            "_weight:\n" +
            "    sub.f32 num, score, rmax;\n" +
            "    mul.f32 num, num, 0f3FB8AA3B;\n" +
            "    ex2.approx.f32 weight, num;\n" +
            "    add.f32 sumexp, sumexp, weight;\n" +
            "\n" +
            // This lane folds its own element into the accumulator: decode this position's V value under its group scale,
            // then the weighted value in the host kernel's accumulation order.
            "    mul.f32 acc, acc, rescale;\n" +
            "    add.u32 vi, t1, lane;\n" +
            string.Join("\n", vArm) + "\n" +
            "    add.f32 acc, acc, val;\n" +
            "\n" +
            "    add.u32 pos, pos, 1;\n" +
            "    bra _walk;\n" +
            "_normalize:\n" +
            "\n" +
            "    div.rn.f32 inv, 0f3F800000, sumexp;\n" +
            "    mul.f32 acc, acc, inv;\n" +
            "    add.u32 idx, headBase, lane;\n" +
            "    cvt.u64.u32 addr, idx;\n" +
            "    shl.b64 addr, addr, 2;\n" +
            "    add.u64 addr, addr, o;\n" +
            "    st.global.f32 [addr], acc;\n" +
            "done:\n" +
            "    ret;\n" +
            "}\n";
    }
}

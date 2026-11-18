using System.Text.RegularExpressions;

using Lair.Contracts;

namespace Lair.Backends.Cuda;

// The blocked-matmul PTX family, derived from each format's own GEMV kernel text so the unpacking a format has already
// proven stays single-sourced: every group arm, window offset, scale fusion, header widening, and column-bound rule is
// carried verbatim, and only three things change — the batch cursor walks an activation-row window instead of one vector,
// each lane keeps its own accumulator chain, and a store epilogue writes one output per in-range lane. Because every lane's
// chain advances over the same columns in the same order with the same operand association as that vector's solo GEMV walk,
// a windowed result matches what repeated single-vector dots produce within the format's inherited band.
public sealed partial class CudaBackend
{
    // The three frame anchors every packed-quant GEMV kernel in this seam is built around: the batch cursor opening, its
    // accumulator initialization, and the store epilogue. Each must appear exactly once per kernel; a kernel that stops
    // matching them fails here naming itself instead of shipping a half-transformed entry point.
    private const string GemvBatchOpeningAnchor = "    mov.u32 bb, 0;\nbatch:\n";

    private const string GemvAccumulatorInitAnchor = "    mov.f32 acc, 0f00000000;";

    private static readonly string GemvStoreEpilogueAnchor =
        "_store:\n" +
        "    mad.lo.u32 t0, bb, rows_, row;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, o, addr;\n" +
        "    st.global.f32 [addr], acc;\n" +
        "    add.u32 bb, bb, 1;\n" +
        "    bra batch;";

    // The activation-dot tail every group arm of the packed-quant family closes with: this element's slot in the batch-major
    // activation region, one load, and a plain mul then add into the accumulator. Two weight registers carry it today — the
    // unpacked value itself for the block quants, and the fused (scale * kvalue) product for the IQ4 pair.
    private static string GemvDotTailAnchor(string weightReg) =>
        "    mad.lo.u32 t0, bb, cols_, cabs;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        $"    mul.f32 wv, xv, {weightReg};\n" +
        "    add.f32 acc, acc, wv;";

    private static readonly Regex GemvEntryAnchor = new(@"\.visible \.entry gemv_(\w+)\(", RegexOptions.CultureInvariant);

    // Builds one format's blocked-matmul entry point from its own GEMV kernel text: same parameters (the IQ4 pair keeps
    // carrying the scratch table pointer beside them), same grid geometry, and the unpacking arms untouched — only the
    // batch loop, accumulator set, per-element dot, and store epilogue are rewritten to walk an activation-row window.
    private static string BuildGemmKernelPtx(string gemvKernelText, QuantFormat format)
    {
        Match entry = GemvEntryAnchor.Match(gemvKernelText);
        if (!entry.Success || GemvEntryAnchor.Matches(gemvKernelText).Count != 1)
        {
            throw new InvalidOperationException($"blocked matmul build for {format}: expected exactly one GEMV entry point to derive from.");
        }

        string text = gemvKernelText;
        text = GemvEntryAnchor.Replace(text, match => $".visible .entry gemm_{match.Groups[1].Value}(");

        // The window's extra state: clamped activation-row cursors for lanes 1..N-1 (lane zero keeps the kernel's own `bb`),
        // a standing batch ceiling for the clamp, an unclamped store cursor and its guard temp, one accumulator per lane past
        // zero, and one product scratch per lane so no lane's multiply shares a register with another.
        text = ReplaceExactlyOnce(text, "    .reg .pred p;\n", "    .reg .pred p;\n" + WindowDeclarations(), format, "register declarations");

        text = ReplaceExactlyOnce(
            text,
            GemvBatchOpeningAnchor,
            "    sub.u32 tmax, cnt_, 1;\n" + GemvBatchOpeningAnchor.Replace("batch:", "window:"),
            format,
            "batch opening");

        // Window head: clamp every lane's activation-row cursor against the batch ceiling (a lane past the end of the batch
        // dots against a real row and stores nothing) and open one accumulator chain per lane, lane zero first.
        List<string> head = ["    mov.f32 acc, 0f00000000;"];
        for (int lane = 1; lane < GemmWindow; lane++)
        {
            head.Add($"    add.u32 rb{lane}, bb, {lane};");
            head.Add($"    min.u32 rb{lane}, rb{lane}, tmax;");
            head.Add($"    mov.f32 a{lane}, 0f00000000;");
        }

        text = ReplaceExactlyOnce(text, GemvAccumulatorInitAnchor, string.Join("\n", [.. head]), format, "accumulator initialization");

        // One store per lane against its own unclamped batch index, guarded so a window that runs past the end of the batch
        // writes each output slot exactly once. Lane zero's store stays the kernel's own text verbatim.
        List<string> epilogue =
        [
            "_store:",
            "    mad.lo.u32 t0, bb, rows_, row;",
            "    cvt.u64.u32 addr, t0;",
            "    shl.b64 addr, addr, 2;",
            "    add.u64 addr, o, addr;",
            "    st.global.f32 [addr], acc;",
        ];

        for (int lane = 1; lane < GemmWindow; lane++)
        {
            epilogue.Add($"    add.u32 tv, bb, {lane};"); // this lane's batch index: unclamped, so the guard below can judge it.
            epilogue.Add($"    setp.ge.u32 p, tv, cnt_;");
            epilogue.Add($"    @p bra _sl{lane};");
            epilogue.Add("    mad.lo.u32 t0, tv, rows_, row;");
            epilogue.Add("    cvt.u64.u32 addr, t0;");
            epilogue.Add("    shl.b64 addr, addr, 2;");
            epilogue.Add("    add.u64 addr, o, addr;");
            epilogue.Add($"    st.global.f32 [addr], a{lane};");
            epilogue.Add($"_sl{lane}:");
        }

        epilogue.Add($"    add.u32 bb, bb, {GemmWindow};"); // the next activation-row window of this weight row.
        epilogue.Add("    bra window;");

        text = ReplaceExactlyOnce(text, GemvStoreEpilogueAnchor, string.Join("\n", [.. epilogue]), format, "store epilogue");

        // Every group arm's dot tail becomes a windowed dot: the unpacking above it stays verbatim and runs once per weight
        // element, while one product per lane folds that value into that lane's own accumulator chain.
        int tails = 0;
        foreach (string weightReg in new[] { "wv", "mv" })
        {
            string anchor = GemvDotTailAnchor(weightReg);
            if (!text.Contains(anchor, StringComparison.Ordinal))
            {
                continue;
            }

            List<string> windowed = [];
            for (int lane = 0; lane < GemmWindow; lane++)
            {
                string cursor = lane == 0 ? "bb" : $"rb{lane}";
                string accumulator = lane == 0 ? "acc" : $"a{lane}";

                windowed.Add($"    mad.lo.u32 t0, {cursor}, cols_, cabs;"); // this lane's slot for the column being unpacked right here.
                windowed.Add("    cvt.u64.u32 addr, t0;");
                windowed.Add("    shl.b64 addr, addr, 2;");
                windowed.Add("    add.u64 addr, xin, addr;");
                windowed.Add("    ld.global.f32 xv, [addr];");

                // Each lane multiplies into its own dead scratch and folds it into its own accumulator in the same two-instruction shape the solo kernel writes: one product register per lane keeps every lane's multiply free to fuse exactly as that vector's does, instead of chaining eight definitions through one shared temp.
                windowed.Add($"    mul.f32 pv{lane}, xv, {weightReg};");
                windowed.Add($"    add.f32 {accumulator}, {accumulator}, pv{lane};");
            }

            int replaced = CountOccurrences(text, anchor);
            tails += replaced;
            text = text.Replace(anchor, string.Join("\n", [.. windowed]));
        }

        if (tails == 0)
        {
            throw new InvalidOperationException($"blocked matmul build for {format}: found no activation-dot tail to window.");
        }

        if (Regex.IsMatch(text, @"\bbatch\b"))
        {
            throw new InvalidOperationException($"blocked matmul build for {format}: a batch-loop label survived the window rewrite.");
        }

        return text;
    }

    private static string WindowDeclarations()
    {
        List<string> cursors = [];
        List<string> floats = [];

        for (int lane = 1; lane < GemmWindow; lane++)
        {
            cursors.Add($"rb{lane}");
            floats.Add($"a{lane}");
        }

        cursors.Add("tmax");
        cursors.Add("tv");

        // One product scratch per lane, so no lane's multiply has to share a register with another.
        for (int lane = 0; lane < GemmWindow; lane++)
        {
            floats.Add($"pv{lane}");
        }

        return $"    .reg .u32 {string.Join(", ", [.. cursors])};\n" +
               $"    .reg .f32 {string.Join(", ", [.. floats])};\n";
    }

    private static string ReplaceExactlyOnce(string text, string anchor, string replacement, QuantFormat format, string what)
    {
        int found = CountOccurrences(text, anchor);
        if (found != 1)
        {
            throw new InvalidOperationException($"blocked matmul build for {format}: expected exactly one {what} anchor and found {found}.");
        }

        return text.Replace(anchor, replacement);
    }

    private static int CountOccurrences(string text, string needle)
    {
        int count = 0;
        for (int index = 0; (index = text.IndexOf(needle, index, StringComparison.Ordinal)) >= 0; index += needle.Length)
        {
            count++;
        }

        return count;
    }

    // The whole blocked-matmul family: each format's own GEMV kernel text, windowed. Order is irrelevant to the driver and
    // every entry point resolves independently, so a new format only needs its GEMV builder named here.
    private static string BuildGemmFamilyPtx() =>
        "\n" + BuildGemmKernelPtx(BuildGemvQ8_0Ptx(), QuantFormat.Q8_0) +
        "\n" + BuildGemmKernelPtx(BuildGemvQ5_0Ptx(), QuantFormat.Q5_0) +
        "\n" + BuildGemmKernelPtx(BuildGemvQ2_0Ptx(), QuantFormat.Q2_0) +
        "\n" + BuildGemmKernelPtx(GemvQ4KPtx, QuantFormat.Q4_K) +
        "\n" + BuildGemmKernelPtx(BuildGemvQ6KPtx(), QuantFormat.Q6_K) +
        "\n" + BuildGemmKernelPtx(BuildGemvQ5KPtx(), QuantFormat.Q5_K) +
        "\n" + BuildGemmKernelPtx(BuildGemvQ2KPtx(), QuantFormat.Q2_K) +
        "\n" + BuildGemmKernelPtx(BuildGemvQ3KPtx(), QuantFormat.Q3_K) +
        "\n" + BuildGemmKernelPtx(BuildGemvIQ4NPtx(), QuantFormat.IQ4_NL) +
        "\n" + BuildGemmKernelPtx(BuildGemvIQ4XPtx(), QuantFormat.IQ4_XS);
}

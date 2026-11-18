using System.Text.RegularExpressions;

using Lair.Contracts;

namespace Lair.Backends.Cuda;

// The unquantized dense-matmul entries of the blocked family, derived from each format's own GEMV kernel text exactly like
// the packed-quant pair is: same parameters, same grid geometry, and the format's weight-widening arms carried verbatim — only
// the batch cursor walks an activation-row window instead of one vector, per-lane accumulator chains replace the single chain,
// and each in-range lane stores its own output. Because a lane's chain advances over the same columns in the same order with
// the same operand association as that vector's solo GEMV walk — one plain mul then add per element against the very weight
// value that widening produced — an unquantized windowed result is bit-for-bit what repeated single-vector dots produce, so no
// precision band moves for F32/F16/BF16 and no format widens through a lossy conversion. The reuse this family exists to get is
// the weight side: each row's element is loaded (and for F16 widened through integer bit math) once per activation-row window,
// not once per vector, which is what a dense-matrix selection buys over N single-vector launches on these formats today. No
// tensor-core tile stands behind any of these entries in this build: `mma.sync` needs an sm_70-or-higher target declaration for
// the whole module every kernel here shares, and its hardware accumulation order is not the ascending mul-then-add chain these
// three formats are pinned to today. That posture is recorded rather than silently claimed; a tensor-core dense tile lands only
// against measured drift it can be held inside.
public sealed partial class CudaBackend
{
    // The frame anchors every unquantized GEMV kernel in this seam is built around, in the shape those kernels actually write:
    // their column loop branches to `store`, and their activation dot tail multiplies weight first against that format's own
    // widened value. Each must appear exactly once per entry point; a kernel that stops matching them fails here naming itself
    // instead of shipping a half-transformed dense-matrix walk.
    private static readonly string DenseStoreEpilogueAnchor =
        "store:\n" +
        "    mad.lo.u32 t0, bb, rows_, row;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, o, addr;\n" +
        "    st.global.f32 [addr], acc;\n" +
        "    add.u32 bb, bb, 1;\n" +
        "    bra batch;";

    private static readonly string DenseActivationDotAnchor =
        "    mad.lo.u32 t0, bb, cols_, col;\n" +
        "    cvt.u64.u32 addr, t0;\n" +
        "    shl.b64 addr, addr, 2;\n" +
        "    add.u64 addr, xin, addr;\n" +
        "    ld.global.f32 xv, [addr];\n" +
        "    mul.f32 wv, wv, xv;\n" +
        "    add.f32 acc, acc, wv;";

    // Builds one unquantized format's blocked-matmul entry point from its own GEMV kernel text. The window scaffolding is the
    // same shape the packed-quant family installs — clamped activation-row cursors for lanes 1..N-1 against a standing batch
    // ceiling, one accumulator and one product scratch per lane, an unclamped store cursor with its guard — while everything
    // format-specific (widening arms, column-bound rule, operand order) stays verbatim from that GEMV.
    private static string BuildDenseGemmKernelPtx(string gemvKernelText, QuantFormat format)
    {
        Match entry = GemvEntryAnchor.Match(gemvKernelText);
        if (!entry.Success || GemvEntryAnchor.Matches(gemvKernelText).Count != 1)
        {
            throw new InvalidOperationException($"dense matmul build for {format}: expected exactly one GEMV entry point to derive from.");
        }

        string text = gemvKernelText;
        text = GemvEntryAnchor.Replace(text, match => $".visible .entry gemm_{match.Groups[1].Value}(");

        text = ReplaceExactlyOnce(text, "    .reg .pred p;\n", "    .reg .pred p;\n" + WindowDeclarations(), format, "register declarations");

        text = ReplaceExactlyOnce(
            text,
            GemvBatchOpeningAnchor,
            "    sub.u32 tmax, cnt_, 1;\n" + GemvBatchOpeningAnchor.Replace("batch:", "window:"),
            format,
            "batch opening");

        // Window head: clamp every lane's activation-row cursor against the batch ceiling (a lane past the end of the batch dots
        // against a real row and stores nothing) and open one accumulator chain per lane. The column cursor stays its GEMV's own
        // line, which follows this anchor untouched.
        List<string> head = ["    mov.f32 acc, 0f00000000;"];
        for (int lane = 1; lane < GemmWindow; lane++)
        {
            head.Add($"    add.u32 rb{lane}, bb, {lane};");
            head.Add($"    min.u32 rb{lane}, rb{lane}, tmax;");
            head.Add($"    mov.f32 a{lane}, 0f00000000;");
        }

        text = ReplaceExactlyOnce(text, GemvAccumulatorInitAnchor, string.Join("\n", [.. head]), format, "accumulator initialization");

        // The column loop's exit branch has to name the renamed store epilogue.
        text = ReplaceExactlyOnce(text, "    @p bra store;\n", "    @p bra _store;\n", format, "column-loop store branch");

        // One store per lane against its own unclamped batch index, guarded so a window running past the end of the batch writes
        // each output slot exactly once. Lane zero's store stays the kernel's own text verbatim.
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

        text = ReplaceExactlyOnce(text, DenseStoreEpilogueAnchor, string.Join("\n", [.. epilogue]), format, "store epilogue");

        // The GEMV's activation dot becomes a windowed dot: the widening above it stays verbatim and runs once per weight
        // element, while one product per lane folds that same value into that lane's own accumulator chain with the format's own
        // operand order. Each lane reads its own activation row at its clamped cursor — no out-of-range activation ever reaches
        // memory because every cursor is inside the batch ceiling.
        if (!text.Contains(DenseActivationDotAnchor, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"dense matmul build for {format}: found no activation-dot tail to window.");
        }

        List<string> windowed = [];
        for (int lane = 0; lane < GemmWindow; lane++)
        {
            string cursor = lane == 0 ? "bb" : $"rb{lane}";
            string accumulator = lane == 0 ? "acc" : $"a{lane}";

            windowed.Add($"    mad.lo.u32 t0, {cursor}, cols_, col;"); // this lane's slot for the column being unpacked right here.
            windowed.Add("    cvt.u64.u32 addr, t0;");
            windowed.Add("    shl.b64 addr, addr, 2;");
            windowed.Add("    add.u64 addr, xin, addr;");
            windowed.Add("    ld.global.f32 xv, [addr];");

            // One product register per lane keeps every lane's multiply free to fold on its own chain instead of chaining eight definitions through one shared temp.
            windowed.Add($"    mul.f32 pv{lane}, wv, xv;");
            windowed.Add($"    add.f32 {accumulator}, {accumulator}, pv{lane};");
        }

        text = ReplaceExactlyOnce(text, DenseActivationDotAnchor, string.Join("\n", [.. windowed]), format, "activation-dot tail");

        if (Regex.IsMatch(text, @"\bbatch\b"))
        {
            throw new InvalidOperationException($"dense matmul build for {format}: a batch-loop label survived the window rewrite.");
        }

        return text;
    }

    // The unquantized trio: each format's own GEMV kernel text, windowed. F32 dots plain f32 weights, F16 keeps its integer-bit
    // widening of every half (subnormal branch included), and BF16 keeps its high-half shift — none of them converts a weight or
    // activation to another precision to make the dot run.
    private static string BuildDenseGemmFamilyPtx() =>
        "\n" + BuildDenseGemmKernelPtx(GemvF32Ptx, QuantFormat.F32) +
        "\n" + BuildDenseGemmKernelPtx(GemvF16Ptx, QuantFormat.F16) +
        "\n" + BuildDenseGemmKernelPtx(GemvBf16Ptx, QuantFormat.BF16);
}

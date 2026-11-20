using System.Buffers;

using Lair.Contracts;

namespace Lair.Backends.Cuda;

public sealed partial class CudaBackend
{
    // The sub-4-bit i-quant family dots against a Q8_K quantization of the activation instead of its FP values, exactly as
    // the CPU oracle's RowDotQ8K path does: one float block scale per 256-value super-block followed by 256 signed quants.
    // The host-side quantizer mirrors QuantizeActivationToQ8K step for step - iscale = -127/max negates so every stored
    // quant lands in [-127, 127], nearest_int rounds ties to even, and whole super-blocks are covered with zero quants past
    // the real width under a scale built from real values only. The device kernel then consumes whole sub-blocks without
    // per-column bounds: padding products against zero quants contribute exactly nothing on either side of the seam.
    private const int Q8KMaxQuant = 127;

    // Device scratch holding every pinned table the staging arms look up, allocated once per instance and freed with it.
    // Each grid base is a separate sub-pointer so each kernel's launch passes its own table's device address.
    private ulong _iqScratchDevice;
    private ulong _tabKSigns;
    private ulong _tabIq4Values;
    private ulong _tabIq2XXSGrid;
    private ulong _tabIq2XSGrid;
    private ulong _tabIq2SGrid;
    private ulong _tabIq3XGrid;
    private ulong _tabIq3SGrid;

    private static int Q8KBlockCount(int cols) => (cols + 255) / 256;

    // Mirrors the oracle's quantizer exactly: first strict improvement carries the block maximum's sign, iscale negates it
    // so the maximum quantizes to -127 and no quant reaches -128, and padding past the real width stores zero quants.
    private static void QuantizeActivationToQ8K(ReadOnlySpan<float> x, int cols, Span<byte> qa, Span<float> da)
    {
        int blocks = Q8KBlockCount(cols);

        for (int i = 0; i < blocks; i++)
        {
            int start = i * 256;
            int limit = Math.Min(256, cols - start); // never negative: blocks covers exactly the real columns' super-blocks.
            Span<byte> q = qa.Slice(start, 256);

            float amax = 0f;
            float max = 0f;
            for (int j = 0; j < limit; j++)
            {
                float v = x[start + j];
                float ax = Math.Abs(v);
                if (ax > amax)
                {
                    amax = ax;
                    max = v; // the reference keeps the first strict improvement, sign included.
                }
            }

            if (amax == 0f)
            {
                da[i] = 0f; // all-zero block: scale zero, quants zero, and the dot contributes exactly nothing.
                q.Fill(0);
                continue;
            }

            float iscale = -127f / max;
            for (int j = 0; j < limit; j++)
            {
                qa[start + j] = (byte)(sbyte)Math.Min(Q8KMaxQuant, NearestInt(iscale * x[start + j]));
            }

            q.Slice(limit).Fill(0); // padding past the real width: zero quants under the real block scale.
            da[i] = 1f / iscale;
        }
    }

    private static void QuantizeActivationsToQ8K(ReadOnlySpan<float> x, int count, int cols, Span<byte> qa, Span<float> da)
    {
        int blocks = Q8KBlockCount(cols);

        for (int b = 0; b < count; b++)
        {
            QuantizeActivationToQ8K(x.Slice(checked((int)b * cols), cols), cols, qa.Slice(checked((int)(b * blocks * 256)), blocks * 256), da.Slice(b * blocks, blocks));
        }
    }

    private static int NearestInt(float v) => (int)Math.Round(v, MidpointRounding.ToEven);

    // Scalar read of one mirrored IQ4 table entry: nibble n indexes byte n widened with sign, exactly as the oracle's shared value table reads.
    private static float KValueIQ4(int nibble) => (sbyte)Iq4ValueBytes()[nibble];

    // The Q8_K activation crossing every sub-4-bit GEMV runs through: quantize host-side once per call, copy quants and scales to device, launch
    // the format's kernel against them, read outputs back. Allocation, accounting, timing, and cleanup follow RunGemv's discipline exactly - one
    // pinned region per tensor for weights (either residency posture), fresh activation/output buffers per call, every crossing accounted. The
    // scratch tables ride as two extra launch parameters: this format's grid base and the shared ksigns base (raw-sign formats pass it unused).
    private unsafe void RunGemvQ8K(WeightRef w, ReadOnlySpan<float> activations, int count, Span<float> output, IntPtr function)
    {
        (int rows, int cols) = ValidateShape(w);

        int blocks = Q8KBlockCount(cols);
        int qaBytes = checked(count * blocks * 256); // whole super-blocks per vector: padding quants carry zero.
        int daFloats = count * blocks;

        byte[] qaPool = ArrayPool<byte>.Shared.Rent(qaBytes);
        float[] daPool = ArrayPool<float>.Shared.Rent(daFloats); // rented buffers stay pinned for exactly the quantization pass below.

        try
        {
            QuantizeActivationsToQ8K(activations, count, cols, new Span<byte>(qaPool, 0, qaBytes), new Span<float>(daPool, 0, daFloats));

            nuint weightBytes = checked((nuint)(rows * w.Shape.RowStrideBytes(w.Format)));
            nuint outBytes = checked((nuint)count * (nuint)rows * sizeof(float));

            lock (_gate)
            {
                SetCurrentContext(); // the driver API keeps the current context per thread, so every call re-binds this instance's retained primary one

                ulong deviceWeights;
                fixed (byte* wBytes = w.Bytes)
                {
                    // The mapped posture launches against the device pointer of a registered host region; the default posture keeps its once-copied device buffer. Either answer is pinned for exactly this call and cached per tensor for the instance's lifetime.
                    deviceWeights = _weightResidency == CudaWeightResidency.HostMapped
                        ? EnsureHostMappedWeights((nint)wBytes, weightBytes)
                        : EnsureDeviceWeights((nint)wBytes, weightBytes); // pinned for exactly this call; the copy runs now and never again per tensor
                }

                EnsureIqScratch(); // the pinned tables exist once per instance: every launch reads them through its own sub-pointer pair.

                ulong gridBase = GridDeviceBase(w.Format); // this format's table region ...
                ulong ksignsBase = _tabKSigns; // ... and the shared sign rows, which raw-sign formats pass without reading.

                ulong deviceQa = 0;
                ulong deviceDa = 0;
                ulong deviceOut = 0;
                try
                {
                    // Stream-ordered transients on this lane, exactly as the FP-activation family now allocates them.
                    CudaException.ThrowIf("cuMemAllocAsync", CudaNative.cuMemAllocAsync(out deviceQa, (nuint)qaBytes, _stream));
                    CudaException.ThrowIf("cuMemAllocAsync", CudaNative.cuMemAllocAsync(out deviceDa, (nuint)(daFloats * sizeof(float)), _stream));
                    CudaException.ThrowIf("cuMemAllocAsync", CudaNative.cuMemAllocAsync(out deviceOut, outBytes, _stream));

                    // Same pinned staging and instance-lane queueing as the FP-activation family: quants stage at one end of
                    // the input arena and their scales beside them (qa sizes are whole 256-byte super-blocks, so the second
                    // region is already aligned), each upload enqueues on this backend's own stream without blocking.
                    EnsurePinnedIn((nuint)qaBytes + (nuint)(daFloats * sizeof(float)));

                    fixed (byte* pQa = qaPool)
                    {
                        Buffer.MemoryCopy(pQa, (byte*)_pinnedIn, (long)_pinnedInBytes, qaBytes);
                        CudaException.ThrowIf("cuMemcpyHtoDAsync_v2", CudaNative.cuMemcpyHtoDAsync(deviceQa, (byte*)_pinnedIn, (nuint)qaBytes, _stream));

                        AccountRuntimeCopy(qaBytes); // this call's quantized activation crossing the seam ...
                    }

                    fixed (float* pDa = daPool)
                    {
                        Buffer.MemoryCopy(pDa, (byte*)_pinnedIn + qaBytes, (long)(_pinnedInBytes - (nuint)qaBytes), daFloats * sizeof(float));

                        CudaException.ThrowIf("cuMemcpyHtoDAsync_v2", CudaNative.cuMemcpyHtoDAsync(deviceDa, (byte*)_pinnedIn + qaBytes, (nuint)(daFloats * sizeof(float)), _stream));

                        AccountRuntimeCopy(daFloats * sizeof(float)); // ... and its per-super-block scales beside it.
                    }

                    uint rowsU = (uint)rows;
                    uint colsU = (uint)cols;
                    uint countU = (uint)count;
                    void** kernelParams = stackalloc void*[9]; // the uniform Q8_K launch shape: weight, quants, scales, output, geometry, then two table bases.
                    kernelParams[0] = &deviceWeights;
                    kernelParams[1] = &deviceQa;
                    kernelParams[2] = &deviceDa;
                    kernelParams[3] = &deviceOut;
                    kernelParams[4] = &rowsU;
                    kernelParams[5] = &colsU;
                    kernelParams[6] = &countU;
                    kernelParams[7] = &gridBase;
                    kernelParams[8] = &ksignsBase;

                    uint blocksG = (uint)((rows + KernelThreads - 1) / KernelThreads); // same row-per-thread grid geometry as every other GEMV in the family.
                    bool timed = BeginKernelTiming(); // device-measured busy time for exactly this launch, when the seam is open here
                    CudaException.ThrowIf("cuLaunchKernel", CudaNative.cuLaunchKernel(function, blocksG, 1, 1, KernelThreads, 1, 1, 0, _stream, kernelParams, null));
                    if (timed)
                    {
                        StampKernelEnd();
                    }

                    EnsurePinnedOut(outBytes);
                    CudaException.ThrowIf("cuMemcpyDtoHAsync_v2", CudaNative.cuMemcpyDtoHAsync((byte*)_pinnedOut, deviceOut, outBytes, _stream));

                    CudaException.ThrowIf("cuMemFreeAsync", CudaNative.cuMemFreeAsync(deviceOut, _stream));
                    CudaException.ThrowIf("cuMemFreeAsync", CudaNative.cuMemFreeAsync(deviceDa, _stream));
                    CudaException.ThrowIf("cuMemFreeAsync", CudaNative.cuMemFreeAsync(deviceQa, _stream));
                    deviceOut = 0;
                    deviceDa = 0;
                    deviceQa = 0;

                    SyncInstanceStream(); // joins this instance's own stream only: another model's kernels keep running while this lane drains
                    if (timed)
                    {
                        FoldKernelElapsed(); // both stamps have completed by the time this instance's stream synchronize returns
                    }

                    fixed (float* pOut = output)
                    {
                        Buffer.MemoryCopy((byte*)_pinnedOut, pOut, outBytes, outBytes);
                        AccountRuntimeCopy((long)outBytes); // this call's kernel output crossing back to host memory
                    }
                }
                catch
                {
                    CudaNative.cuStreamSynchronize(_stream); // drain the lane before finally releases its buffers: a failed enqueue may leave earlier staging pending

                    throw;
                }
                finally
                {
                    if (deviceOut != 0)
                    {
                        CudaNative.cuMemFree(deviceOut);
                    }

                    if (deviceDa != 0)
                    {
                        CudaNative.cuMemFree(deviceDa);
                    }

                    if (deviceQa != 0)
                    {
                        CudaNative.cuMemFree(deviceQa);
                    }
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(qaPool);
            ArrayPool<float>.Shared.Return(daPool);
        }
    }

    // The scratch sub-pointer each Q8_K kernel's grid_ptr parameter reads: the table its staging arms index, per format.
    private ulong GridDeviceBase(QuantFormat format) => format switch
    {
        QuantFormat.IQ2_XXS => _tabIq2XXSGrid,
        QuantFormat.IQ2_XS => _tabIq2XSGrid,
        QuantFormat.IQ2_S => _tabIq2SGrid,
        QuantFormat.IQ3_XXS => _tabIq3XGrid,
        QuantFormat.IQ3_S => _tabIq3SGrid,
        _ => throw new ArgumentOutOfRangeException(nameof(format), format, "Only the sub-4-bit i-quant family dots against Q8_K activations."),
    };

    // Ensures the fused device scratch of pinned tables exists and answers each table's sub-pointer. cuMemAlloc hands back an
    // aligned base; every region length here is a multiple of four bytes, so grid regions land dword-aligned inside it and
    // whole-entry loads stay legal without per-table allocations (ksigns and the IQ4 value table ride in scratch too). Every
    // table fill rides the instance stream and the whole scratch joins before answering, exactly as a resident weight copy
    // does: the launches that index these tables are enqueued on this lane behind the fills, and the join covers the source
    // tables' own lifetime because each is pinned only for the copy that stages it.
    private void EnsureIqScratch()
    {
        if (_iqScratchDevice != 0)
        {
            return;
        }

        int total = GetKSignsIQ2XS().Length + Iq4ValueBytes().Length
            + GetIq2XXSGrid().Length + GetIq2XSGrid().Length + GetIq2SGrid().Length
            + GetIq3XXSGrid().Length + GetIq3SGrid().Length;

        CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out ulong scratch, (nuint)total));
        _iqScratchDevice = scratch;

        int offset = 0;
        UploadTable(scratch, ref offset, GetKSignsIQ2XS(), out _tabKSigns, _stream); // 128 bytes: dword-aligned at the aligned base.
        UploadTable(scratch, ref offset, Iq4ValueBytes(), out _tabIq4Values, _stream); // 16 bytes on top of it: still a multiple of four.
        UploadTable(scratch, ref offset, GetIq2XXSGrid(), out _tabIq2XXSGrid, _stream);
        UploadTable(scratch, ref offset, GetIq2XSGrid(), out _tabIq2XSGrid, _stream);
        UploadTable(scratch, ref offset, GetIq2SGrid(), out _tabIq2SGrid, _stream);
        UploadTable(scratch, ref offset, GetIq3XXSGrid(), out _tabIq3XGrid, _stream);
        UploadTable(scratch, ref offset, GetIq3SGrid(), out _tabIq3SGrid, _stream);

        SyncInstanceStream(); // every table is device-complete before the first launch that indexes one of them

        static void UploadTable(ulong scratch, ref int offset, ReadOnlySpan<byte> table, out ulong deviceBase, IntPtr stream)
        {
            deviceBase = scratch + (uint)offset; // byte offsets stay dword-aligned by construction: every region length is a multiple of four.
            unsafe
            {
                fixed (byte* p = table)
                {
                    CudaException.ThrowIf("cuMemcpyHtoDAsync_v2", CudaNative.cuMemcpyHtoDAsync(deviceBase, p, (nuint)table.Length, stream));
                }
            }

            offset += table.Length;
        }
    }

    private void FreeIqScratch()
    {
        if (_iqScratchDevice != 0)
        {
            CudaNative.cuMemFree(_iqScratchDevice);
            _iqScratchDevice = 0;
        }
    }

    // The Q8_K GEMV frame: same row-per-thread geometry and batch loop as the FP-activation family, but every block walks a
    // whole 256-value super-block against its own quantized activation window. Accumulation discipline mirrors RowDotQ8K's
    // scalar reference exactly - exact int32 sub-block totals under their ls factors into an integer bsum, then d * bsum as
    // one float step per super-block in block order (d = the widened half header times this vector's da scale), and the
    // format's final factor once at store. Integer state stays two's-complement throughout: magnitudes are unsigned bytes,
    // quant bytes decode to signed values by low7 minus their 128-bit weight, sign folding lands on one operand of an exact
    // mad.lo chain - so every partial is the same int32 total the scalar loop runs and lane grouping cannot drift anywhere.
    private static string BuildQ8KGemvPtx(string entryName, int blockBytes, IEnumerable<string> subBlockArms, string? finalFactor)
    {
        ArgumentNullException.ThrowIfNull(subBlockArms);

        List<string> lines =
        [
            $".visible .entry {entryName}(",
            "    .param .u64 w_ptr,",
            "    .param .u64 qa_ptr,",
            "    .param .u64 da_ptr,",
            "    .param .u64 out_ptr,",
            "    .param .u32 rows_arg,",
            "    .param .u32 cols_arg,",
            "    .param .u32 count_arg,",
            "    .param .u64 grid_ptr,", // this format's pinned grid table inside the fused scratch region.
            "    .param .u64 tab_ptr", // ksigns for the packed-sign formats; raw-sign formats pass it unused (uniform host params).
            ") {",
            // `qblk` is this batch vector's Q8_K quant window and `dacur` walks its per-super-block float scales.
            // Per-arm state: aW/mW hold one activation word and its staged magnitude word, gi indexes an entry load or holds a
            // packed aux/scale field during setup, sg carries the sub-block's four sign bytes once they are all folded in, lsA/lsB
            // carry scale factors (and double as aux fields during setup), loU/hiU/bsum accumulate exact int32 totals.
            "    .reg .u64 w, qa, dasc, o, gridBase, tabBase, blockAddr, addr, qblk, dacur;",
            "    .reg .u32 row, bb, rows_, cols_, cnt_, tid, t0, strideBlocks, strideBytes, rowBase, base, qb_, gi, b1, aW, mW, mvv, sbt, zr, t1, t2, sg, lsA, lsB, loU, hiU, bsum;",
            "    .reg .s32 sv;", // the one bridge register: mov.b32 reinterprets the two's-complement block total as signed for cvt.

            // The half-header widening helper every block in this family runs carries its own scratch set, declared here too.
            "    .reg .u32 bits, payload, exp, mant, sign, fbits;",
            "    .reg .b16 hw;",
            "    .reg .pred p;",
            "    .reg .f32 dv, daf, sumf, wf, sub_, hf;",
            "    .local .align 4 .b8 slot[4];", // declared at column zero so the frame's own stores never collide with it

            "",
            "    ld.param.u64 w, [w_ptr];",
            "    ld.param.u64 qa, [qa_ptr];",
            "    ld.param.u64 dasc, [da_ptr];",
            "    ld.param.u64 o, [out_ptr];",
            "    ld.param.u32 rows_, [rows_arg];",
            "    ld.param.u32 cols_, [cols_arg];",
            "    ld.param.u32 cnt_, [count_arg];",
            "    ld.param.u64 gridBase, [grid_ptr];",
            "    ld.param.u64 tabBase, [tab_ptr];",
            "",
            "    mov.u32 tid, %ctaid.x;",
            "    mov.u32 t0, %ntid.x;",
            "    mul.lo.u32 tid, tid, t0;",
            "    mov.u32 t0, %tid.x;",
            "    add.u32 row, tid, t0;",
            "    setp.ge.u32 p, row, rows_;",
            "    @p bra done;",
            "",
            // Row stride in bytes and the whole-super-block geometry both sides of the seam agree on: ceil(cols / 256) blocks.
            "    add.u32 strideBlocks, cols_, 255;",
            "    shr.u32 strideBlocks, strideBlocks, 8;",
            $"    mul.lo.u32 strideBytes, strideBlocks, {blockBytes};",
            "    mov.u32 qb_, strideBlocks;", // the activation windows cover exactly the same whole-super-block count as the weight row.
            "",
            "    mul.lo.u32 rowBase, row, strideBytes;",
            "    mov.u32 zr, 0;", // a standing zero register for predicated negations: PTX arithmetic takes no immediate operands here.
            "    mov.u32 bb, 0;", // the batch counter starts at this vector: every other loop cursor in these kernels is initialized before its test too.
            "",
            "batch:",
            "    setp.ge.u32 p, bb, cnt_;",
            "    @p bra done;",
            "    mov.f32 sumf, 0f00000000;",
            // This vector's quant window and scale array: block-order contiguous regions per batch vector, as the host laid them out.
            "    mad.lo.u32 t0, bb, qb_, 0;",
            "    cvt.u64.u32 addr, t0;",
            "    shl.b64 addr, addr, 8;", // blocks * 256 bytes of quants per vector ...
            "    add.u64 qblk, qa, addr;",
            "    mad.lo.u32 t1, bb, qb_, 0;",
            "    cvt.u64.u32 addr, t1;", // ... and blocks * four-byte float scales per vector.
            "    shl.b64 addr, addr, 2;",
            "    add.u64 dacur, dasc, addr;",
            "",
            "    cvt.u64.u32 blockAddr, rowBase;",
            "    add.u64 blockAddr, w, blockAddr;",
            "",
            "    mov.u32 base, 0;",
            "_block:",
            "    setp.ge.u32 p, base, cols_;", // whole-super-block walk: same iteration count as the oracle's blocks loop.
            "    @p bra _store;",

            "",
        ];

        lines.AddRange(PtxWidenHalfLE(0, "dv", "_hd"));

        lines.AddRange(
        [
            "    ld.global.f32 daf, [dacur];", // this super-block's activation scale ...
            "    mul.f32 dv, dv, daf;", // ... times the widened half header, in the reference order.
            "",
            "    mov.u32 bsum, 0;",
        ]);

        lines.AddRange(subBlockArms);

        lines.AddRange(
        [
            "    mov.b32 sv, bsum;", // reinterpret the two's-complement block total as signed ...
            "    cvt.rn.f32.s32 wf, sv;", // ... for one exact int -> float conversion of it.
            "    mul.f32 wf, dv, wf;", // sumf += d * bsum: multiplied by d once per super-block in block order.
            "    add.f32 sumf, sumf, wf;",
            "_nextBlock:",
            "    add.u32 base, base, 256;",
            $"    add.u64 blockAddr, blockAddr, {blockBytes};",
            "    add.u64 qblk, qblk, 256;", // the next super-block's quant window sits at the next 256 bytes of this vector's region ...
            "    add.u64 dacur, dacur, 4;", // ... and its scale at the next float slot.
            "    bra _block;",
            "_store:",
        ]);

        if (finalFactor is not null)
        {
            lines.Add($"    mul.f32 sumf, sumf, {finalFactor};"); // the format's final factor lands once at the end.
        }

        lines.AddRange(
        [
            "    mad.lo.u32 t0, bb, rows_, row;",
            "    cvt.u64.u32 addr, t0;",
            "    shl.b64 addr, addr, 2;",
            "    add.u64 addr, o, addr;",
            "    st.global.f32 [addr], sumf;",
            "    add.u32 bb, bb, 1;",
            "    bra batch;",
            "done:",
            "    ret;",
            "}",
        ]);

        return string.Join("\n", [.. lines]) + "\n";
    }

    // Loads one activation dword at this batch vector's window plus a constant byte offset. The region base is allocation-
    // aligned and every site adds a multiple of four, so the load stays aligned on a computed address (the seam answers only
    // computed-address forms). Result lands in `aW`.
    private static IEnumerable<string> PtxLoadActWord(int constOffset) =>
        [
            $"    mov.u32 t0, {constOffset};",
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, qblk;",
            "    ld.global.b32 aW, [addr];",
        ];

    // The four magnitude bytes one activation word dots against come straight from the grid entry: this word's elements sit at
    // laneOffset inside its group's entry, so one dword load answers them and no staging buffer exists on either side. Entry
    // addressing is computed from an index register (widen/shift/add), matching how every other address in these kernels gets there.
    private static IEnumerable<string> PtxLoadEntryDword(string idxReg, int shift, int laneOffset) =>
        [
            "    cvt.u64.u32 addr, " + idxReg + ";",
            $"    shl.b64 addr, addr, {shift};", // 2 for four-byte entries, 3 for eight-byte ones.
            "    add.u64 addr, gridBase, addr;",
            $"    add.u64 addr, addr, {laneOffset};", // the entry's second half sits at plus four inside it.
            "    ld.global.b32 mW, [addr];",
        ];

    // One whole activation word's exact contribution: decode each byte lane's signed value (low7 minus its 128-bit weight - the
    // two's-complement identity that holds for every byte), fold this group's sign bit onto the magnitude with a predicated
    // negate, and accumulate magnitude * quant-bits into `into`. mad.lo.u32 keeps only the low 32 bits of each product, which is
    // where signed and unsigned multiplication agree; magnitudes stay unsigned bytes read from mW. Words feed loU for elements
    // 0..15 and hiU for 16..31, matching SubBlockDotQ8KScalar's two halves exactly. `sg` carries the whole sub-block's four
    // sign rows packed at byte positions 0..3, so each lane tests its group row through an absolute bit position inside it.
    private static IEnumerable<string> PtxWordDot(int wordIndex, string into) => WordDotCore(wordIndex, "sg", into);

    private static List<string> WordDotCore(int j, string signReg, string into)
    {
        // Every element of this word belongs to group (j >> 1): the eight-value rows split at even/odd words and a lane's bit
        // position inside its row is 4 * (word parity) + lane - exactly SubBlockDotQ8KScalar's signs >> 8*(e>>3) & KMask(e&7).
        List<string> lines = [];

        for (int k = 0; k < 4; k++)
        {
            int groupBit = 8 * (j >> 1) + 4 * (j & 1) + k; // absolute position of this element's sign bit inside the packed rows.

            lines.AddRange(
            [
                $"    shr.u32 mvv, mW, {8 * k};", // this lane's magnitude byte ...
                "    and.b32 mvv, mvv, 255;", // ... isolated unsigned from its staged word.
                $"    shr.u32 sbt, {signReg}, {groupBit};", // test this group sign row's bit for the element ...
                "    and.b32 sbt, sbt, 1;",
                "    setp.ne.u32 p, sbt, 0;", // a set bit negates the magnitude; multiplication commutes with the reference's activation-side fold.
                $"    @p sub.u32 mvv, zr, mvv;",
                $"    shr.u32 t1, aW, {8 * k};", // this lane's raw quant byte ...
                "    and.b32 t1, t1, 255;",
                "    mov.u32 t2, t1;", // copy so both halves of the signed decode stay available ...
                "    and.b32 t1, t1, 127;", // low seven bits: the magnitude half ...
                "    and.b32 t2, t2, 128;", // bit 7 carries weight -128 for a negative quant ...
                "    sub.u32 t1, t1, t2;", // ... so the difference is the two's-complement value of this lane's quant byte.
                $"    mad.lo.u32 {into}, mvv, t1, {into};", // exact product accumulated in int32: bounded well inside 32 bits here.
            ]);
        }

        return lines;
    }

    // One byte load at a constant offset inside the current weight block, on this family's computed-address discipline.
    private static IEnumerable<string> PtxLoadBlockByte(int constOffset, string dstReg) =>
        [
            $"    mov.u32 t0, {constOffset};",
            "    cvt.u64.u32 addr, t0;",
            "    add.u64 addr, addr, blockAddr;",
            $"    ld.global.u8 {dstReg}, [addr];",
        ];

    // One ksigns-table byte at a code held in `codeReg` (already masked to seven bits by the caller), loaded on computed-
    // address discipline into `dstReg`. The scratch table region is plain bytes, so alignment carries no constraint.
    private static IEnumerable<string> PtxLoadKSignsByte(string codeReg, string dstReg) =>
        [
            "    cvt.u64.u32 addr, " + codeReg + ";",
            "    add.u64 addr, tabBase, addr;",
            $"    ld.global.u8 {dstReg}, [addr];",
        ];

    // Extracts one seven-bit sign-group code out of a packed aux word at its constant shift and loads the table byte it selects.
    private static IEnumerable<string> PtxSignByteFromAux(string auxReg, int bitShift, string dstReg) =>
        [
            $"    shr.u32 mvv, {auxReg}, {bitShift};", // scratch cursor for the seven-bit code ...
            "    and.b32 mvv, mvv, 127;",
            .. PtxLoadKSignsByte("mvv", dstReg),
        ];

    // Folds a just-loaded sign byte into `sg` at its constant position (0, 8, 16, or 24). A u8 load zero-extends, so the shift
    // and fold land exactly eight bits without masking first.
    private static IEnumerable<string> PtxFoldSignByte(string srcReg, int positionShift) =>
        [
            $"    shl.b32 t1, {srcReg}, {positionShift};",
            "    or.b32 sg, sg, t1;", // hmm: caller must have cleared `sg` once per sub-block before the first fold.

        ];

    // Assembles one little-endian u32 field out of four byte loads at constant block offsets into `dstReg`, using the frame's
    // assembly temp `b1` for each shifted byte. The seam answers only computed addresses, and a packed row is not dword-aligned.
    private static IEnumerable<string> PtxAssembleFieldLE(string dstReg, int offset) =>
        [
            .. PtxLoadBlockByte(offset, dstReg), // low byte lands first ...
            .. PtxLoadBlockByte(offset + 1, "b1"), // each later byte shifts into place and folds in ...
            "    shl.b32 b1, b1, 8;",
            $"    or.b32 {dstReg}, {dstReg}, b1;",
            .. PtxLoadBlockByte(offset + 2, "b1"),
            "    shl.b32 b1, b1, 16;",
            $"    or.b32 {dstReg}, {dstReg}, b1;",
            .. PtxLoadBlockByte(offset + 3, "b1"),
            "    shl.b32 b1, b1, 24;",
            $"    or.b32 {dstReg}, {dstReg}, b1;",
        ];

    // Assembles one little-endian u16 field out of two byte loads at constant block offsets into `dstReg`.
    private static IEnumerable<string> PtxAssembleFieldLE16(string dstReg, int offset) =>
        [
            .. PtxLoadBlockByte(offset, dstReg),
            .. PtxLoadBlockByte(offset + 1, "b1"),
            "    shl.b32 b1, b1, 8;",
            $"    or.b32 {dstReg}, {dstReg}, b1;",
        ];

    // The ls factor 2 * code + 1 (equivalently 1 + 2 * code) out of a scale byte's nibble at its constant block offset: load,
    // isolate the half, double it through one shift and add one. Same integer value either way, matching both reference spellings.
    private static IEnumerable<string> PtxLsFromScaleByte(string dstReg, int constOffset, bool highNibble)
    {
        List<string> lines = [.. PtxLoadBlockByte(constOffset, dstReg)];

        if (highNibble)
        {
            lines.Add($"    shr.u32 {dstReg}, {dstReg}, 4;"); // the odd sub-block's code rides in this byte's high half ...
        }

        lines.AddRange(
        [
            $"    and.b32 {dstReg}, {dstReg}, 15;", // ... either way, four bits isolate it (a low nibble needs only the mask).
            $"    shl.b32 {dstReg}, {dstReg}, 1;", // ls = 2 * code: one shift carries the doubling.
            $"    add.u32 {dstReg}, {dstReg}, 1;", // ... and the odd factor lands as a plain immediate addition.
        ]);

        return lines;
    }

    // The whole-sub-block ls read out of an aux word's top nibble (bits 28..31), left in `dstReg` while `auxReg` stays live for
    // the sign codes that still need it: only a shifted copy is taken, never the field itself.
    private static IEnumerable<string> PtxLsFromAuxTopNibble(string auxReg, string dstReg) =>
        [
            $"    shr.u32 {dstReg}, {auxReg}, 28;", // the scale code sits above all four sign codes in this word ...
            $"    and.b32 {dstReg}, {dstReg}, 15;",
            $"    shl.b32 {dstReg}, {dstReg}, 1;", // ls = 2 * code + 1 again: shift then immediate addition.
            $"    add.u32 {dstReg}, {dstReg}, 1;",
        ];

    // The sub-block tail for the whole-sub-block-scale formats: bsum += (loSum + hiSum) * ls, one exact integer step in the
    // same order RowDotQ8K's scalar reference writes it. `sbt` is dead by every arm's tail and carries the half sum here.
    private static IEnumerable<string> PtxBsumWhole(string lsReg) =>
        [
            "    add.u32 sbt, loU, hiU;",
            $"    mad.lo.u32 bsum, sbt, {lsReg}, bsum;", // exact int32 accumulation of the scaled sub-block total.
        ];

    // The split-scale tail for the formats whose scale byte carries two codes: each half scales by its own factor before it
    // lands in bsum - exactly as `bsum += lo * ls1; bsum += hi * ls2` reads in both reference spellings.
    private static IEnumerable<string> PtxBsumSplit() =>
        [
            "    mad.lo.u32 bsum, loU, lsA, bsum;", // groups 0..1 under the low nibble's factor ...
            "    mad.lo.u32 bsum, hiU, lsB, bsum;", // ... and groups 2..3 under the high nibble's.
        ];

    // Clears both half accumulators at one sub-block arm's head: loSum/hiSum exist only within a sub-block in the reference too.
    private static IEnumerable<string> PtxOpenSubBlock() =>
        [
            "    mov.u32 sg, 0;", // sign rows fold in from here; every group lands at its own byte position later.
            "    mov.u32 loU, 0;",
            "    mov.u32 hiU, 0;",
        ];

    // Extends the index byte already in `gi` with this group's two qh bits at positions eight and nine - exactly StageIQ2SMagnitudes'
    // `qs[l] | (bit 2l << 8) | (bit 2l+1 << 9)` for a ten-bit-capable grid. The pair rides in one byte per sub-block, so both
    // extractions read the same live qh register through shift-and-mask against scratch temps the dot phase has not touched yet.
    private static IEnumerable<string> PtxExtendIndexQhPair(string qhReg, int l) =>
        [
            $"    shr.u32 t1, {qhReg}, {2 * l};", // bit 2l ...
            "    and.b32 t1, t1, 1;",
            "    shl.b32 t1, t1, 8;", // ... lands as the index's eighth bit ...
            "    or.b32 gi, gi, t1;",
            $"    shr.u32 t2, {qhReg}, {2 * l + 1};", // bit 2l+1 ...
            "    and.b32 t2, t2, 1;",
            "    shl.b32 t2, t2, 9;", // ... as its ninth.
            "    or.b32 gi, gi, t2;",
        ];

    // Extends the index byte already in `gi` with one qh bit at position j - StageIQ3SMagnitudes' `qs[j] | (bit j << 8)` for a
    // nine-bit grid whose extra bit spreads across all eight indices of the sub-block.
    private static IEnumerable<string> PtxExtendIndexQhBit(string qhReg, int j) =>
        [
            $"    shr.u32 t1, {qhReg}, {j};", // this index's extension bit ...
            "    and.b32 t1, t1, 1;",
            "    shl.b32 t1, t1, 8;", // ... lands as the index's eighth bit.
            "    or.b32 gi, gi, t1;",
        ];

    // One group of one sub-block for the packed-sign formats: extract its seven-bit code out of a live aux word at the constant
    // shift, look the row up in ksigns, and fold it into `sg` at this group's byte position. The caller keeps every group folded
    // before any dot runs so each lane can test its own absolute bit inside the packed rows.
    // One group of one sub-block for the packed-sign formats: extract its seven-bit code out of a live aux word at the constant
    // shift, look the row up in ksigns, and fold it into `sg` at this group's byte position. The caller keeps every group folded
    // before any dot runs so each lane can test its own absolute bit inside the packed rows.
    private static IEnumerable<string> PtxSignGroup(string auxReg, int l) =>
        [
            .. PtxSignByteFromAux(auxReg, 7 * l, "t1"), // code cursor rides in mvv; the table byte lands ready to fold ...
            $"    shl.b32 t1, t1, {8 * l};", // ... at this group's position inside the packed sign word ...
            "    or.b32 sg, sg, t1;", // ... folded without disturbing any earlier group.

        ];
}

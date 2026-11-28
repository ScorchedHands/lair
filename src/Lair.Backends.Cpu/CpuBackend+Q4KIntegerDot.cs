using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Lair.Contracts;

namespace Lair.Backends.Cpu;

/// <summary>
/// Integer-domain AVX2 Q4_K dense batched dot (internal, unpromoted): every batch vector of the pass is staged through
/// the same Q8_K activation quantization the sub-4-bit i-quant dots already build (<see cref="QuantizeActivationsToQ8K"/>),
/// and each weight row's nibbles are dotted against those signed quants in integer lanes — one AND/SRL nibble unpack per
/// group pair, adjacent-pair products summed in int16 lanes (pmaddubsw) then the group scale folded into int32 lane
/// positions (pmaddwd), and at every super-block boundary exactly one fold of each chain's exact int32 lane totals plus
/// two float FMAs: the block's fused activation-scale x weight-scale coefficient against the nibble total, then its
/// negated min coefficient against the group sums weighted by the mins. This is the instruction shape the pinned
/// reference build applies to a Q4_K row dotted against quantized activations; LAIR's chain groups carry up to eight
/// accumulator chains across one shared nibble-unpack pass, which that walk cannot — it serves one activation pair per
/// weight traversal. Sharing runs both ways: within a group the AND/SRL unpack and both group-scale shuffles are computed
/// once per window for every chain, each chain only loads its own staged quants and issues its two pmaddubsw/pmaddwd
/// pairs, and the block's eight group mins are unpacked once for the whole group. What a chain does not re-read is the
/// activation side of its min term: this variant quantizes nothing inside a weight-row walk (staging happens once per
/// call) and folds each chain's exact per-group activation sums out of a table built alongside that staging, so one pass
/// through a row serves the group at integer-MAC cost rather than re-summing quants per chain per block. The variant runs
/// only when its stated batch width is reached and the ISA oracle resolved this backend to the AVX2 rung: a host whose auto
/// selection lands on vpdpbusd or on scalar reference dots keeps today's float traversals untouched, whatever a seam states
/// about it. Quantizing FP32 activations to signed bytes loses precision against them, so outputs are not expected
/// bit-identical to the float kernels; what this variant guarantees instead is its own determinism contract — every chain
/// walks ascending super-blocks with fixed int32 lane accumulation order and a fixed per-block float fold sequence, so
/// batched equals solo equals threaded at any worker count, whatever chain grouping or row partition splits them. Ragged
/// columns past the real width are exactly zero quants in every vector's staged region (the quantizer pads whole
/// super-blocks), so they carry no nibble product and add nothing to a group sum: each chain sees identical operands at any
/// column count, which is how the Q8_K staging already defines padding for the i-quant dots. Overflow bounds by
/// construction: an int16 pair sum is at most 2*15*127 = 3810 and a scale-folded lane sums two such products under sc <= 63,
/// while one chain's exact int32 total covers only the current super-block (bounded by 256 * 63 * 15 * 127) before every
/// block boundary folds it into float — so no integer accumulator saturates anywhere. A group sum is at most 32 * 127 and a
/// min-weighted product of one fits an int32 (63 * 4064), as do the eight such products folded per chain-block.
/// </summary>
public sealed unsafe partial class CpuBackend
{
    // Q4_K's eight groups per super-block: the count of activation group sums this variant stages per chain and block, and
    // the width of every min/scale unpack it performs (see the layout documented at QkK in CpuBackend.cs).
    private const int Q4KGroups = 8;

    // Whether one Q4_K dense pass takes the unpromoted integer executor: only when a width is stated (no resolved posture
    // states one), the batch reached it, and this backend's ISA resolution actually landed on the AVX2 rung. A Scalar or
    // Vnni resolution keeps today's float kernels — the variant is stated for pmaddubsw/pmaddwd work only, and a resolved
    // encoding never substitutes another integer-dot ladder for what was asked of it.
    private bool Q4KIntegerDotSelected(QuantFormat format, int count) =>
        format == QuantFormat.Q4_K && _useAvx2 && _integerDot == IntegerDotPath.Avx2 && _tuning.Q4KIntegerDotBatchWidth > 0 && count >= _tuning.Q4KIntegerDotBatchWidth;

    // Stage each chain's exact activation group sums for the whole batch, once per call: one int16 lane value per (chain,
    // super-block, group), computed with the same pmaddubsw-against-1s plus pmaddwd fold this variant's dots use. A group of
    // 32 signed quants sums within +/-4064, so an exact sum always lands in an int16 lane. These are pure functions of the
    // staged quants — no weight byte touches them — so every row and worker of the call reads identical values from one
    // table, which is what keeps a chain-group pass at integer-MAC cost instead of re-summing each chain's 32-element groups
    // inside every weight-row walk. Padding past the real column count quantizes as zero quants, so a ragged tail
    // contributes exactly nothing to its group sums, exactly as it does in the staged regions themselves.
    private static void StageQ4KIntegerGroupSums(byte* qaBase, int blocks, int count, short* groupSums)
    {
        for (int b = 0; b < count; b++)
        {
            byte* qaRow = qaBase + (long)b * blocks * QkK; // the same contiguous region QuantizeActivationsToQ8K laid out for this vector.

            for (int i = 0; i < blocks; i++)
            {
                byte* blockQuants = qaRow + i * QkK;

                for (int j = 0; j < Q4KGroups; j++)
                {
                    Vector256<sbyte> act = Avx.LoadVector256(blockQuants + j * 32).AsSByte(); // this group's signed quants, in [-127, 127] by the quantizer.

                    Vector256<short> pairs = Avx2.MultiplyAddAdjacent(Vector256.Create((byte)1), act); // pmaddubsw against 1s: adjacent pair sums in short lanes (|pair| <= 254).
                    Vector128<int> quadsLo = Sse2.MultiplyAddAdjacent(pairs.GetLower(), GetOnesShort()); // fold pairs into quad totals in exact int32 lanes.
                    Vector128<int> quadsHi = Sse2.MultiplyAddAdjacent(pairs.GetUpper(), GetOnesShort());

                    groupSums[((long)b * blocks + i) * Q4KGroups + j] = (short)(HSumInt32(quadsLo) + HSumInt32(quadsHi)); // this chain's exact sum of one group, ascending (chain, block, group) order — within +/- 4064 by the quantizer's own range.
                }
            }
        }
    }

    private static Vector128<short> GetOnesShort() => Vector128.Create((short)1); // the 1s that fold this variant's pair sums into exact int32 lane totals.

    // One row's integer batch dot. Chain grouping arrives from the call's resolved tuning snapshot and only decides how
    // many accumulator chains share one nibble-unpack pass: each chain advances its own staged activation region in the
    // same ascending order whatever group it sits in, so every output is bit-identical at any group width or split. A full
    // register-budget group runs with all eight int32 lane accumulators live across the row; a narrower pinned group and a
    // ragged tail below one group run the same chains staged through stack lanes instead of live YMM registers, exactly as
    // the scratch batch dot stages its float chains for grouped walks. Group sums come from this call's shared table: a
    // chain-block fold reads that chain's eight exact sums and weights them by the block's unpacked mins in int16 lanes.
    private static void RowBatchDotQ4KInteger(byte* rowBytes, byte* qa, float* da, short* groupSums, int blocks, int count, int rows, int chainGroupWidth, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch];

        fixed (float* pAcc = acc)
        {
            for (int b0 = 0; b0 < count; b0++)
            {
                pAcc[b0] = 0f; // every chain starts its row cold: the fold sequence is identical whatever group walked it.
            }

            int width = Math.Min(chainGroupWidth, CpuKernelTuning.ChainGroupMax); // a pinned group never exceeds the register budget here either.

            long activationStride = (long)blocks * QkK; // one vector's staged region: whole super-blocks, as QuantizeActivationsToQ8K laid them out.
            long sumsStride = (long)blocks * Q4KGroups; // the same chain-by-chain layout for this call's group-sum table (eight exact sums per block).

            for (int b0 = 0; b0 < count;)
            {
                int chains = Math.Min(width, count - b0); // a ragged tail below one group runs through the stacked-chain arm.

                byte* qaGroup = qa + b0 * activationStride; // each vector's staged region is whole super-blocks, as QuantizeActivationsToQ8K laid them out.
                float* daGroup = da + b0 * blocks;
                short* sumsGroup = groupSums + b0 * sumsStride; // the same chains' staged group sums, one eight-lane row per super-block.

                if (chains == CpuKernelTuning.ChainGroupMax)
                {
                    Q4KIntegerChains8(rowBytes, qaGroup, daGroup, sumsGroup, blocks, activationStride, sumsStride, pAcc + b0);
                }
                else
                {
                    Q4KIntegerStackChains(rowBytes, qaGroup, daGroup, sumsGroup, activationStride, sumsStride, blocks, chains, pAcc + b0);
                }

                b0 += chains;
            }

            for (int b = 0; b < count; b++)
            {
                outBase[(long)b * rows] = pAcc[b]; // each chain's output is its own accumulator: grouping never folds one chain into another.
            }
        }
    }

    // The register-budget arm: eight chains' int32 lane accumulators stay live across the whole row while every shared
    // nibble window (both AND/SRL unpacks) and both group-scale shuffles feed all of them, and only each chain's staged
    // activation windows load per (chain, group pair) — one weight traversal amortized over a full group. The block's eight
    // mins are unpacked once for the whole group too, and each chain's min term folds its own table row against those mins
    // in int32 lanes rather than re-summing quants per window. Block boundaries fold through the shared helpers in fixed order.
    private static void Q4KIntegerChains8(byte* row, byte* qaBase, float* daBase, short* groupSumsBase, int blocks, long activationStride, long sumsStride, float* acc)
    {
        Vector256<byte> nibbleMask = Vector256.Create((byte)0x0F); // the same nibble mask every Q4_K kernel applies, broadcast across both halves.

        Span<short> mins = stackalloc short[Q4KGroups]; // this arm's scratch row for one block's unpacked mins: read once per block and shared by every chain's fold.

        for (int i = 0; i < blocks; i++)
        {
            byte* block = row + (long)i * Q4KBlockBytes;

            float dHalf = ReadHalfLE(block); // the same half-float coefficients the float kernels read, folded per chain at this block boundary.
            float minHalf = ReadHalfLE(block + 2);

            Vector256<short> scales = LoadGroupScalesQ4K(block); // the unpacked group scales GetScaleMinK4 yields, widened to int16 lanes for pmaddwd.
            LoadGroupMinsQ4K(block, mins); // this block's eight unpacked mins: read once here and shared by every chain's fold.

            byte* qaRow = qaBase + (long)i * QkK; // each vector's staged region is whole super-blocks, so chains sit one full row-region apart — not one block apart.
            short* sumsRow = groupSumsBase + i * Q4KGroups; // the same ascending layout for this block's eight exact group sums per chain.

            Vector256<int> s0 = Vector256<int>.Zero, s1 = Vector256<int>.Zero, s2 = Vector256<int>.Zero, s3 = Vector256<int>.Zero;
            Vector256<int> s4 = Vector256<int>.Zero, s5 = Vector256<int>.Zero, s6 = Vector256<int>.Zero, s7 = Vector256<int>.Zero;

            for (int jp = 0; jp < QkK / 64; jp++)
            {
                byte* q4 = block + 16 + 32 * jp; // [q, q+32) covers groups 2jp and 2jp+1 entirely inside the block.

                Vector256<byte> nibbles = Avx.LoadVector256(q4);
                Vector256<short> scaleL = Avx2.Shuffle(scales.AsByte(), GetScaleShuffleK4(2 * jp)).AsInt16(); // group 2jp broadcast across every pair position of each half.
                Vector256<short> scaleH = Avx2.Shuffle(scales.AsByte(), GetScaleShuffleK4(2 * jp + 1)).AsInt16();

                Vector256<byte> lowNibbles = Avx2.And(nibbles, nibbleMask); // the shared unpack: group 2jp's values in each byte lane's low bits.
                Vector256<byte> highNibbles = Avx2.And(Avx2.ShiftRightLogical(nibbles.AsUInt16(), 4).AsByte(), nibbleMask); // srli_epi16 then byte mask: group 2jp+1, unpacked once for the whole group.

                byte* q8 = qaRow + 64 * jp; // each chain's own staged activation region at the same ascending position, one row-region (blocks x QkK) apart per chain.

                s0 = Q4KAccumulatePairDot(s0, lowNibbles, highNibbles, scaleL, scaleH, q8);
                s1 = Q4KAccumulatePairDot(s1, lowNibbles, highNibbles, scaleL, scaleH, q8 + activationStride);
                s2 = Q4KAccumulatePairDot(s2, lowNibbles, highNibbles, scaleL, scaleH, q8 + 2 * activationStride);
                s3 = Q4KAccumulatePairDot(s3, lowNibbles, highNibbles, scaleL, scaleH, q8 + 3 * activationStride);
                s4 = Q4KAccumulatePairDot(s4, lowNibbles, highNibbles, scaleL, scaleH, q8 + 4 * activationStride);
                s5 = Q4KAccumulatePairDot(s5, lowNibbles, highNibbles, scaleL, scaleH, q8 + 5 * activationStride);
                s6 = Q4KAccumulatePairDot(s6, lowNibbles, highNibbles, scaleL, scaleH, q8 + 6 * activationStride);
                s7 = Q4KAccumulatePairDot(s7, lowNibbles, highNibbles, scaleL, scaleH, q8 + 7 * activationStride);
            }

            // Block boundary: every chain folds its exact int32 totals into float in fixed order — nibble total under the
            // fused activation x weight scale first, then the min-weighted group sums under the negated min coefficient. Each
            // fold reads that chain's own table row for this block (chains sit blocks x Q4KGroups apart in the shared table).
            FoldQ4KIntegerBlock(acc + 0, s0, Q4KMinTermFromSums(sumsRow, mins), daBase[i] * dHalf, -(daBase[i] * minHalf));
            FoldQ4KIntegerBlock(acc + 1, s1, Q4KMinTermFromSums(sumsRow + sumsStride, mins), daBase[blocks + i] * dHalf, -(daBase[blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 2, s2, Q4KMinTermFromSums(sumsRow + 2 * sumsStride, mins), daBase[2 * blocks + i] * dHalf, -(daBase[2 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 3, s3, Q4KMinTermFromSums(sumsRow + 3 * sumsStride, mins), daBase[3 * blocks + i] * dHalf, -(daBase[3 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 4, s4, Q4KMinTermFromSums(sumsRow + 4 * sumsStride, mins), daBase[4 * blocks + i] * dHalf, -(daBase[4 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 5, s5, Q4KMinTermFromSums(sumsRow + 5 * sumsStride, mins), daBase[5 * blocks + i] * dHalf, -(daBase[5 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 6, s6, Q4KMinTermFromSums(sumsRow + 6 * sumsStride, mins), daBase[6 * blocks + i] * dHalf, -(daBase[6 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 7, s7, Q4KMinTermFromSums(sumsRow + 7 * sumsStride, mins), daBase[7 * blocks + i] * dHalf, -(daBase[7 * blocks + i] * minHalf));
        }
    }

    // The grouped arm for narrower pinned groups and ragged tails: the same ascending walk with each chain's eight int32
    // accumulator lanes staged through a stack slot instead of live YMM registers — more load/store traffic, identical bits.
    private static void Q4KIntegerStackChains(byte* row, byte* qaBase, float* daBase, short* groupSumsBase, long activationStride, long sumsStride, int blocks, int chains, float* acc)
    {
        Span<int> lanes = stackalloc int[CpuKernelTuning.ChainGroupMax * 8]; // eight exact accumulator lanes per chain.

        Vector256<byte> nibbleMask = Vector256.Create((byte)0x0F);

        Span<short> mins = stackalloc short[Q4KGroups]; // this arm's scratch row for one block's unpacked mins: read once per block and shared by every chain's fold.

        for (int i = 0; i < blocks; i++)
        {
            byte* block = row + (long)i * Q4KBlockBytes;

            float dHalf = ReadHalfLE(block);
            float minHalf = ReadHalfLE(block + 2);

            Vector256<short> scales = LoadGroupScalesQ4K(block); // the same two shuffles serve every chain in this group.
            LoadGroupMinsQ4K(block, mins);

            byte* qaRow = qaBase + (long)i * QkK;
            short* sumsRow = groupSumsBase + i * Q4KGroups;

            fixed (int* pLanes = lanes)
            {
                for (int c = 0; c < chains * 8; c++)
                {
                    pLanes[c] = 0; // every chain starts each super-block cold: lane totals cover exactly one block.
                }

                for (int jp = 0; jp < QkK / 64; jp++)
                {
                    byte* q4 = block + 16 + 32 * jp; // the shared nibble window feeds every chain in the group.

                    Vector256<byte> nibbles = Avx.LoadVector256(q4);
                    Vector256<short> scaleL = Avx2.Shuffle(scales.AsByte(), GetScaleShuffleK4(2 * jp)).AsInt16(); // group 2jp broadcast across every pair position of each half.
                    Vector256<short> scaleH = Avx2.Shuffle(scales.AsByte(), GetScaleShuffleK4(2 * jp + 1)).AsInt16();

                    Vector256<byte> lowNibbles = Avx2.And(nibbles, nibbleMask); // unpacked once for the group: even pairs belong to group 2jp.
                    Vector256<byte> highNibbles = Avx2.And(Avx2.ShiftRightLogical(nibbles.AsUInt16(), 4).AsByte(), nibbleMask); // odd pairs belong to group 2jp+1.

                    byte* q8 = qaRow + 64 * jp; // each nibble window spans two activation groups: 32 packed bytes cover 64 quant positions.

                    for (int c = 0; c < chains; c++)
                    {
                        int* chainLanes = pLanes + c * 8; // the accumulator slot: one load and store per group pair, identical lane order to the register arm.

                        Vector256<int> accChain = Avx.LoadVector256(chainLanes);
                        accChain = Q4KAccumulatePairDot(accChain, lowNibbles, highNibbles, scaleL, scaleH, q8 + (long)c * activationStride);
                        Avx.Store(chainLanes, accChain);
                    }
                }

                for (int c = 0; c < chains; c++)
                {
                    FoldQ4KIntegerBlock(
                        acc + c,
                        Avx.LoadVector256(pLanes + c * 8), // the chain's eight exact lane totals, folded in fixed order.
                        Q4KMinTermFromSums(sumsRow + c * sumsStride, mins), // this chain's table row for this block: chains sit blocks x eight sums apart.
                        daBase[(long)c * blocks + i] * dHalf,
                        -(daBase[(long)c * blocks + i] * minHalf));
                }
            }
        }
    }

    // One (chain, group pair) of the dot against weight nibbles already unpacked once for the whole chain group: each group's
    // values multiplied by that chain's signed quants into adjacent-pair int16 sums (pmaddubsw). A 32-byte window covers two
    // groups' elements in interleaved pair positions — even pairs belong to the low-nibble group and odd pairs to the
    // high-nibble group — so each scale is applied through pmaddwd against a per-group position mask (0xFFFF for one half of
    // the window's pairs, 0x0000 for the other) before both groups' exact int32 totals are summed into this chain's block
    // accumulator. Pair products are bounded by 15 * 127 and their adjacent sum by 3810; a scale-folded dword holds one such
    // product times sc <= 63, so no lane saturates anywhere. Only the two activation loads and four integer MACs per chain
    // remain in this shape: the unpack, both shuffles, and the min-term fold all belong to shared or table-side work.
    private static Vector256<int> Q4KAccumulatePairDot(Vector256<int> accChain, Vector256<byte> lowNibbles, Vector256<byte> highNibbles, Vector256<short> scaleL, Vector256<short> scaleH, byte* q8)
    {
        Vector256<sbyte> aL = Avx.LoadVector256(q8).AsSByte(); // group 2jp's quants for this chain: signed, in [-127, 127] by the quantizer.
        Vector256<short> pairsL = Avx2.MultiplyAddAdjacent(lowNibbles, aL);

        Vector256<sbyte> aH = Avx.LoadVector256(q8 + 32).AsSByte(); // group 2jp+1's quants: the high nibbles of the same weight window.
        Vector256<short> pairsH = Avx2.MultiplyAddAdjacent(highNibbles, aH);

        Vector256<int> sumj = Avx2.Add(Avx2.MultiplyAddAdjacent(pairsL, scaleL), Avx2.MultiplyAddAdjacent(pairsH, scaleH));
        return Avx2.Add(accChain, sumj); // ascending group-pair order inside the chain's block accumulator.
    }

    // The min term for one (chain, block): that chain's eight exact activation group sums weighted by this block's unpacked
    // mins and folded to an exact total — LAIR's fold of what the pinned reference precomputes as quantized-activation bsums
    // (LAIR's Q8_K staging carries no bsums; its dots read these sums from the call's shared table instead). The eight
    // products are plain int32 arithmetic: a group sum lands in an int16 at +/- 32 * 127 and a min is at most 63, so one
    // product tops out at 63 * 4064 and the eight of them fold exactly. It runs once per (chain, block) — outside every
    // nibble window — which is what keeps the shared table worth its staging cost.
    private static int Q4KMinTermFromSums(short* chainGroupSums, Span<short> mins)
    {
        int total = 0;

        for (int j = 0; j < Q4KGroups; j++)
        {
            total += chainGroupSums[j] * mins[j]; // ascending group order in both rows: the same eight pairs whatever arm folded them.
        }

        return total;
    }

    private static void FoldQ4KIntegerBlock(float* accSlot, Vector256<int> lanes, int minTerm, float scaleCoef, float negMinCoef)
    {
        int tInt = HSumInt32(lanes.GetLower()) + HSumInt32(lanes.GetUpper()); // fixed-order fold: the lower half's four exact lane totals, then the upper half's.

        accSlot[0] += scaleCoef * (float)tInt; // the block's two rounded accumulate steps, in fixed order: nibble total under the fused activation x weight scale first...
        accSlot[0] += negMinCoef * (float)minTerm; // ...then the min-weighted group sums under the negated min coefficient.
    }

    // The block's eight group scales as int16 lanes duplicated across both YMM halves: the same GetScaleMinK4 unpacking every
    // Q4_K kernel already performs, staged for pmaddwd instead of a per-lane float multiply.
    private static Vector256<short> LoadGroupScalesQ4K(byte* block)
    {
        Span<short> scales = stackalloc short[Q4KGroups];

        for (int j = 0; j < Q4KGroups; j++)
        {
            GetScaleMinK4(j, block + 4, out byte sc, out _);
            scales[j] = sc;
        }

        Vector128<short> half;
        fixed (short* p = scales)
        {
            half = Sse2.LoadVector128(p); // eight unpacked scales as int16 lanes 0..7 of each duplicated half.
        }

        return Vector256.Create(half, half);
    }

    // The same block's eight group mins, unpacked once per super-block and shared by every chain's min-term fold — which is
    // what replaces the old per-(chain, group) GetScaleMinK4 unpacking inside each nibble window. A min is at most 63 after
    // unpacking, so it lands in an int16 exactly like a staged group sum does.
    private static void LoadGroupMinsQ4K(byte* block, Span<short> mins)
    {
        for (int j = 0; j < Q4KGroups; j++)
        {
            GetScaleMinK4(j, block + 4, out _, out byte m);
            mins[j] = m; // ascending group order in lanes 0..7, aligned with a chain's group-sum row.
        }
    }

    // The pmaddwd shuffle that broadcasts one group's scale across every pair position of its window: destination bytes take the
    // two halves of that group's int16 lane in alternating order (pshufb indexes bytes; an int16 value spans exactly two), and
    // the same pattern fills both YMM halves because a chain's 32-element group spans both.
    private static Vector256<byte> GetScaleShuffleK4(int group)
    {
        byte lo = (byte)(2 * group); // this group's scale lane: its low byte at even destination positions, high byte at odd ones.

        Span<byte> pattern = stackalloc byte[32];
        for (int d = 0; d < 32; d++)
        {
            pattern[d] = (byte)((d & 1) == 0 ? lo : (byte)(lo + 1)); // alternating half-selection across the sixteen int16 positions of both halves.
        }

        fixed (byte* p = pattern)
        {
            return Vector256.Create(Sse2.LoadVector128(p), Sse2.LoadVector128(p + 16)); // identical in both halves: group scales land at their own pair windows' lanes there too.
        }
    }
}

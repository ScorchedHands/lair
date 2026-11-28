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
/// weight traversal. The variant runs only when its stated batch width is reached and the ISA oracle resolved this
/// backend to the AVX2 rung: a host whose auto selection lands on vpdpbusd or on scalar reference dots keeps today's
/// float traversals untouched, whatever a seam states about it. Quantizing FP32 activations to signed bytes loses
/// precision against them, so outputs are not expected bit-identical to the float kernels; what this variant guarantees
/// instead is its own determinism contract — every chain walks ascending super-blocks with fixed int32 lane accumulation
/// order and a fixed per-block float fold sequence, so batched equals solo equals threaded at any worker count, whatever
/// chain grouping or row partition splits them. Ragged columns past the real width are exactly zero quants in every
/// vector's staged region (the quantizer pads whole super-blocks), so they carry no nibble product and add nothing to a
/// group sum: each chain sees identical operands at any column count, which is how the Q8_K staging already defines
/// padding for the i-quant dots. Overflow bounds by construction: an int16 pair sum is at most 2*15*127 = 3810 and a
/// scale-folded lane sums two such products under sc <= 63, while one chain's exact int32 total covers only the current
/// super-block (bounded by 256 * 63 * 15 * 127) before every block boundary folds it into float — so no integer
/// accumulator saturates anywhere.
/// </summary>
public sealed unsafe partial class CpuBackend
{
    // Whether one Q4_K dense pass takes the unpromoted integer executor: only when a width is stated (no resolved posture
    // states one), the batch reached it, and this backend's ISA resolution actually landed on the AVX2 rung. A Scalar or
    // Vnni resolution keeps today's float kernels — the variant is stated for pmaddubsw/pmaddwd work only, and a resolved
    // encoding never substitutes another integer-dot ladder for what was asked of it.
    private bool Q4KIntegerDotSelected(QuantFormat format, int count) =>
        format == QuantFormat.Q4_K && _useAvx2 && _integerDot == IntegerDotPath.Avx2 && _tuning.Q4KIntegerDotBatchWidth > 0 && count >= _tuning.Q4KIntegerDotBatchWidth;

    // One row's integer batch dot. Chain grouping arrives from the call's resolved tuning snapshot and only decides how
    // many accumulator chains share one nibble-unpack pass: each chain advances its own staged activation region in the
    // same ascending order whatever group it sits in, so every output is bit-identical at any group width or split. A full
    // register-budget group runs with all eight int32 lane accumulators live across the row; a narrower pinned group and a
    // ragged tail below one group run the same chains staged through stack lanes instead of live YMM registers, exactly as
    // the scratch batch dot stages its float chains for grouped walks.
    private static void RowBatchDotQ4KInteger(byte* rowBytes, byte* qa, float* da, int blocks, int count, int rows, int chainGroupWidth, float* outBase)
    {
        Span<float> acc = stackalloc float[MaxBatch];

        fixed (float* pAcc = acc)
        {
            for (int b0 = 0; b0 < count; b0++)
            {
                pAcc[b0] = 0f; // every chain starts its row cold: the fold sequence is identical whatever group walked it.
            }

            int width = Math.Min(chainGroupWidth, CpuKernelTuning.ChainGroupMax); // a pinned group never exceeds the register budget here either.

            for (int b0 = 0; b0 < count;)
            {
                int chains = Math.Min(width, count - b0); // a ragged tail below one group runs through the stacked-chain arm.

                byte* qaGroup = qa + (long)b0 * blocks * QkK; // each vector's staged region is whole super-blocks, as QuantizeActivationsToQ8K laid them out.
                float* daGroup = da + b0 * blocks;

                if (chains == CpuKernelTuning.ChainGroupMax)
                {
                    Q4KIntegerChains8(rowBytes, qaGroup, daGroup, blocks, pAcc + b0);
                }
                else
                {
                    Q4KIntegerStackChains(rowBytes, qaGroup, daGroup, blocks * QkK, blocks, chains, pAcc + b0);
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
    // nibble window and scale shuffle feeds all of them, and only each chain's staged activation windows load per
    // (chain, group pair) — one weight traversal amortized over a full group. Block boundaries fold through the shared
    // helpers in fixed order.
    private static void Q4KIntegerChains8(byte* row, byte* qaBase, float* daBase, int blocks, float* acc)
    {
        Span<int> minTerms = stackalloc int[CpuKernelTuning.ChainGroupMax]; // exact per-chain group sums weighted by each block's mins; re-zeroed at every block boundary.
        int activationStride = blocks * QkK; // the staged region size one vector occupies, matching how QuantizeActivationsToQ8K lays them out.

        for (int i = 0; i < blocks; i++)
        {
            byte* block = row + (long)i * Q4KBlockBytes;

            float dHalf = ReadHalfLE(block); // the same half-float coefficients the float kernels read, folded per chain at this block boundary.
            float minHalf = ReadHalfLE(block + 2);

            Vector256<short> scales = LoadGroupScalesQ4K(block); // the unpacked group scales GetScaleMinK4 yields, widened to int16 lanes for pmaddwd.

            byte* qaRow = qaBase + (long)i * QkK; // each vector's staged region is whole super-blocks, so chains sit one full row-region apart — not one block apart.
            minTerms.Clear();

            Vector256<int> s0 = Vector256<int>.Zero, s1 = Vector256<int>.Zero, s2 = Vector256<int>.Zero, s3 = Vector256<int>.Zero;
            Vector256<int> s4 = Vector256<int>.Zero, s5 = Vector256<int>.Zero, s6 = Vector256<int>.Zero, s7 = Vector256<int>.Zero;

            for (int jp = 0; jp < QkK / 64; jp++)
            {
                byte* q4 = block + 16 + 32 * jp; // [q, q+32) covers groups 2jp and 2jp+1 entirely inside the block.

                Vector256<byte> nibbles = Avx.LoadVector256(q4);
                Vector256<short> scaleL = Avx2.Shuffle(scales.AsByte(), GetScaleShuffleK4(2 * jp)).AsInt16(); // group 2jp broadcast across every pair position of each half.
                Vector256<short> scaleH = Avx2.Shuffle(scales.AsByte(), GetScaleShuffleK4(2 * jp + 1)).AsInt16();

                byte* q8 = qaRow + 64 * jp; // each chain's own staged activation region at the same ascending position, one row-region (blocks x QkK) apart per chain.

                s0 = Q4KAccumulatePairDot(s0, nibbles, scaleL, scaleH, q8);
                minTerms[0] += Q4KMinTerm(block, 2 * jp, q8) + Q4KMinTerm(block, 2 * jp + 1, q8 + 32);

                s1 = Q4KAccumulatePairDot(s1, nibbles, scaleL, scaleH, q8 + activationStride);
                minTerms[1] += Q4KMinTerm(block, 2 * jp, q8 + activationStride) + Q4KMinTerm(block, 2 * jp + 1, q8 + activationStride + 32);

                s2 = Q4KAccumulatePairDot(s2, nibbles, scaleL, scaleH, q8 + 2 * activationStride);
                minTerms[2] += Q4KMinTerm(block, 2 * jp, q8 + 2 * activationStride) + Q4KMinTerm(block, 2 * jp + 1, q8 + 2 * activationStride + 32);

                s3 = Q4KAccumulatePairDot(s3, nibbles, scaleL, scaleH, q8 + 3 * activationStride);
                minTerms[3] += Q4KMinTerm(block, 2 * jp, q8 + 3 * activationStride) + Q4KMinTerm(block, 2 * jp + 1, q8 + 3 * activationStride + 32);

                s4 = Q4KAccumulatePairDot(s4, nibbles, scaleL, scaleH, q8 + 4 * activationStride);
                minTerms[4] += Q4KMinTerm(block, 2 * jp, q8 + 4 * activationStride) + Q4KMinTerm(block, 2 * jp + 1, q8 + 4 * activationStride + 32);

                s5 = Q4KAccumulatePairDot(s5, nibbles, scaleL, scaleH, q8 + 5 * activationStride);
                minTerms[5] += Q4KMinTerm(block, 2 * jp, q8 + 5 * activationStride) + Q4KMinTerm(block, 2 * jp + 1, q8 + 5 * activationStride + 32);

                s6 = Q4KAccumulatePairDot(s6, nibbles, scaleL, scaleH, q8 + 6 * activationStride);
                minTerms[6] += Q4KMinTerm(block, 2 * jp, q8 + 6 * activationStride) + Q4KMinTerm(block, 2 * jp + 1, q8 + 6 * activationStride + 32);

                s7 = Q4KAccumulatePairDot(s7, nibbles, scaleL, scaleH, q8 + 7 * activationStride);
                minTerms[7] += Q4KMinTerm(block, 2 * jp, q8 + 7 * activationStride) + Q4KMinTerm(block, 2 * jp + 1, q8 + 7 * activationStride + 32);
            }

            // Block boundary: every chain folds its exact int32 totals into float in fixed order — nibble total under the
            // fused activation x weight scale first, then the min-weighted group sums under the negated min coefficient.
            FoldQ4KIntegerBlock(acc + 0, s0, minTerms[0], daBase[i] * dHalf, -(daBase[i] * minHalf));
            FoldQ4KIntegerBlock(acc + 1, s1, minTerms[1], daBase[blocks + i] * dHalf, -(daBase[blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 2, s2, minTerms[2], daBase[2 * blocks + i] * dHalf, -(daBase[2 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 3, s3, minTerms[3], daBase[3 * blocks + i] * dHalf, -(daBase[3 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 4, s4, minTerms[4], daBase[4 * blocks + i] * dHalf, -(daBase[4 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 5, s5, minTerms[5], daBase[5 * blocks + i] * dHalf, -(daBase[5 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 6, s6, minTerms[6], daBase[6 * blocks + i] * dHalf, -(daBase[6 * blocks + i] * minHalf));
            FoldQ4KIntegerBlock(acc + 7, s7, minTerms[7], daBase[7 * blocks + i] * dHalf, -(daBase[7 * blocks + i] * minHalf));
        }
    }

    // The grouped arm for narrower pinned groups and ragged tails: the same ascending walk with each chain's eight int32
    // accumulator lanes staged through a stack slot instead of live YMM registers — more load/store traffic, identical bits.
    private static void Q4KIntegerStackChains(byte* row, byte* qaBase, float* daBase, int activationStride, int blocks, int chains, float* acc)
    {
        Span<int> lanes = stackalloc int[CpuKernelTuning.ChainGroupMax * 8]; // eight exact accumulator lanes per chain.
        Span<int> minTerms = stackalloc int[CpuKernelTuning.ChainGroupMax]; // exact per-chain group sums weighted by each block's mins; re-zeroed at every block boundary.

        for (int i = 0; i < blocks; i++)
        {
            byte* block = row + (long)i * Q4KBlockBytes;

            float dHalf = ReadHalfLE(block);
            float minHalf = ReadHalfLE(block + 2);

            Vector256<short> scales = LoadGroupScalesQ4K(block);

            byte* qaRow = qaBase + (long)i * QkK;
            minTerms.Clear();

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

                    byte* q8 = qaRow + 64 * jp; // each nibble window spans two activation groups: 32 packed bytes cover 64 quant positions.

                    for (int c = 0; c < chains; c++)
                    {
                        int* chainLanes = pLanes + c * 8; // the accumulator slot: one load and store per group pair, identical lane order to the register arm.

                        Vector256<int> accChain = Avx.LoadVector256(chainLanes);
                        accChain = Q4KAccumulatePairDot(accChain, nibbles, scaleL, scaleH, q8 + (long)c * activationStride);
                        Avx.Store(chainLanes, accChain);

                        minTerms[c] += Q4KMinTerm(block, 2 * jp, q8 + (long)c * activationStride) + Q4KMinTerm(block, 2 * jp + 1, q8 + (long)c * activationStride + 32);
                    }
                }

                for (int c = 0; c < chains; c++)
                {
                    FoldQ4KIntegerBlock(
                        acc + c,
                        Avx.LoadVector256(pLanes + c * 8), // the chain's eight exact lane totals, folded in fixed order.
                        minTerms[c],
                        daBase[(long)c * blocks + i] * dHalf,
                        -(daBase[(long)c * blocks + i] * minHalf));
                }
            }
        }
    }

    // One (chain, group pair) of the dot: AND/SRL nibble unpack once per window, then each group's nibbles multiplied by
    // that chain's signed quants into adjacent-pair int16 sums (pmaddubsw). A 32-byte nibble window covers two groups'
    // elements in interleaved pair positions — even pairs belong to the low-nibble group and odd pairs to the high-nibble
    // group — so each scale is applied through pmaddwd against a per-group position mask (0xFFFF for one half of the
    // window's pairs, 0x0000 for the other) before both groups' exact int32 totals are summed into this chain's block
    // accumulator. Pair products are bounded by 15 * 127 and their adjacent sum by 3810; a scale-folded dword holds one
    // such product times sc <= 63, so no lane saturates anywhere.
    private static Vector256<int> Q4KAccumulatePairDot(Vector256<int> accChain, Vector256<byte> nibbles, Vector256<short> scaleL, Vector256<short> scaleH, byte* q8)
    {
        Vector256<byte> mask = Vector256.Create((byte)0x0F); // the same nibble mask every Q4_K kernel applies, broadcast across both halves.

        Vector256<sbyte> aL = Avx.LoadVector256(q8).AsSByte(); // group 2jp's quants for this chain: signed, in [-127, 127] by the quantizer.
        Vector256<short> pairsL = Avx2.MultiplyAddAdjacent(Avx2.And(nibbles, mask), aL);

        Vector256<sbyte> aH = Avx.LoadVector256(q8 + 32).AsSByte(); // group 2jp+1's quants: the high nibbles of the same weight window.
        Vector256<short> pairsH = Avx2.MultiplyAddAdjacent(Avx2.And(Avx2.ShiftRightLogical(nibbles.AsUInt16(), 4).AsByte(), mask), aH); // srli_epi16 then byte mask: the classic high-nibble unpack.

        Vector256<int> sumj = Avx2.Add(Avx2.MultiplyAddAdjacent(pairsL, scaleL), Avx2.MultiplyAddAdjacent(pairsH, scaleH));
        return Avx2.Add(accChain, sumj); // ascending group-pair order inside the chain's block accumulator.
    }

    // The min term for one (chain, group): this group's activation quants summed exactly — adjacent pair sums in int16 lanes
    // folded through 1s into exact int32 lane totals — weighted by the group's unpacked min. Group sums stay bounded at
    // 32 * 127 and a min-weighted term at 63 * 4064, so every partial is an exact int32: this is LAIR's fold for what the
    // pinned reference precomputes as quantized-activation bsums (LAIR's Q8_K staging carries no bsums — its dots never read them).
    private static int Q4KMinTerm(byte* block, int group, byte* q8Group)
    {
        GetScaleMinK4(group, block + 4, out _, out byte m);

        Vector256<sbyte> act = Avx.LoadVector256(q8Group).AsSByte();
        Vector256<short> pairs = Avx2.MultiplyAddAdjacent(Vector256.Create((byte)1), act); // pmaddubsw against 1s: adjacent pair sums in short lanes (|pair| <= 254).

        Vector128<int> quadsLo = Sse2.MultiplyAddAdjacent(pairs.GetLower(), GetOnesShort()); // fold pairs into quad totals in exact int32 lanes.
        Vector128<int> quadsHi = Sse2.MultiplyAddAdjacent(pairs.GetUpper(), GetOnesShort());

        return m * (HSumInt32(quadsLo) + HSumInt32(quadsHi)); // fixed-order lane fold of the group's exact sum, weighted by its min.
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
        Span<short> scales = stackalloc short[8];

        for (int j = 0; j < 8; j++)
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

    private static Vector128<byte> GetOnesByte() => Vector128.Create((byte)1); // pmaddubsw's unsigned operand for a pure activation sum.

    private static Vector128<short> GetOnesShort() => Vector128.Create((short)1); // the same 1s as int16 lanes for the pair-sum fold.
}

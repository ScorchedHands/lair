namespace Lair.Kv;

/// <summary>
/// The cold-path KV cache format descriptor: one model's resolved pair of per-side formats, used for config validation,
/// footprint accounting, and pool sizing — never by an attention loop. Each side contributes its own region inside every
/// layer region of a physical block, so the pair describes both sides' addressing shape at once: the float sides store
/// plain values with no scale padding, and each block-quant side rounds its per-region value count up to whole GGML
/// 32-value groups of payload plus one half-precision scale.
/// </summary>
public readonly record struct KvFormatPair(KvSideFormat K, KvSideFormat V)
{
    /// <summary>The pinned default pair: FP32 on both sides, exactly the storage every pool in this build has always run.</summary>
    public static KvFormatPair Default { get; } = new(KvSideFormat.F32, KvSideFormat.F32);

    /// <summary>Values one block-quant group covers. Both quantized side regions round their per-region value count up to whole groups of this many values.</summary>
    public const int GroupValues = 32;

    /// <summary>True when both sides store plain FP32 floats — the only layout today's float views, attention kernels, and copy-on-write helpers address.</summary>
    public bool IsFloatStorage => K == KvSideFormat.F32 && V == KvSideFormat.F32;

    /// <summary>True for the measured-failed pair that names q4_0 on both sides. The q4-K viability spike dropped that combo: <c>q4_0</c> stays a per-side format available for asymmetric work, but a pair stating it twice is not one of the named combos and resolves to no pool.</summary>
    public bool IsDroppedQ4Pair => K == KvSideFormat.Q4_0 && V == KvSideFormat.Q4_0;

    /// <summary>Total bytes one physical block stores across every layer and both sides under this pair. For the default pair that is exactly today's formula, <c>layerCount * 2 * blockTokens * kvDim * sizeof(float)</c>; for a quantized side it is that side's whole-group payload+scale bytes summed per side.</summary>
    public long BytesPerBlock(int layerCount, int blockTokens, int kvDim)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);

        return checked((long)layerCount * (SideRegionBytes(K, blockTokens, kvDim) + SideRegionBytes(V, blockTokens, kvDim)));
    }

    /// <summary>Bytes one side's region occupies inside one layer region: values times element bytes for a float side with no scale padding, or whole-group payload+scale bytes for a quantized side.</summary>
    public long SideRegionBytes(KvSideFormat side, int blockTokens, int kvDim)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvDim);

        return side switch
        {
            KvSideFormat.F32 => checked((long)blockTokens * kvDim * sizeof(float)),
            KvSideFormat.F16 => checked((long)blockTokens * kvDim * 2),
            _ => checked(GroupsPerSideRegion(side, blockTokens, kvDim) * GroupBytes(side)),
        };
    }

    /// <summary>Whole 32-value groups one quantized side region stores its per-region values in. A float side has no groups to count and reports zero.</summary>
    public static long GroupsPerSideRegion(KvSideFormat side, int blockTokens, int kvDim)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvDim);

        if (side is KvSideFormat.F32 or KvSideFormat.F16)
        {
            return 0; // the float sides store values, not groups: no scale padding exists to round up into.
        }

        long values = checked((long)blockTokens * kvDim);
        return (values + GroupValues - 1) / GroupValues;
    }

    /// <summary>Bytes one group of a quantized side region stores: whole payload plus that group's half-precision scale. A float side has no groups and reports zero.</summary>
    public static int GroupBytes(KvSideFormat side) => side switch
    {
        KvSideFormat.F32 or KvSideFormat.F16 => 0, // neither float side stores a group: fp16 carries no scale at all.
        KvSideFormat.Q8_0 => 34, // 32 one-byte payloads + one half-precision scale
        _ => 18,                  // q4_0: 16 nibble-packed bytes covering 32 values + one half-precision scale
    };

    public override string ToString() => $"{KvSideFormatNames.Of(K)}/{KvSideFormatNames.Of(V)}";
}

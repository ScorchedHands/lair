using Lair.Contracts;

namespace Lair.Loader;

/// <summary>
/// Registry of the ggml tensor types this loader accepts; membership in this
/// registry is the acceptance rule. Each entry carries the block geometry the
/// container reader uses to compute tensor payload sizes.
///
/// Block geometry verified against
/// https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h
/// and the ggml_type_traits table in ggml/src/ggml.c:
///   F32: blck_size 1,   type_size 4   (sizeof(float))
///   F16: blck_size 1,   type_size 2   (sizeof(ggml_fp16_t))
///   Q8_0: blck_size 32,  type_size 34  (QK8_0 = 32; block_q8_0 = half d + int8 qs[32])
///   Q4_K: blck_size 256, type_size 144 (QK_K = 256, K_SCALE_SIZE = 12;
///         block_q4_K = 2*half dm + 12 scale bytes + 128 nibble bytes)
///   Q5_0: blck_size 32,  type_size 22  (QK5_0 = 32; block_q5_0 = half d + uint8 qh[4] + 16 nibble bytes)
///   Q5_K: blck_size 256, type_size 176 (block_q5_K = 2*half dm + 12 scale bytes + uint8 qh[32] + 128 nibble bytes)
///   Q6_K: blck_size 256, type_size 210 (block_q6_K = uint8 ql[128] + uint8 qh[64] + int8 scales[16] + half d)
/// Q2_K and Q3_K geometry is taken from the pinned reference tag b12032 of the same header, from which
/// those two blocks are implemented, and is the value their sizeof static_asserts enforce:
///   Q2_K: blck_size 256, type_size 84  (block_q2_K = uint8 scales[QK_K/16] + uint8 qs[QK_K/4] + 2*half dm;
///         static_assert sizeof == 2*sizeof(ggml_half) + QK_K/16 + QK_K/4)
///   Q3_K: blck_size 256, type_size 110 (block_q3_K = uint8 hmask[QK_K/8] + uint8 qs[QK_K/4] + uint8 scales[12]
///         + half d; static_assert sizeof == sizeof(ggml_half) + QK_K/4 + QK_K/8 + 12)
/// The IQ3_XXS/IQ4_NL/IQ4_XS geometry is likewise taken from the pinned reference tag b12032 of the same header,
/// from which those blocks' sizeof static_asserts are read:
///   IQ3_XXS: blck_size 256, type_size 98  (block_iq3_xxs = half d + uint8 qs[3*QK_K/8];
///           static_assert sizeof == sizeof(ggml_half) + 3*(QK_K/8))
///   IQ4_NL: blck_size 32,  type_size 18   (block_iq4_nl = half d + uint8 qs[QK4_NL/2], QK4_NL = 32;
///           static_assert sizeof == sizeof(ggml_half) + QK4_NL/2)
///   IQ4_XS: blck_size 256, type_size 136  (block_iq4_xs = half d + uint16 scales_h + uint8 scales_l[QK_K/64]
///           + uint8 qs[QK_K/2]; static_assert sizeof == sizeof(ggml_half) + sizeof(uint16_t) + QK_K/64 + QK_K/2)
/// ggml_row_size(type, ne) = type_size * ne / blck_size.
/// </summary>
internal static class GgmlTypeRegistry
{
    private static readonly Dictionary<uint, QuantDescriptor> Supported = new()
    {
        [GgmlType.F32] = new(GgmlType.F32, "F32", QuantFormat.F32, 1, 4),
        [GgmlType.F16] = new(GgmlType.F16, "F16", QuantFormat.F16, 1, 2),
        [GgmlType.Q5_0] = new(GgmlType.Q5_0, "Q5_0", QuantFormat.Q5_0, 32, 22),
        [GgmlType.Q8_0] = new(GgmlType.Q8_0, "Q8_0", QuantFormat.Q8_0, 32, 34),
        [GgmlType.Q4_K] = new(GgmlType.Q4_K, "Q4_K", QuantFormat.Q4_K, 256, 144),
        [GgmlType.Q5_K] = new(GgmlType.Q5_K, "Q5_K", QuantFormat.Q5_K, 256, 176),
        [GgmlType.Q6_K] = new(GgmlType.Q6_K, "Q6_K", QuantFormat.Q6_K, 256, 210),
        [GgmlType.Q2_K] = new(GgmlType.Q2_K, "Q2_K", QuantFormat.Q2_K, 256, 84),
        [GgmlType.Q3_K] = new(GgmlType.Q3_K, "Q3_K", QuantFormat.Q3_K, 256, 110),
        [GgmlType.IQ3_XXS] = new(GgmlType.IQ3_XXS, "IQ3_XXS", QuantFormat.IQ3_XXS, 256, 98),
        [GgmlType.IQ4_NL] = new(GgmlType.IQ4_NL, "IQ4_NL", QuantFormat.IQ4_NL, 32, 18),
        [GgmlType.IQ4_XS] = new(GgmlType.IQ4_XS, "IQ4_XS", QuantFormat.IQ4_XS, 256, 136),
    };

    public static bool TryResolve(uint id, out QuantDescriptor descriptor) =>
        Supported.TryGetValue(id, out descriptor);
}

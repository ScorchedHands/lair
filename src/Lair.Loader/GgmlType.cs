using Lair.Core.Contracts;

namespace Lair.Loader;

/// <summary>
/// ggml tensor type ids and block geometry for the GGUF container.
///
/// Format facts verified against llama.cpp master
/// (https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h,
/// enum ggml_type):
///   0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 4/5=Q4_2/Q4_3 (removed), 6=Q5_0, 7=Q5_1, 8=Q8_0,
///   9=Q8_1, 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K, 15=Q8_K, 16=IQ2_XXS,
///   17=IQ2_XS, 18=IQ3_XXS, 19=IQ1_S, 20=IQ4_NL, 21=IQ3_S, 22=IQ2_S, 23=IQ4_XS,
///   24=I8, 25=I16, 26=I32, 27=I64, 28=F64, 29=IQ1_M, 30=BF16, 34=TQ1_0, 35=TQ2_0,
///   39=MXFP4, 40=NVFP4, 41=Q1_0, 42=Q2_0, 43=F8_E4M3.
///
/// Block geometry verified against
/// https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h
/// and the ggml_type_traits table in ggml/src/ggml.c:
///   F16:  blck_size 1,   type_size 2   (sizeof(ggml_fp16_t))
///   Q8_0: blck_size 32,  type_size 34  (QK8_0 = 32; block_q8_0 = half d + int8 qs[32])
///   Q4_K: blck_size 256, type_size 144 (QK_K = 256, K_SCALE_SIZE = 12;
///         block_q4_K = 2*half dm + 12 scale bytes + 128 nibble bytes)
/// ggml_row_size(type, ne) = type_size * ne / blck_size.
/// </summary>
internal static class GgmlType
{
    public const uint F16 = 1;
    public const uint Q5_0 = 6;
    public const uint Q8_0 = 8;
    public const uint Q4_K = 12;

    public static bool TryGetSupported(uint id, out QuantFormat format, out int blockElements, out int blockBytes)
    {
        switch (id)
        {
            case F16:
                format = QuantFormat.F16;
                blockElements = 1;
                blockBytes = 2;
                return true;
            case Q8_0:
                format = QuantFormat.Q8_0;
                blockElements = 32;
                blockBytes = 34;
                return true;
            case Q4_K:
                format = QuantFormat.Q4_K;
                blockElements = 256;
                blockBytes = 144;
                return true;
            default:
                format = default;
                blockElements = 0;
                blockBytes = 0;
                return false;
        }
    }

    public static string Name(uint id)
    {
        return id switch
        {
            0 => "F32",
            F16 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            4 => "Q4_2",
            5 => "Q4_3",
            Q5_0 => "Q5_0",
            7 => "Q5_1",
            Q8_0 => "Q8_0",
            9 => "Q8_1",
            10 => "Q2_K",
            11 => "Q3_K",
            Q4_K => "Q4_K",
            13 => "Q5_K",
            14 => "Q6_K",
            15 => "Q8_K",
            16 => "IQ2_XXS",
            17 => "IQ2_XS",
            18 => "IQ3_XXS",
            19 => "IQ1_S",
            20 => "IQ4_NL",
            21 => "IQ3_S",
            22 => "IQ2_S",
            23 => "IQ4_XS",
            24 => "I8",
            25 => "I16",
            26 => "I32",
            27 => "I64",
            28 => "F64",
            29 => "IQ1_M",
            30 => "BF16",
            34 => "TQ1_0",
            35 => "TQ2_0",
            39 => "MXFP4",
            40 => "NVFP4",
            41 => "Q1_0",
            42 => "Q2_0",
            43 => "F8_E4M3",
            _ => "unknown",
        };
    }
}

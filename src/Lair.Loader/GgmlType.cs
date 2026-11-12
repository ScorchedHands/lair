namespace Lair.Loader;

/// <summary>
/// ggml tensor type ids and their reference names for the GGUF container. The
/// accepted types and their block geometry live in <see cref="GgmlTypeRegistry"/>;
/// this name table covers every known id so rejection diagnostics can name the
/// ggml type even when the loader does not support it.
///
/// Format facts verified against llama.cpp master
/// (https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h,
/// enum ggml_type):
///   0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 4/5=Q4_2/Q4_3 (removed), 6=Q5_0, 7=Q5_1, 8=Q8_0,
///   9=Q8_1, 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K, 15=Q8_K, 16=IQ2_XXS,
///   17=IQ2_XS, 18=IQ3_XXS, 19=IQ1_S, 20=IQ4_NL, 21=IQ3_S, 22=IQ2_S, 23=IQ4_XS,
///   24=I8, 25=I16, 26=I32, 27=I64, 28=F64, 29=IQ1_M, 30=BF16, 34=TQ1_0, 35=TQ2_0,
///   39=MXFP4, 40=NVFP4, 41=Q1_0, 42=Q2_0, 43=F8_E4M3.
/// </summary>
internal static class GgmlType
{
    public const uint F32 = 0;
    public const uint F16 = 1;
    public const uint Q5_0 = 6;
    public const uint Q8_0 = 8;
    public const uint Q2_K = 10;
    public const uint Q3_K = 11;
    public const uint Q4_K = 12;
    public const uint Q5_K = 13;
    public const uint Q6_K = 14;
    public const uint IQ2_XXS = 16;
    public const uint IQ2_XS = 17;
    public const uint IQ3_XXS = 18;
    public const uint IQ4_NL = 20;
    public const uint IQ3_S = 21;
    public const uint IQ2_S = 22;
    public const uint IQ4_XS = 23;
    public const uint BF16 = 30;
    public const uint Q2_0 = 42;

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
            Q2_K => "Q2_K",
            Q3_K => "Q3_K",
            Q4_K => "Q4_K",
            Q5_K => "Q5_K",
            Q6_K => "Q6_K",
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

using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lair.Backends.Cpu;

public sealed unsafe partial class CpuBackend
{
    // The shared sign table ksigns_iq2xs: a seven-bit code selects the eight-value sign byte of a grid pair. Every
    // row carries an even number of set bits, matching the reference lattice; bit i tests value i of the pair.
    private static ReadOnlySpan<byte> GetKSignsIQ2XS() => [
        0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
        144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
        160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
        48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
        192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
        80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
        96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
        240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
    ];

    // The shared sign mask kmask_iq2xs: bit i of a sign byte tests value i of the eight-value grid pair.
    private static int KMaskIQ2XS(int i) => 1 << i;

    // The shared 16-value IQ4 table: kvalues_iq4nl = -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53,
    // 69, 89, 113. Byte lane n holds the value for nibble n, so a pshufb lookup takes the nibble directly.
    private static Vector128<sbyte> GetIQ4KValues() => Vector128.Create(
        (sbyte)-127, (sbyte)-104, (sbyte)-83, (sbyte)-65, (sbyte)-49, (sbyte)-35, (sbyte)-22, (sbyte)-10,
        (sbyte)1, (sbyte)13, (sbyte)25, (sbyte)38, (sbyte)53, (sbyte)69, (sbyte)89, (sbyte)113);

    // Scalar read of one IQ4 table entry: nibble n indexes byte n of the shared table.
    private static float KValueIQ4(int nibble) => GetIQ4KValues().GetElement(nibble);

    // IQ4 chunk loader for eight elements starting at c in {0, 8, 16, 24}: element j < 16 is the low nibble of
    // qs[j] and element 16+j the high nibble of the same byte. The window always loads the full 16-byte qs region
    // (the only 16-byte read that stays inside the block — IQ4_NL's whole nibble region is exactly one vector, as
    // in LoadQ5_0Chunk) and slides the high half down for c in {8, 24}; each nibble is a code into the shared
    // 16-value IQ4 table, looked up with pshufb inside every byte lane. The looked-up int8 values widen with sign
    // (pmovsxbd) so negative entries keep their value, then land pre-scaled in element order. Both IQ4 formats use
    // the same layout; only the broadcast scale differs because an IQ4_XS scale changes every 32 values.
    private static Vector256<float> LoadIQ4Chunk(byte* qs, int c, Vector256<float> scaleV)
    {
        Vector128<byte> window = Sse2.LoadVector128(qs); // full nibble region, always inside the block
        if ((c & 8) != 0)
        {
            window = Ssse3.Shuffle(window, GetSlideHighBytesShuffle()); // qs bytes 8..15 land in lanes 0..7 for c in {8, 24}
        }

        if (c >= 16)
        {
            // High nibbles per byte lane: srli_epi16 then a 0x0F byte mask (the classic SSE2 pattern).
            window = Sse2.ShiftRightLogical(window.AsUInt16(), 4).AsByte();
        }

        Vector128<byte> masked = Sse2.And(window, GetNibbleMask()); // needed nibbles now in each byte lane's low bits

        Vector128<byte> lookedUp = Ssse3.Shuffle(GetIQ4KValues().AsByte(), masked); // every nibble looked up in its own byte lane

        Vector128<int> lo = Sse41.ConvertToVector128Int32(lookedUp.AsSByte()); // bytes 0..3 -> dword lanes (pmovsxbd, table sign kept)
        Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(lookedUp, GetUpperBytesShuffle()).AsSByte()); // bytes 4..7 -> dword lanes

        Vector256<float> kv = Avx.ConvertToVector256Single(Vector256.Create(lo, hi)); // eight table values in element order
        return Avx.Multiply(kv, scaleV); // kvalue * block or sub-block scale in element order
    }

    // Build the four float sign-bit lanes of one grid entry: pshufb broadcasts bit laneBase+i of the sign byte into
    // dword lane i, a signed compare against zero builds the all-ones test mask, and AND-ing 0x80000000 turns each set
    // test into a float sign-bit flip - exactly the reference ternary's -1.f/1.f selector.
    private static Vector128<int> SignMaskIQ2XS(byte signs, int laneBase)
    {
        Vector128<byte> bitRows = Ssse3.Shuffle(
            Vector128.Create(signs),
            Vector128.Create((byte)laneBase, (byte)(laneBase + 1), (byte)(laneBase + 2), (byte)(laneBase + 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
        Vector128<int> test = Sse2.And(
            Sse41.ConvertToVector128Int32(bitRows),
            laneBase switch
            {
                0 => Vector128.Create(1, 2, 4, 8), // the shared kmask_iq2xs row for bits 0..3
                _ => Vector128.Create(16, 32, 64, 128), // and for bits 4..7
            });

        return Sse2.And(Sse2.CompareGreaterThan(test, Vector128<int>.Zero), Vector128.Create(unchecked((int)0x80000000)));
    }
}

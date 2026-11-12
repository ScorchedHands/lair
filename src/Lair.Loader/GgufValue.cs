namespace Lair.Loader;

/// <summary>
/// GGUF KV value types. The ids match the GGUF vtype encoding:
/// 0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32, 5=INT32, 6=FLOAT32, 7=BOOL,
/// 8=STRING, 9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64.
/// </summary>
public enum GgufValueType
{
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// <summary>
/// A single GGUF metadata value. Scalars are stored in the widest field of their
/// class (integer, real, text, or element list) and read back through the typed
/// accessors, which throw <see cref="GgufException"/> when the kind does not match.
/// </summary>
public sealed class GgufValue
{
    private readonly long _integer;
    private readonly double _real;
    private readonly string? _text;
    private readonly GgufValue[]? _elements;

    public GgufValueType Type { get; }

    private GgufValue(GgufValueType type, long integer, double real, string? text, GgufValue[]? elements)
    {
        Type = type;
        _integer = integer;
        _real = real;
        _text = text;
        _elements = elements;
    }

    internal static GgufValue FromInteger(GgufValueType type, long value) =>
        new(type, value, 0d, null, null);

    internal static GgufValue FromUnsigned(GgufValueType type, ulong value) =>
        new(type, unchecked((long)value), 0d, null, null);

    internal static GgufValue FromReal(GgufValueType type, double value) =>
        new(type, 0L, value, null, null);

    internal static GgufValue FromString(string value) =>
        new(GgufValueType.String, 0L, 0d, value, null);

    internal static GgufValue FromArray(GgufValue[] values) =>
        new(GgufValueType.Array, 0L, 0d, null, values);

    public long AsLong()
    {
        if (!IsIntegerKind(Type) && Type != GgufValueType.Bool)
        {
            throw new GgufException($"GGUF value of type {Type} is not an integer");
        }

        if (Type == GgufValueType.UInt64 && _integer < 0)
        {
            throw new GgufException(
                $"GGUF UINT64 value {unchecked((ulong)_integer)} does not fit in Int64");
        }

        return _integer;
    }

    public ulong AsULong()
    {
        if (!IsIntegerKind(Type) && Type != GgufValueType.Bool)
        {
            throw new GgufException($"GGUF value of type {Type} is not an integer");
        }

        if (Type is GgufValueType.Int8 or GgufValueType.Int16 or GgufValueType.Int32 or GgufValueType.Int64
            && _integer < 0)
        {
            throw new GgufException($"GGUF value {_integer} of type {Type} is negative");
        }

        return unchecked((ulong)_integer);
    }

    public double AsDouble()
    {
        if (Type is GgufValueType.Float32 or GgufValueType.Float64)
        {
            return _real;
        }

        if (IsIntegerKind(Type) || Type == GgufValueType.Bool)
        {
            return _integer;
        }

        throw new GgufException($"GGUF value of type {Type} is not numeric");
    }

    public bool AsBool()
    {
        if (!IsIntegerKind(Type) && Type != GgufValueType.Bool)
        {
            throw new GgufException($"GGUF value of type {Type} is not a bool");
        }

        return _integer != 0;
    }

    public string AsString()
    {
        if (Type != GgufValueType.String)
        {
            throw new GgufException($"GGUF value of type {Type} is not a string");
        }

        return _text!;
    }

    public IReadOnlyList<GgufValue> AsArray()
    {
        if (Type != GgufValueType.Array)
        {
            throw new GgufException($"GGUF value of type {Type} is not an array");
        }

        return _elements!;
    }

    public int[] AsInt32Array()
    {
        IReadOnlyList<GgufValue> elements = AsArray();
        int[] values = new int[elements.Count];
        for (int i = 0; i < elements.Count; i++)
        {
            long value = elements[i].AsLong();
            if (value < int.MinValue || value > int.MaxValue)
            {
                throw new GgufException(
                    $"GGUF array element {i} value {value} does not fit in Int32");
            }

            values[i] = (int)value;
        }

        return values;
    }

    public float[] AsFloatArray()
    {
        IReadOnlyList<GgufValue> elements = AsArray();
        float[] values = new float[elements.Count];
        for (int i = 0; i < elements.Count; i++)
        {
            values[i] = (float)elements[i].AsDouble();
        }

        return values;
    }

    public override string ToString()
    {
        return Type switch
        {
            GgufValueType.String => _text!,
            GgufValueType.Array => $"[{_elements!.Length} {Type}]",
            GgufValueType.Float32 or GgufValueType.Float64 => _real.ToString(System.Globalization.CultureInfo.InvariantCulture),
            GgufValueType.UInt64 => unchecked((ulong)_integer).ToString(System.Globalization.CultureInfo.InvariantCulture),
            _ => _integer.ToString(System.Globalization.CultureInfo.InvariantCulture),
        };
    }

    private static bool IsIntegerKind(GgufValueType type) =>
        type is GgufValueType.UInt8
            or GgufValueType.Int8
            or GgufValueType.UInt16
            or GgufValueType.Int16
            or GgufValueType.UInt32
            or GgufValueType.Int32
            or GgufValueType.UInt64
            or GgufValueType.Int64;
}

using System.Globalization;
using System.Text;

namespace Lair.Templating.Minja;

/// <summary>The kind of value a <see cref="MinjaValue"/> carries.</summary>
public enum MinjaValueKind
{
    /// <summary>None, which minja also uses for undefined names, missing keys and out-of-range lookups.</summary>
    None,
    Bool,
    Int,
    Float,
    String,
    List,
    Map,
}

/// <summary>
/// A minja-compatible template value: the public currency for contexts, chat inputs, filters and macros.
/// Values are immutable snapshots; maps and lists are copied at construction so a context handed to the
/// engine cannot be mutated through it. None and undefined are the same value: missing names, missing map
/// keys and out-of-range indexes all produce <see cref="None"/>. Truthiness follows Python: none, false,
/// zero, empty string, empty list and empty map are falsey and everything else is truthy.
/// </summary>
public sealed class MinjaValue : IEquatable<MinjaValue>
{
    public static readonly MinjaValue None = new(MinjaValueKind.None, false, 0L, 0.0, null);
    public static readonly MinjaValue True = new(MinjaValueKind.Bool, true, 0L, 0.0, null);
    public static readonly MinjaValue False = new(MinjaValueKind.Bool, false, 0L, 0.0, null);
    public static readonly MinjaValue Zero = Of(0L);
    public static readonly MinjaValue EmptyString = new(MinjaValueKind.String, false, 0L, 0.0, string.Empty);
    public static readonly MinjaValue EmptyList = List([]);
    public static readonly MinjaValue EmptyMap = Map([]);

    private readonly bool _bool;
    private readonly long _int;
    private readonly double _double;
    private readonly string? _string;
    private readonly MinjaList? _list;
    private readonly MinjaMap? _map;

    private MinjaValue(MinjaValueKind kind, bool boolValue, long intValue, double doubleValue, string? stringValue, MinjaList? listValue = null, MinjaMap? mapValue = null)
    {
        Kind = kind;
        _bool = boolValue;
        _int = intValue;
        _double = doubleValue;
        _string = stringValue;
        _list = listValue;
        _map = mapValue;
    }

    public MinjaValueKind Kind { get; }

    public bool IsNone => Kind == MinjaValueKind.None;

    /// <summary>Python truthiness: falsey for none, false, zero, empty string/list/map; truthy otherwise.</summary>
    public bool IsTruthy => Kind switch
    {
        MinjaValueKind.None => false,
        MinjaValueKind.Bool => _bool,
        MinjaValueKind.Int => _int != 0,
        MinjaValueKind.Float => _double != 0.0,
        MinjaValueKind.String => _string!.Length > 0,
        MinjaValueKind.List => _list!.Count > 0,
        MinjaValueKind.Map => _map!.Count > 0,
        _ => false,
    };

    public bool IsNumber => Kind is MinjaValueKind.Int or MinjaValueKind.Float;

    /// <summary>Raw boolean payload; valid only when <see cref="Kind"/> is Bool.</summary>
    public bool BoolValue => Kind == MinjaValueKind.Bool ? _bool : throw KindMismatch(MinjaValueKind.Bool);

    /// <summary>Integer payload; valid only when <see cref="Kind"/> is Int.</summary>
    public long IntValue => Kind == MinjaValueKind.Int ? _int : throw KindMismatch(MinjaValueKind.Int);

    /// <summary>Float payload; valid only when <see cref="Kind"/> is Float.</summary>
    public double DoubleValue => Kind == MinjaValueKind.Float ? _double : throw KindMismatch(MinjaValueKind.Float);

    /// <summary>Numeric value of an Int or Float; throws for other kinds.</summary>
    public double NumberValue => Kind switch
    {
        MinjaValueKind.Int => _int,
        MinjaValueKind.Float => _double,
        _ => throw KindMismatch(MinjaValueKind.Int),
    };

    /// <summary>String payload; valid only when <see cref="Kind"/> is String.</summary>
    public string StringValue => Kind == MinjaValueKind.String ? _string! : throw KindMismatch(MinjaValueKind.String);

    /// <summary>List items in order; valid only when <see cref="Kind"/> is List.</summary>
    public IReadOnlyList<MinjaValue> Items => Kind == MinjaValueKind.List ? _list! : throw KindMismatch(MinjaValueKind.List);

    /// <summary>Map contents in insertion order; valid only when <see cref="Kind"/> is Map.</summary>
    public MinjaMap MapValue => Kind == MinjaValueKind.Map ? _map! : throw KindMismatch(MinjaValueKind.Map);

    public static MinjaValue Of(bool value) => value ? True : False;

    public static MinjaValue Of(long value) => new(MinjaValueKind.Int, false, value, 0.0, null);

    public static MinjaValue Of(double value) => new(MinjaValueKind.Float, false, 0L, value, null);

    public static MinjaValue Of(string value) => value.Length == 0 ? EmptyString : new(MinjaValueKind.String, false, 0L, 0.0, value);

    public static MinjaValue List(IEnumerable<MinjaValue> items) =>
        new(MinjaValueKind.List, false, 0L, 0.0, null, new MinjaList(items));

    public static MinjaValue Map(IEnumerable<KeyValuePair<string, MinjaValue>> entries) =>
        new(MinjaValueKind.Map, false, 0L, 0.0, null, mapValue: new MinjaMap(entries));

    /// <summary>
    /// Looks a key up in a map or a string, returning <see cref="None"/> when absent. Lists are not
    /// searched: minja treats membership in a list as an element comparison, not a key lookup.
    /// </summary>
    public bool TryLookup(string key, out MinjaValue value)
    {
        switch (Kind)
        {
            case MinjaValueKind.Map when MapValue.TryGet(key, out MinjaValue found):
                value = found;
                return true;

            case MinjaValueKind.String when StringValue.Length > 0 && StringValue.Contains(key, StringComparison.Ordinal):
                value = Of(key);
                return true;

            default:
                value = None;
                return false;
        }
    }

    /// <summary>
    /// Jinja/minja <c>str()</c> rendering: none and undefined render as empty text, booleans render as
    /// Python <c>True</c>/<c>False</c>, lists and maps render with Python repr syntax.
    /// </summary>
    public override string ToString()
    {
        switch (Kind)
        {
            case MinjaValueKind.None:
                return string.Empty;

            case MinjaValueKind.Bool:
                return _bool ? "True" : "False";

            case MinjaValueKind.Int:
                return _int.ToString(CultureInfo.InvariantCulture);

            case MinjaValueKind.Float:
                return FormatPythonFloat(_double);

            case MinjaValueKind.String:
                return _string!;

            case MinjaValueKind.List:
            {
                StringBuilder builder = new();
                builder.Append('[');

                for (int i = 0; i < _list!.Count; i++)
                {
                    if (i > 0)
                        builder.Append(", ");

                    builder.Append(_list[i].Repr());
                }

                builder.Append(']');
                return builder.ToString();
            }

            case MinjaValueKind.Map:
            {
                StringBuilder builder = new();
                builder.Append('{');
                bool first = true;

                foreach (KeyValuePair<string, MinjaValue> entry in _map!)
                {
                    if (!first)
                        builder.Append(", ");

                    first = false;
                    builder.Append('\'').Append(EscapePythonString(entry.Key)).Append('\'').Append(": ").Append(entry.Value.Repr());
                }

                builder.Append('}');
                return builder.ToString();
            }

            default:
                return string.Empty;
        }
    }

    /// <summary>Python repr used when a value appears inside a rendered list or map.</summary>
    public string Repr() => Kind == MinjaValueKind.String ? $"'{EscapePythonString(_string!)}'" : ToString();

    /// <summary>
    /// minja-compatible JSON dump (the behavior behind the tojson filter): maps serialize as objects in
    /// insertion order, strings and non-ASCII text escape like Python json.dumps with ensure_ascii,
    /// none renders as null and floats use Python repr formatting.
    /// </summary>
    public string ToJson()
    {
        StringBuilder builder = new();
        AppendJson(builder, indentUnit: null, depth: 0);
        return builder.ToString();
    }

    /// <summary>
    /// Pretty JSON dump matching Python json.dumps(indent=N): every list element and map entry sits on its own
    /// line indented N spaces deeper than its container, keys keep the <c>": "</c> separator, and empty
    /// containers still render as <c>[]</c> and <c>{}</c> on a single line. Escaping, ordering and number
    /// formatting are identical to the compact dump.
    /// </summary>
    public string ToJson(int indent)
    {
        StringBuilder builder = new();
        AppendJson(builder, new string(' ', Math.Max(0, indent)), depth: 0);
        return builder.ToString();
    }

    private void AppendJson(StringBuilder builder, string? indentUnit, int depth)
    {
        switch (Kind)
        {
            case MinjaValueKind.None:
                builder.Append("null");
                break;

            case MinjaValueKind.Bool:
                builder.Append(_bool ? "true" : "false");
                break;

            case MinjaValueKind.Int:
                builder.Append(_int.ToString(CultureInfo.InvariantCulture));
                break;

            case MinjaValueKind.Float:
                builder.Append(FormatPythonFloat(_double));
                break;

            case MinjaValueKind.String:
                AppendJsonString(builder, _string!);
                break;

            case MinjaValueKind.List:
            {
                if (_list!.Count == 0)
                {
                    builder.Append("[]");
                    break;
                }

                if (indentUnit is null)
                {
                    builder.Append('[');

                    for (int i = 0; i < _list.Count; i++)
                    {
                        if (i > 0)
                            builder.Append(", ");

                        _list[i].AppendJson(builder, indentUnit: null, depth);
                    }

                    builder.Append(']');
                    break;
                }

                builder.Append("[\n");

                for (int i = 0; i < _list.Count; i++)
                {
                    if (i > 0)
                        builder.Append(",\n");

                    AppendIndent(builder, indentUnit, depth + 1);
                    _list[i].AppendJson(builder, indentUnit, depth + 1);
                }

                builder.Append('\n');
                AppendIndent(builder, indentUnit, depth);
                builder.Append(']');
                break;
            }

            case MinjaValueKind.Map:
            {
                if (_map!.Count == 0)
                {
                    builder.Append("{}");
                    break;
                }

                if (indentUnit is null)
                {
                    builder.Append('{');
                    bool first = true;

                    foreach (KeyValuePair<string, MinjaValue> entry in _map)
                    {
                        if (!first)
                            builder.Append(", ");

                        first = false;
                        AppendJsonString(builder, entry.Key);
                        builder.Append(": ");
                        entry.Value.AppendJson(builder, indentUnit: null, depth);
                    }

                    builder.Append('}');
                    break;
                }

                builder.Append('{');
                bool prettyFirst = true;

                foreach (KeyValuePair<string, MinjaValue> entry in _map)
                {
                    if (!prettyFirst)
                        builder.Append(',');

                    prettyFirst = false;
                    builder.Append('\n');
                    AppendIndent(builder, indentUnit, depth + 1);
                    AppendJsonString(builder, entry.Key);
                    builder.Append(": ");
                    entry.Value.AppendJson(builder, indentUnit, depth + 1);
                }

                builder.Append('\n');
                AppendIndent(builder, indentUnit, depth);
                builder.Append('}');
                break;
            }
        }
    }

    private static void AppendIndent(StringBuilder builder, string indentUnit, int count)
    {
        for (int i = 0; i < count; i++)
            builder.Append(indentUnit);
    }

    /// <summary>Deep equality with Python semantics: int and float compare numerically, lists compare element-wise, maps compare entries regardless of insertion order.</summary>
    public bool Equals(MinjaValue? other)
    {
        if (other is null)
            return false;

        if (Kind == MinjaValueKind.None)
            return other.Kind == MinjaValueKind.None;

        if (other.Kind == MinjaValueKind.None)
            return false;

        switch (Kind)
        {
            case MinjaValueKind.Bool:
                return other.Kind == MinjaValueKind.Bool && _bool == other._bool;

            case MinjaValueKind.Int:
                return other.Kind switch
                {
                    MinjaValueKind.Int => _int == other._int,
                    MinjaValueKind.Float => (double)_int == other._double,
                    _ => false,
                };

            case MinjaValueKind.Float:
                return other.Kind switch
                {
                    MinjaValueKind.Float => _double == other._double,
                    MinjaValueKind.Int => _double == (double)other._int,
                    _ => false,
                };

            case MinjaValueKind.String:
                return other.Kind == MinjaValueKind.String && _string == other._string;

            case MinjaValueKind.List:
            {
                if (other.Kind != MinjaValueKind.List || _list!.Count != other._list!.Count)
                    return false;

                for (int i = 0; i < _list.Count; i++)
                {
                    if (!_list[i].Equals(other._list[i]))
                        return false;
                }

                return true;
            }

            case MinjaValueKind.Map:
            {
                if (other.Kind != MinjaValueKind.Map || _map!.Count != other._map!.Count)
                    return false;

                foreach (KeyValuePair<string, MinjaValue> entry in _map)
                {
                    if (!other._map.TryGet(entry.Key, out MinjaValue theirs) || !entry.Value.Equals(theirs))
                        return false;
                }

                return true;
            }

            default:
                return false;
        }
    }

    public override bool Equals(object? obj) => obj is MinjaValue other && Equals(other);

    public override int GetHashCode()
    {
        switch (Kind)
        {
            case MinjaValueKind.None:
                return 0;

            case MinjaValueKind.Bool:
                return _bool.GetHashCode();

            // Int and float must hash alike when numerically equal so dictionary keys keep Python equality.
            case MinjaValueKind.Int:
                return ((double)_int).GetHashCode();

            case MinjaValueKind.Float:
                return _double.GetHashCode();

            case MinjaValueKind.String:
                return _string!.GetHashCode(StringComparison.Ordinal);

            case MinjaValueKind.List:
            {
                int hash = 17;

                foreach (MinjaValue item in _list!)
                    hash = unchecked(hash * 31 + item.GetHashCode());

                return hash;
            }

            case MinjaValueKind.Map:
            {
                // Order-insensitive to match Python dict equality.
                int hash = 19;

                foreach (KeyValuePair<string, MinjaValue> entry in _map!)
                    hash = unchecked(hash * 31 + entry.Key.GetHashCode(StringComparison.Ordinal) + entry.Value.GetHashCode() * 7);

                return hash;
            }

            default:
                return 0;
        }
    }

    private InvalidOperationException KindMismatch(MinjaValueKind expected) =>
        new($"expected a {expected} value but found {Kind}");

    // ---------------------------------------------------------------- formatting helpers

    /// <summary>Python repr formatting for doubles: integral values keep a .0, others use the shortest round-tripping digits with e-style exponents.</summary>
    internal static string FormatPythonFloat(double value)
    {
        if (double.IsNaN(value))
            return "nan";

        if (double.IsPositiveInfinity(value))
            return "inf";

        if (double.IsNegativeInfinity(value))
            return "-inf";

        if (value == Math.Truncate(value) && Math.Abs(value) < 1e15)
            return value.ToString("0.0", CultureInfo.InvariantCulture);

        string shortest = value.ToString("G17", CultureInfo.InvariantCulture);

        for (int precision = 1; precision <= 17; precision++)
        {
            string candidate = value.ToString("G" + precision, CultureInfo.InvariantCulture);

            if (double.TryParse(candidate, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) && parsed == value)
            {
                shortest = candidate;
                break;
            }
        }

        int exponentMark = shortest.IndexOfAny(['E', 'e']);

        if (exponentMark < 0)
            return shortest.Contains('.') ? shortest : shortest + ".0";

        string mantissa = shortest[..exponentMark];
        int exponent = int.Parse(shortest[(exponentMark + 1)..], CultureInfo.InvariantCulture);

        // Python keeps the mantissa bare in exponent form: repr(1e-05) is '1e-05', not '1.0e-05'.
        return $"{mantissa}e{(exponent >= 0 ? "+" : "-")}{Math.Abs(exponent):D2}";
    }

    private static string EscapePythonString(string value)
    {
        StringBuilder builder = new(value.Length + 8);

        foreach (char c in value)
        {
            switch (c)
            {
                case '\\':
                    builder.Append("\\\\");
                    break;

                case '\'':
                    builder.Append("\\'");
                    break;

                case '\n':
                    builder.Append("\\n");
                    break;

                case '\r':
                    builder.Append("\\r");
                    break;

                case '\t':
                    builder.Append("\\t");
                    break;

                default:
                    if (c < 0x20 || c == 0x7F)
                        builder.Append("\\x").Append(((int)c).ToString("x2", CultureInfo.InvariantCulture));
                    else
                        builder.Append(c);

                    break;
            }
        }

        return builder.ToString();
    }

    private static void AppendJsonString(StringBuilder builder, string value)
    {
        builder.Append('"');

        foreach (char c in value)
        {
            switch (c)
            {
                case '"':
                    builder.Append("\\\"");
                    break;

                case '\\':
                    builder.Append("\\\\");
                    break;

                case '\b':
                    builder.Append("\\b");
                    break;

                case '\f':
                    builder.Append("\\f");
                    break;

                case '\n':
                    builder.Append("\\n");
                    break;

                case '\r':
                    builder.Append("\\r");
                    break;

                case '\t':
                    builder.Append("\\t");
                    break;

                default:
                    if (c < 0x20 || c > 0x7E)
                        builder.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    else
                        builder.Append(c);

                    break;
            }
        }

        builder.Append('"');
    }
}

using System.Globalization;
using System.Text;

namespace Lair.Templating.Minja;

/// <summary>
/// The built-in minja namespace shared by filter chains, tests, function calls and <c>{% filter %}</c> blocks.
/// One registry answers all three call shapes: a filter applies the builtin to the filtered value as its leading
/// argument, a test answers a predicate about the tested value (the same predicates drive select/reject), and a
/// function is called directly with no prepended value. Names bound in template scope shadow these builtins, so
/// macros keep priority over same-named built-ins; names outside this collection fail as actionable render errors
/// naming the construct and its position instead of silently succeeding.
/// </summary>
internal static class MinjaBuiltins
{
    private static readonly Dictionary<string, BuiltinFunction> Functions = BuildFunctions();

    // ---------------------------------------------------------------- call shapes

    /// <summary>Evaluates a filter application: the filtered value leads, then the evaluated arguments follow.</summary>
    public static MinjaValue ApplyFilter(FilterApplication filter, MinjaValue source, TemplateScope scope, MinjaLimits limits)
    {
        if (!Functions.TryGetValue(filter.Name, out BuiltinFunction? builtin) || builtin is null)
            throw Unknown("filter", filter.Name, filter.Offset);

        return builtin.Invoke(EvaluateArguments(filter.Arguments, scope, limits), scope, filter.Offset, input: source, limits);
    }

    /// <summary>Evaluates a test expression against the tested value and negates for 'is not'.</summary>
    public static MinjaValue ApplyTest(TestExpression test, MinjaValue left, TemplateScope scope, MinjaLimits limits)
    {
        if (!Functions.TryGetValue(test.Name, out BuiltinFunction? builtin) || builtin is null)
            throw Unknown("test", test.Name, test.Offset);

        MinjaValue result = builtin.Invoke(EvaluateArguments(test.Arguments, scope, limits), scope, test.Offset, input: left, limits);

        return MinjaValue.Of(test.Negated ? !result.IsTruthy : result.IsTruthy);
    }

    /// <summary>Evaluates a function-style call on a plain name that no template binding owns.</summary>
    public static MinjaValue InvokeFunction(string name, IReadOnlyList<CallArgument> arguments, TemplateScope scope, int offset, MinjaLimits limits)
    {
        if (!Functions.TryGetValue(name, out BuiltinFunction? builtin) || builtin is null)
            throw new MinjaRenderException("call", offset, $"unknown function '{name}'");

        return builtin.Invoke(EvaluateArguments(arguments, scope, limits), scope, offset, input: null, limits);
    }

    /// <summary>Unknown-name wording shared by filters and tests; it names the construct plus the supported set.</summary>
    public static string UnknownNameDetail(string construct, string name) =>
        $"unknown {construct} '{name}'; built-in filters include count, dictsort, equalto, e, escape, items, join, joiner, length, map, namespace, raise_exception, range, reject, rejectattr, select, selectattr, sort, tojson and trim, and tests include defined, undefined, none, boolean, integer, float, string, mapping, sequence, iterable, number, even, odd, divisibleby, equalto and in";

    // ---------------------------------------------------------------- registry

    private static Dictionary<string, BuiltinFunction> BuildFunctions()
    {
        Dictionary<string, BuiltinFunction> registry = [];

        void Add(string name, Func<BuiltinCall, MinjaValue> handler) => registry[name] = new BuiltinFunction(name, handler);

        // Collection shape and transformation.
        Add("count", Length);
        Add("length", Length);
        Add("items", Items);
        Add("dictsort", DictSort);
        Add("join", Join);
        Add("map", MapFilter);
        Add("select", Select);
        Add("reject", Reject);
        Add("selectattr", SelectAttr);
        Add("rejectattr", RejectAttr);
        Add("sort", Sort);

        // Conversion, escaping and equality.
        Add("tojson", ToJson);
        Add("trim", Trim);
        Add("e", EscapeFilter);
        Add("escape", EscapeFilter);
        Add("equalto", EqualTo);

        // Chat-template helpers and deliberate template failure.
        Add("range", RangeFunction);
        Add("namespace", NamespaceFunction);
        Add("raise_exception", RaiseException);

        // Tests used by chat templates. Each answers a predicate about the leading value, so they also serve as
        // select/reject predicates and (for the value-taking ones) as filters. Undefined equivalence follows this
        // library's single none/undefined value: an undefined name is none, so 'defined' tests for non-none.
        Add("defined", call => MinjaValue.Of(!RequireInput(call).IsNone));
        Add("undefined", call => MinjaValue.Of(RequireInput(call).IsNone));
        Add("none", call => MinjaValue.Of(RequireInput(call).IsNone));
        Add("boolean", call => KindPredicate(call, kind => kind == MinjaValueKind.Bool));
        Add("integer", call => KindPredicate(call, kind => kind == MinjaValueKind.Int));
        Add("float", call => KindPredicate(call, kind => kind == MinjaValueKind.Float));
        Add("string", call => KindPredicate(call, kind => kind == MinjaValueKind.String));
        Add("mapping", call => KindPredicate(call, kind => kind == MinjaValueKind.Map));
        Add("sequence", call => KindPredicate(call, kind => kind is MinjaValueKind.List or MinjaValueKind.String));
        Add("iterable", call => MinjaValue.Of(IsIterable(RequireInput(call))));
        Add("number", call => KindPredicate(call, kind => kind is MinjaValueKind.Int or MinjaValueKind.Float or MinjaValueKind.Bool));
        Add("even", call => IntegerParity(call, even: true));
        Add("odd", call => IntegerParity(call, even: false));
        Add("divisibleby", DivisibleBy);
        Add("in", InTest);

        return registry;
    }

    // ---------------------------------------------------------------- collection builtins

    private static MinjaValue Length(BuiltinCall call)
    {
        RejectAnyArguments(call);

        return MinjaValue.Of((long)LengthOf(RequireInput(call), call.Name, call.Offset));
    }

    /// <summary>
    /// Element count for strings (characters), lists and maps. Undefined takes the empty-sequence reading a for
    /// loop already gives it; other kinds are actionable errors rather than a silent zero.
    /// </summary>
    private static int LengthOf(MinjaValue value, string construct, int offset) => value.Kind switch
    {
        MinjaValueKind.String => value.StringValue.Length,
        MinjaValueKind.List => value.Items.Count,
        MinjaValueKind.Map => value.MapValue.Count,
        MinjaValueKind.None => 0,
        _ => throw new MinjaRenderException(construct, offset, $"cannot take the length of a {value.Kind} value"),
    };

    private static MinjaValue Items(BuiltinCall call)
    {
        RejectAnyArguments(call);

        MinjaValue input = RequireInput(call);

        if (input.IsNone)
            return MinjaValue.EmptyList;

        if (input.Kind != MinjaValueKind.Map)
            throw new MinjaRenderException("filter", call.Offset, $"items needs a map value but found {input.Kind}");

        call.Limits.RequireSequenceLength(input.MapValue.Count, "items", call.Offset);

        return PairsOf(input);
    }

    private static MinjaValue DictSort(BuiltinCall call)
    {
        MinjaValue input = RequireInput(call);

        if (input.IsNone)
            return MinjaValue.EmptyList;

        if (input.Kind != MinjaValueKind.Map)
            throw new MinjaRenderException("filter", call.Offset, $"dictsort needs a map value but found {input.Kind}");

        IReadOnlyList<EvaluatedArgument> positional = PositionalArguments(call);

        // dictsort(case_sensitive=False, by='key'): the first positional is the case flag and the second names the mode.
        bool caseSensitive = TryNamed(call, "case_sensitive", out MinjaValue flag) ? AsBool(flag, "case_sensitive", call.Offset) : positional.Count > 0 && AsBool(positional[0].Value, "case_sensitive", call.Offset);
        string by = NamedOrPositional(call, "by", positionalIndex: 1) ?? "key";

        RejectExtraPositional(call, allowed: 2);
        RejectUnknownNamedArguments(call, ["case_sensitive", "by"]);

        if (by != "key" && by != "value")
            throw new MinjaRenderException("filter", call.Offset, $"dictsort sorts by 'key' or 'value' but found '{by}'");

        call.Limits.RequireSequenceLength(input.MapValue.Count, "dictsort", call.Offset);

        List<KeyValuePair<string, MinjaValue>> entries = [.. input.MapValue];
        StringComparison comparison = caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase;

        entries.Sort((left, right) => by == "key"
            ? string.Compare(left.Key, right.Key, comparison)
            : MinjaEvaluator.CompareOrderValues(left.Value, right.Value, call.Offset));

        List<MinjaValue> pairs = [.. entries.Select(entry => PairOf(entry.Key, entry.Value))];

        return MinjaValue.List(pairs);
    }

    private static MinjaValue Join(BuiltinCall call)
    {
        List<MinjaValue> items = EnumerateItems(call);

        IReadOnlyList<EvaluatedArgument> positional = PositionalArguments(call);
        string delimiter = positional.Count > 0 ? positional[0].Value.ToString() : string.Empty;
        RejectExtraPositional(call, allowed: 1);
        RejectUnknownNamedArguments(call, ["attribute"]);

        IReadOnlyList<string>? path = AttributePath(call);

        StringBuilder builder = new();

        for (int i = 0; i < items.Count; i++)
        {
            if (i > 0)
                builder.Append(delimiter);

            MinjaValue picked = path is null ? items[i] : LookupPath(items[i], path, call.Offset, fallback: null);
            builder.Append(picked.ToString());
        }

        call.Limits.RequireOutputLength(builder.Length, "join", call.Offset);

        return MinjaValue.Of(builder.ToString());
    }

    private static MinjaValue MapFilter(BuiltinCall call)
    {
        IReadOnlyList<EvaluatedArgument> positional = PositionalArguments(call);
        IReadOnlyList<string>? path = AttributePath(call);
        string? predicateName = positional.Count > 0 ? RequireString(positional[0].Value, "map", call.Offset) : null;
        List<MinjaValue> extra = [.. positional.Skip(1).Select(argument => argument.Value)];

        MinjaValue fallback = TryNamed(call, "default", out MinjaValue providedDefault) ? providedDefault : MinjaValue.None;

        RejectUnknownNamedArguments(call, ["attribute", "default"]);

        if (path is null && predicateName is null)
            throw new MinjaRenderException("filter", call.Offset, "map needs attribute=... or a filter/test name to apply to each item");

        List<MinjaValue> mapped = [];

        foreach (MinjaValue item in EnumerateItems(call))
            mapped.Add(path is not null ? LookupPath(item, path, call.Offset, fallback) : InvokePredicate(call, item, predicateName!, extra));

        return MaterializedList(call, mapped, "map");
    }

    private static MinjaValue Select(BuiltinCall call) => FilterItems(call, "select", keep: true);

    private static MinjaValue Reject(BuiltinCall call) => FilterItems(call, "reject", keep: false);

    private static MinjaValue SelectAttr(BuiltinCall call) => FilterAttributes(call, "selectattr", keep: true);

    private static MinjaValue RejectAttr(BuiltinCall call) => FilterAttributes(call, "rejectattr", keep: false);

    private static MinjaValue Sort(BuiltinCall call)
    {
        List<MinjaValue> items = EnumerateItems(call);

        IReadOnlyList<EvaluatedArgument> positional = PositionalArguments(call);
        string? predicateName = positional.Count > 0 ? RequireString(positional[0].Value, "sort", call.Offset) : null;
        RejectExtraPositional(call, allowed: 1);

        IReadOnlyList<string>? path = AttributePath(call);
        bool reverse = NamedBool(call, "reverse", defaultValue: false);

        RejectUnknownNamedArguments(call, ["attribute", "reverse"]);

        List<(MinjaValue Key, MinjaValue Item)> keyed = [];

        foreach (MinjaValue item in items)
        {
            MinjaValue key = path is not null
                ? LookupPath(item, path, call.Offset, fallback: null)
                : predicateName is not null
                    ? InvokePredicate(call, item, predicateName, [])
                    : item;

            keyed.Add((key, item));
        }

        // Ordering runs through a stable OrderBy so equal keys keep their input order like Python's sort. The
        // sorting helper wraps comparer failures, so a key type mismatch is unwrapped back to its render error.
        List<(MinjaValue Key, MinjaValue Item)> ordered;

        try
        {
            ordered = [.. keyed.OrderBy(
                pair => pair.Key,
                Comparer<MinjaValue>.Create((left, right) =>
                {
                    int order = MinjaEvaluator.CompareOrderValues(left, right, call.Offset);

                    return reverse ? -order : order;
                }))];
        }
        catch (InvalidOperationException failure) when (failure.InnerException is MinjaRenderException render)
        {
            throw render;
        }

        return MinjaValue.List(ordered.Select(pair => pair.Item));
    }

    // ---------------------------------------------------------------- conversion and escaping

    private static MinjaValue ToJson(BuiltinCall call)
    {
        // tojson(value) dumps compactly; tojson(value, indent=N) or tojson(indent=N) pretty-prints with N-space
        // nesting like Python json.dumps. Only an indent argument is accepted so typos still fail actionably.
        IReadOnlyList<EvaluatedArgument> positional = PositionalArguments(call);
        int? indent = TryNamed(call, "indent", out MinjaValue namedIndent)
            ? (int)RequireInteger(namedIndent, "tojson", call.Offset)
            : positional.Count > 0
                ? (int)RequireInteger(positional[0].Value, "tojson", call.Offset)
                : null;

        RejectExtraPositional(call, allowed: 1);
        RejectUnknownNamedArguments(call, ["indent"]);

        // The value model owns the dump: maps keep insertion order, strings escape like Python json.dumps.
        MinjaValue input = RequireInput(call);

        string dumped = indent is int width ? input.ToJson(width) : input.ToJson();
        call.Limits.RequireOutputLength(dumped.Length, "tojson", call.Offset);

        return MinjaValue.Of(dumped);
    }

    private static MinjaValue Trim(BuiltinCall call)

    {
        IReadOnlyList<EvaluatedArgument> positional = PositionalArguments(call);

        RejectExtraPositional(call, allowed: 1);
        RejectNamedArguments(call);

        string text = RequireString(RequireInput(call), "trim", call.Offset);
        string chars = positional.Count > 0 ? RequireString(positional[0].Value, "trim", call.Offset) : " \t\n\r\f\v";

        string trimmed = text.Trim(chars.ToCharArray());
        call.Limits.RequireOutputLength(trimmed.Length, "trim", call.Offset);

        return MinjaValue.Of(trimmed);
    }


    private static MinjaValue EscapeFilter(BuiltinCall call)
    {
        RejectAnyArguments(call);

        string escaped = EscapeHtml(RequireInput(call).StringValue);
        call.Limits.RequireOutputLength(escaped.Length, "escape", call.Offset);

        return MinjaValue.Of(escaped);
    }

    /// <summary>HTML escaping for e/escape: the five markupsafe entities, everything else passes through.</summary>
    private static string EscapeHtml(string text)
    {
        StringBuilder builder = new(text.Length + 8);

        foreach (char c in text)
        {
            switch (c)
            {
                case '&':
                    builder.Append("&amp;");
                    break;

                case '<':
                    builder.Append("&lt;");
                    break;

                case '>':
                    builder.Append("&gt;");
                    break;

                case '"':
                    builder.Append("&#34;");
                    break;

                case '\'':
                    builder.Append("&#39;");
                    break;

                default:
                    builder.Append(c);
                    break;
            }
        }

        return builder.ToString();
    }

    // ---------------------------------------------------------------- equality and membership builtins

    private static MinjaValue EqualTo(BuiltinCall call)
    {
        RejectNamedArguments(call);

        IReadOnlyList<EvaluatedArgument> positional = PositionalArguments(call);

        if (positional.Count != 1)
            throw new MinjaRenderException("test", call.Offset, $"equalto compares against exactly one value but found {positional.Count}");

        return MinjaValue.Of(RequireInput(call).Equals(positional[0].Value));
    }

    private static MinjaValue InTest(BuiltinCall call)
    {
        IReadOnlyList<EvaluatedArgument> positional = PositionalArguments(call);

        if (positional.Count != 1)
            throw new MinjaRenderException("test", call.Offset, $"the in test needs exactly one container but found {positional.Count}");

        return MinjaValue.Of(Contains(positional[0].Value, RequireInput(call), call.Offset));
    }

    private static MinjaValue DivisibleBy(BuiltinCall call)
    {
        IReadOnlyList<EvaluatedArgument> positional = PositionalArguments(call);

        if (positional.Count != 1)
            throw new MinjaRenderException("test", call.Offset, $"divisibleby needs exactly one divisor but found {positional.Count}");

        double value = RequireNumber(RequireInput(call), "divisibleby", call.Offset);
        double divisor = RequireNumber(positional[0].Value, "divisibleby", call.Offset);

        if (divisor == 0)
            throw new MinjaRenderException("test", call.Offset, "division by zero");

        return MinjaValue.Of(value % divisor == 0);
    }

    // ---------------------------------------------------------------- chat-template helpers

    private static MinjaValue RangeFunction(BuiltinCall call) => Range(call.Arguments, call.Limits, call.Offset);

    /// <summary>range(stop), range(start, stop) and range(start, stop, step), including negative steps.</summary>
    internal static MinjaValue Range(IReadOnlyList<EvaluatedArgument> arguments, MinjaLimits limits, int offset)
    {
        List<MinjaValue> values = [.. arguments.Select(argument => argument.Value)];

        foreach (EvaluatedArgument argument in arguments)
        {
            if (argument.Name is not null)
                throw new MinjaRenderException("range", offset, $"range does not take a named argument '{argument.Name}'");
        }

        long start = 0;
        long stop;
        long step = 1;

        switch (values.Count)
        {
            case 1:
                stop = RequireInteger(values[0], "range", offset);
                break;

            case 2:
                start = RequireInteger(values[0], "range", offset);
                stop = RequireInteger(values[1], "range", offset);
                break;

            case 3:
                start = RequireInteger(values[0], "range", offset);
                stop = RequireInteger(values[1], "range", offset);
                step = RequireInteger(values[2], "range", offset);

                if (step == 0)
                    throw new MinjaRenderException("range", offset, "range step cannot be zero");

                break;

            default:
                throw new MinjaRenderException("range", offset, $"range takes one to three arguments but found {values.Count}");
        }

        List<MinjaValue> items = [];

        // The count a span produces is computable from its bounds, so an over-budget range is rejected before
        // the first item is allocated. The span subtraction is overflow-guarded: an overflowing span is by
        // definition over budget, so it lands on the same limit error instead of wrapping to a small count.
        long span;

        try
        {
            span = step > 0 ? checked(stop - start) : checked(start - stop);
        }
        catch (OverflowException)
        {
            span = long.MaxValue;
        }

        long stride = Math.Abs(step);
        long count = span <= 0 ? 0 : 1 + (span - 1) / stride;
        limits.RequireSequenceLength(count, "range", offset);

        if (step > 0)
        {
            for (long i = start; i < stop; i += step)
                items.Add(MinjaValue.Of(i));
        }
        else
        {
            for (long i = start; i > stop; i += step)
                items.Add(MinjaValue.Of(i));
        }

        return MinjaValue.List(items);
    }

    private static MinjaValue NamespaceFunction(BuiltinCall call) => BuildNamespace(call.Arguments, call.Offset);

    /// <summary>namespace(**init): a mutable map whose attributes dotted set statements then update.</summary>
    internal static MinjaValue BuildNamespace(IReadOnlyList<EvaluatedArgument> arguments, int offset)
    {
        List<KeyValuePair<string, MinjaValue>> entries = [];

        foreach (EvaluatedArgument argument in arguments)
        {
            if (argument.Name is null)
                throw new MinjaRenderException("namespace", offset, "namespace initial values need name=value arguments");

            entries.Add(new KeyValuePair<string, MinjaValue>(argument.Name, argument.Value));
        }

        return MinjaValue.Map(entries);
    }

    private static MinjaValue RaiseException(BuiltinCall call) => ThrowRaise(call.Arguments, call.Offset);

    /// <summary>
    /// raise_exception(message): a deliberate template failure. The message is carried verbatim so the HTTP layer
    /// can surface it as an invalid request; rendering stops instead of emitting text.
    /// </summary>
    internal static MinjaValue ThrowRaise(IReadOnlyList<EvaluatedArgument> arguments, int offset)
    {
        List<string> messages = [];

        foreach (EvaluatedArgument argument in arguments)
        {
            if (argument.Name is not null)
                throw new MinjaRenderException("raise_exception", offset, $"raise_exception does not take a named argument '{argument.Name}'");

            messages.Add(argument.Value.ToString());
        }

        string detail = messages.Count == 0 ? "raise_exception was reached with no message" : string.Join(" ", messages);

        throw new MinjaRenderException("raise_exception", offset, $"template raise_exception: {detail}");
    }

    // ---------------------------------------------------------------- stateful callables

    /// <summary>joiner(sep) is the one builtin that yields a callable instead of a value; set binds it by name.</summary>
    public static bool TryBindStateful(string calleeName, IReadOnlyList<CallArgument> arguments, TemplateScope scope, int offset, out MinjaCallable? stateful)
    {
        stateful = null;

        if (calleeName != "joiner")
            return false;

        List<string> separators = [];

        foreach (CallArgument argument in arguments)
        {
            if (argument.Name is not null)
                throw new MinjaRenderException("joiner", offset, $"joiner does not take a named argument '{argument.Name}'");

            separators.Add(MinjaEvaluator.EvaluateInScope(argument.Value, scope).ToString());
        }

        if (separators.Count > 1)
            throw new MinjaRenderException("joiner", offset, $"joiner takes at most one separator but found {separators.Count}");

        stateful = new JoinerValue(separators.Count == 0 ? ", " : separators[0]);
        return true;
    }

    /// <summary>The callable joiner produces: the first call yields empty text, every later call yields the separator.</summary>
    private sealed class JoinerValue(string separator) : MinjaCallable
    {
        private bool _first = true;

        public MinjaValue Invoke(IReadOnlyList<CallArgument> arguments, TemplateScope callSite, int offset, CallerBlockValue? caller, MinjaValue? input)
        {
            if (arguments.Count > 0)
                throw new MinjaRenderException("joiner", offset, "a joiner callable takes no arguments");

            MinjaValue result = MinjaValue.Of(_first ? string.Empty : separator);
            _first = false;

            return result;
        }
    }

    // ---------------------------------------------------------------- select/reject/map/sort plumbing

    private static List<MinjaValue> EnumerateItems(BuiltinCall call) => [.. MinjaEvaluator.EnumerateIterable(RequireInput(call), call.Limits, call.Offset)];

    /// <summary>Charges a builtin's materialized list against the render's sequence budget before it is returned.</summary>
    private static MinjaValue MaterializedList(BuiltinCall call, List<MinjaValue> items, string construct)
    {
        call.Limits.RequireSequenceLength(items.Count, construct, call.Offset);

        return MinjaValue.List(items);
    }


    private static MinjaValue FilterItems(BuiltinCall call, string construct, bool keep)
    {
        (string? predicateName, List<MinjaValue> extra) = ReadPredicate(call, construct);

        List<MinjaValue> kept = [];

        foreach (MinjaValue item in EnumerateItems(call))
            if ((predicateName is null ? item.IsTruthy : InvokePredicate(call, item, predicateName, extra).IsTruthy) == keep)
                kept.Add(item);

        return MaterializedList(call, kept, construct);
    }

    private static MinjaValue FilterAttributes(BuiltinCall call, string construct, bool keep)
    {
        IReadOnlyList<EvaluatedArgument> positional = PositionalArguments(call);

        if (positional.Count == 0)
            throw new MinjaRenderException("filter", call.Offset, $"{construct} needs an attribute path as its first argument");

        RejectNamedArguments(call);

        IReadOnlyList<string> path = SplitPath(RequireString(positional[0].Value, construct, call.Offset));
        string? predicateName = positional.Count > 1 ? RequireString(positional[1].Value, construct, call.Offset) : null;
        List<MinjaValue> extra = [.. positional.Skip(2).Select(argument => argument.Value)];

        List<MinjaValue> kept = [];

        foreach (MinjaValue item in EnumerateItems(call))
            if ((predicateName is null
                    ? LookupPath(item, path, call.Offset, fallback: null).IsTruthy
                    : InvokePredicate(call, LookupPath(item, path, call.Offset, fallback: null), predicateName, extra).IsTruthy) == keep)
                kept.Add(item);

        return MaterializedList(call, kept, construct);
    }

    private static (string? PredicateName, List<MinjaValue> Extra) ReadPredicate(BuiltinCall call, string construct)
    {
        RejectNamedArguments(call);

        IReadOnlyList<EvaluatedArgument> positional = PositionalArguments(call);

        if (positional.Count == 0)
            return (null, []);

        return (RequireString(positional[0].Value, construct, call.Offset), [.. positional.Skip(1).Select(argument => argument.Value)]);
    }

    /// <summary>Applies a test name, builtin filter or template macro to one item with the extra select/reject/map/sort arguments.</summary>
    private static MinjaValue InvokePredicate(BuiltinCall call, MinjaValue subject, string predicateName, IReadOnlyList<MinjaValue> extra)
    {
        List<EvaluatedArgument> arguments = [.. extra.Select(value => new EvaluatedArgument(Name: null, value))];

        if (Functions.TryGetValue(predicateName, out BuiltinFunction? builtin) && builtin is not null)
            return builtin.Invoke(arguments, call.Scope, call.Offset, input: subject, limits: call.Limits);

        // A macro or joiner bound in template scope works as a per-item predicate too, so user helpers stay usable.
        if (call.Scope.TryGetCallable(predicateName, out MinjaCallable? callable) && callable is not null)
            return callable.Invoke(ValueArguments(extra), call.Scope, call.Offset, caller: null, input: subject);

        throw new MinjaRenderException("filter", call.Offset, $"'{predicateName}' is neither a built-in test nor a callable in scope");
    }

    /// <summary>Rebinds already-evaluated values as call arguments so user macros receive lists and maps unchanged.</summary>
    private static IReadOnlyList<CallArgument> ValueArguments(IReadOnlyList<MinjaValue> values) =>
        [.. values.Select(value => new CallArgument(Name: null, new ValueExpression(value)))];

    // ---------------------------------------------------------------- argument helpers

    /// <summary>Evaluates parsed call arguments left to right against the call site before a builtin consumes them.</summary>
    private static List<EvaluatedArgument> EvaluateArguments(IReadOnlyList<CallArgument> arguments, TemplateScope scope, MinjaLimits limits) =>
        [.. arguments.Select(argument => new EvaluatedArgument(
            argument.Name,
            MinjaEvaluator.EvaluateInScope(argument.Value, scope, limits)))];

    /// <summary>An argument that has already been evaluated against the call site.</summary>
    internal sealed record EvaluatedArgument(string? Name, MinjaValue Value);

    private static IReadOnlyList<EvaluatedArgument> PositionalArguments(BuiltinCall call) => [.. call.Arguments.Where(argument => argument.Name is null)];

    private static MinjaValue RequireInput(BuiltinCall call) =>
        call.Input ?? throw new MinjaRenderException(call.Name, call.Offset, $"{call.Name} needs a value to act on");

    private static void RejectAnyArguments(BuiltinCall call)
    {
        if (call.Arguments.Count > 0)
            throw new MinjaRenderException(call.Name, call.Offset, $"{call.Name} takes no arguments but found {call.Arguments.Count}");
    }

    private static void RejectExtraPositional(BuiltinCall call, int allowed)
    {
        int positional = call.Arguments.Count(argument => argument.Name is null);

        if (positional > allowed)
            throw new MinjaRenderException(call.Name, call.Offset, $"{call.Name} takes at most {allowed} positional arguments but found {positional}");
    }

    private static void RejectNamedArguments(BuiltinCall call)
    {
        foreach (EvaluatedArgument argument in call.Arguments)
        {
            if (argument.Name is not null)
                throw new MinjaRenderException(call.Name, call.Offset, $"{call.Name} does not take a named argument '{argument.Name}'");
        }
    }

    /// <summary>Named arguments are only accepted where the builtin documents them, so typos fail instead of being ignored.</summary>
    private static void RejectUnknownNamedArguments(BuiltinCall call, IReadOnlyList<string> allowed)
    {
        foreach (EvaluatedArgument argument in call.Arguments)
        {
            if (argument.Name is not null && !allowed.Contains(argument.Name))
                throw new MinjaRenderException(call.Name, call.Offset, $"{call.Name} does not take a named argument '{argument.Name}'");
        }
    }

    private static bool TryNamed(BuiltinCall call, string name, out MinjaValue value)
    {
        foreach (EvaluatedArgument argument in call.Arguments)
        {
            if (argument.Name == name)
            {
                value = argument.Value;
                return true;
            }
        }

        value = MinjaValue.None;
        return false;
    }

    private static string? NamedString(BuiltinCall call, string name) => TryNamed(call, name, out MinjaValue value) ? RequireString(value, name, call.Offset) : null;

    private static bool NamedBool(BuiltinCall call, string name, bool defaultValue) => TryNamed(call, name, out MinjaValue value) ? AsBool(value, name, call.Offset) : defaultValue;

    private static bool AsBool(MinjaValue value, string name, int offset) => value.Kind == MinjaValueKind.Bool
        ? value.BoolValue
        : throw new MinjaRenderException("filter", offset, $"'{name}' needs a boolean but found {value.Kind}");

    private static IReadOnlyList<string>? AttributePath(BuiltinCall call) => NamedString(call, "attribute") is string path ? SplitPath(path) : null;

    private static string? NamedOrPositional(BuiltinCall call, string name, int positionalIndex)
    {
        if (TryNamed(call, name, out MinjaValue named))
            return RequireString(named, name, call.Offset);

        IReadOnlyList<EvaluatedArgument> positional = PositionalArguments(call);

        return positional.Count > positionalIndex ? RequireString(positional[positionalIndex].Value, name, call.Offset) : null;
    }

    private static string RequireString(MinjaValue value, string construct, int offset) => value.Kind switch
    {
        MinjaValueKind.String => value.StringValue,
        MinjaValueKind.None => throw new MinjaRenderException(construct, offset, $"'{construct}' needs a text argument but the value is undefined"),
        _ => value.ToString(),
    };

    private static double RequireNumber(MinjaValue value, string construct, int offset) => value.Kind switch
    {
        MinjaValueKind.Int or MinjaValueKind.Float => value.NumberValue,
        MinjaValueKind.Bool => value.BoolValue ? 1 : 0,
        _ => throw new MinjaRenderException(construct, offset, $"'{construct}' needs a number but found {value.Kind}"),
    };

    private static long RequireInteger(MinjaValue value, string construct, int offset) => value.Kind switch
    {
        MinjaValueKind.Int => value.IntValue,
        MinjaValueKind.Bool => value.BoolValue ? 1L : 0L,
        MinjaValueKind.Float when value.DoubleValue == Math.Truncate(value.DoubleValue) => (long)value.DoubleValue,
        _ => throw new MinjaRenderException(construct, offset, $"'{construct}' needs an integer but found {value.Kind}"),
    };

    private static bool IsIterable(MinjaValue value) =>
        value.Kind is MinjaValueKind.List or MinjaValueKind.Map or MinjaValueKind.String or MinjaValueKind.None;

    /// <summary>Membership for the in test: substring containment, deep element equality and map key presence.</summary>
    private static bool Contains(MinjaValue container, MinjaValue item, int offset) => container.Kind switch
    {
        MinjaValueKind.String => item.Kind == MinjaValueKind.String && container.StringValue.Contains(item.StringValue, StringComparison.Ordinal),
        MinjaValueKind.List => container.Items.Any(candidate => candidate.Equals(item)),
        MinjaValueKind.Map => item.Kind == MinjaValueKind.String && container.MapValue.ContainsKey(item.StringValue),
        MinjaValueKind.None => false,
        _ => throw new MinjaRenderException("test", offset, $"cannot test membership in a {container.Kind}"),
    };

    private static MinjaValue PairOf(string key, MinjaValue value) => MinjaValue.List([MinjaValue.Of(key), value]);

    private static MinjaValue PairsOf(MinjaValue map) => MinjaValue.List(map.MapValue.Select(entry => PairOf(entry.Key, entry.Value)));

    private static IReadOnlyList<string> SplitPath(string path) => [.. path.Split('.', StringSplitOptions.RemoveEmptyEntries)];

    /// <summary>
    /// Walks an attribute path over maps and lists: map segments read keys, numeric segments index lists with the
    /// evaluator's Python-style negative offsets. A missing key or out-of-range index yields none (or the supplied
    /// default), matching how plain attribute reads behave elsewhere in the evaluator.
    /// </summary>
    private static MinjaValue LookupPath(MinjaValue value, IReadOnlyList<string> path, int offset, MinjaValue? fallback)
    {
        MinjaValue current = value;

        foreach (string segment in path)
        {
            switch (current.Kind)
            {
                case MinjaValueKind.Map:
                    current = current.MapValue.Get(segment);
                    continue;

                case MinjaValueKind.List when IsIntegerSegment(segment):
                {
                    long position = long.Parse(segment, CultureInfo.InvariantCulture);
                    int length = current.Items.Count;
                    long resolved = position < 0 ? position + length : position;

                    if (resolved < 0 || resolved >= length)
                        return fallback ?? MinjaValue.None;

                    current = current.Items[(int)resolved];
                    continue;
                }

                default:
                    return fallback ?? MinjaValue.None;
            }
        }

        // A path that walks off a missing key lands on none, which the default replaces just like an early miss.
        return current.IsNone ? fallback ?? MinjaValue.None : current;
    }


    private static bool IsIntegerSegment(string segment) => long.TryParse(segment, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out _);

    /// <summary>Kind tests answer a predicate about the tested value's kind: boolean, integer, float, string, mapping and number.</summary>
    private static MinjaValue KindPredicate(BuiltinCall call, Func<MinjaValueKind, bool> matches) =>
        MinjaValue.Of(matches(RequireInput(call).Kind));

    /// <summary>even/odd apply to integers only; any other kind is false rather than a type error.</summary>
    private static MinjaValue IntegerParity(BuiltinCall call, bool even)
    {
        MinjaValue input = RequireInput(call);

        if (input.Kind != MinjaValueKind.Int)
            return MinjaValue.False;

        long remainder = input.IntValue % 2;

        // Python parity keeps the sign of the dividend, so -3 is odd and -2 is even.
        return MinjaValue.Of((remainder == 0) == even);
    }

    private static MinjaRenderException Unknown(string construct, string name, int offset) => new(construct, offset, UnknownNameDetail(construct, name));

    // ---------------------------------------------------------------- builtin plumbing

    /// <summary>One resolved built-in invocation: evaluated arguments plus the leading value a filter or test supplies.</summary>
    internal sealed class BuiltinCall(
        string name,
        IReadOnlyList<EvaluatedArgument> arguments,
        TemplateScope scope,
        int offset,
        MinjaValue? input,
        MinjaLimits limits)
    {
        /// <summary>The name the template used, which is also the name error messages quote.</summary>
        public string Name { get; } = name;

        public IReadOnlyList<EvaluatedArgument> Arguments { get; } = arguments;
        public TemplateScope Scope { get; } = scope;
        public int Offset { get; } = offset;

        /// <summary>The value the filter chain or test supplies, or null for a plain function call.</summary>
        public MinjaValue? Input { get; } = input;

        /// <summary>The guard that owns the budgets of the render this builtin runs inside.</summary>
        public MinjaLimits Limits { get; } = limits;
    }

    /// <summary>
    /// One entry of the built-in namespace. Built-ins consume arguments that are already evaluated against the
    /// call site, so they keep their own invoke shape instead of the raw-argument callable interface. The active
    /// limits guard rides along on every call so allocation-heavy builtins charge the render's budget.
    /// </summary>
    internal sealed class BuiltinFunction
    {
        private readonly Func<BuiltinCall, MinjaValue> _handler;

        public BuiltinFunction(string name, Func<BuiltinCall, MinjaValue> handler)
        {
            Name = name;
            _handler = handler;
        }

        public string Name { get; }

        public MinjaValue Invoke(IReadOnlyList<EvaluatedArgument> arguments, TemplateScope scope, int offset, MinjaValue? input, MinjaLimits limits) =>
            _handler(new BuiltinCall(Name, arguments, scope, offset, input, limits));
    }
}


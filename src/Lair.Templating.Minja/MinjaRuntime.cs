namespace Lair.Templating.Minja;

/// <summary>
/// Internal runtime plumbing for statement execution: the name scope chain, loop bookkeeping and template
/// callables. Scoping follows minja/llama.cpp chat-template behavior rather than stricter Jinja block scoping:
/// if and for bodies render in the scope they are entered in, so <c>{% set %}</c> assignments and macro
/// definitions stay visible after their block and a loop can rebind a name the loop itself iterates over.
/// A macro, call or filter invocation renders in a child of its defining scope: the body closes over the
/// defining context for reads, its parameters and locals stay private to the call, and its writes never
/// reach the caller.
/// </summary>
internal sealed class TemplateScope
{
    private readonly Dictionary<string, MinjaValue> _values = [];
    private readonly Dictionary<string, MinjaCallable> _callables = [];

    private TemplateScope(TemplateScope? parent) => Parent = parent;

    public TemplateScope? Parent { get; }

    /// <summary>
    /// Resource guard attached when a render builds this root scope, or null for a scope built without one.
    /// Child scopes leave it null and reach the guard through their parent chain, so one render charges one
    /// budget no matter how deep evaluation wanders.
    /// </summary>
    public MinjaLimits? Limits { get; private set; }

    /// <summary>Root scope holding the caller-supplied context. The context dictionary itself is never mutated.</summary>
    public static TemplateScope Root(IReadOnlyDictionary<string, MinjaValue>? context) => Root(context, limits: null);

    /// <summary>Root scope carrying the guard that owns the budgets for the render about to run.</summary>
    public static TemplateScope Root(IReadOnlyDictionary<string, MinjaValue>? context, MinjaLimits? limits)
    {
        TemplateScope scope = new(parent: null) { Limits = limits };

        if (context is not null)
        {
            foreach (KeyValuePair<string, MinjaValue> entry in context)
                scope._values[entry.Key] = entry.Value;
        }

        return scope;
    }

    public static TemplateScope Child(TemplateScope parent) => new(parent);

    /// <summary>Name lookup for value reads through the chain: innermost binding wins. Undefined names are none.</summary>
    public bool TryGetValue(string name, out MinjaValue value)
    {
        for (TemplateScope? candidate = this; candidate is not null; candidate = candidate.Parent)
        {
            if (candidate._values.TryGetValue(name, out MinjaValue? found) && found is not null)
            {
                value = found;
                return true;
            }
        }

        value = MinjaValue.None;
        return false;
    }

    /// <summary>
    /// Assignments bind in the scope they are written from. Template blocks share their enclosing scope, so a
    /// {% set %} inside an if or for stays visible after the block and can rebind even the name the loop is
    /// iterating; macro, call and filter bodies get a child scope, so their writes shadow outer bindings
    /// instead of rewriting them. A value binding also shadows a callable of the same name in that scope.
    /// </summary>
    public void SetValue(string name, MinjaValue value)
    {
        _values[name] = value;
        _callables.Remove(name);
    }

    /// <summary>
    /// Callable lookup for call sites. At each scope a callable binding wins; a value binding there shadows an
    /// outer macro, matching {% set %} overwriting a macro name. The recursive loop object is the one name that
    /// deliberately carries both bindings: attribute reads take the map value and calls take the callable.
    /// </summary>
    public bool TryGetCallable(string name, out MinjaCallable? callable)
    {
        for (TemplateScope? candidate = this; candidate is not null; candidate = candidate.Parent)
        {
            if (candidate._callables.TryGetValue(name, out MinjaCallable? found))
            {
                callable = found;
                return true;
            }

            if (candidate._values.ContainsKey(name))
            {
                callable = null;
                return false;
            }
        }

        callable = null;
        return false;
    }

    public void DefineCallable(string name, MinjaCallable callable) => _callables[name] = callable;

    /// <summary>Removes this scope's own bindings for a name; used when a loop restores its target and loop object on exit.</summary>
    public void Remove(string name)
    {
        _values.Remove(name);
        _callables.Remove(name);
    }
}

/// <summary>Control flow requested by a rendered body: ordinary completion, a break out of the innermost loop, or a continue to its next iteration.</summary>
internal enum RenderFlow
{
    None,
    Break,
    Continue,
}

/// <summary>Bookkeeping for one active for loop: the filtered item count and the current zero-based position behind the loop.* attributes.</summary>
internal sealed class LoopState(int length)
{
    public int Length { get; } = length;

    public int Index0 { get; set; }

    /// <summary>The loop object templates see as {% if loop.first %}: 1-based index/revindex, 0-based index0/revindex0, first/last and length. loop.cycle() is not implemented yet.</summary>
    public MinjaValue ToMap() => MinjaValue.Map([
        new KeyValuePair<string, MinjaValue>("index0", MinjaValue.Of((long)Index0)),
        new KeyValuePair<string, MinjaValue>("index", MinjaValue.Of((long)(Index0 + 1))),
        new KeyValuePair<string, MinjaValue>("revindex0", MinjaValue.Of((long)(Length - Index0 - 1))),
        new KeyValuePair<string, MinjaValue>("revindex", MinjaValue.Of((long)(Length - Index0))),
        new KeyValuePair<string, MinjaValue>("first", MinjaValue.Of(Index0 == 0)),
        new KeyValuePair<string, MinjaValue>("last", MinjaValue.Of(Index0 == Length - 1)),
        new KeyValuePair<string, MinjaValue>("length", MinjaValue.Of((long)Length)),
    ]);
}

/// <summary>
/// A value a template can call. Only user-defined macros, captured call bodies and recursive loop objects are
/// callable today; minja's built-in filters, tests and functions arrive with the filters item.
/// </summary>
internal interface MinjaCallable
{
    /// <summary>
    /// Invokes the callable with parsed call arguments. caller is the _caller closure captured by a {% call %}
    /// block, or null. input is the leading value a {% filter %} block applies the callable to, or null.
    /// </summary>
    MinjaValue Invoke(IReadOnlyList<CallArgument> arguments, TemplateScope callSite, int offset, CallerBlockValue? caller, MinjaValue? input);
}

/// <summary>
/// A macro defined by {% macro %}: its parameters, body and defining scope. Invocations render in a child of
/// the defining scope, so the body closes over the defining context for reads while parameters and locals stay
/// private to the call.
/// </summary>
internal sealed record MacroValue(
    string Name,
    IReadOnlyList<MacroParameter> Parameters,
    IReadOnlyList<MinjaNode> Body,
    TemplateScope DefiningScope) : MinjaCallable
{
    public MinjaValue Invoke(IReadOnlyList<CallArgument> arguments, TemplateScope callSite, int offset, CallerBlockValue? caller, MinjaValue? input) =>
        MinjaEvaluator.InvokeBody(Body, Parameters, DefiningScope, arguments, offset, caller, input, callSite);
}

/// <summary>
/// The body of a {% call %} block captured as the macro's _caller: invoking it renders the body against a child
/// of the scope the call block was written in, so the body sees the loop and set bindings around the tag.
/// </summary>
internal sealed record CallerBlockValue(IReadOnlyList<MinjaNode> Body, TemplateScope DefiningScope) : MinjaCallable
{
    public MinjaValue Invoke(IReadOnlyList<CallArgument> arguments, TemplateScope callSite, int offset, CallerBlockValue? caller, MinjaValue? input) =>
        MinjaEvaluator.InvokeBody(Body, [], DefiningScope, arguments, offset, caller: null, input, callSite);
}

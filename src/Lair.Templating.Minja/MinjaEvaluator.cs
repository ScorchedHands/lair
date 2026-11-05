using System.Globalization;
using System.Text;

namespace Lair.Templating.Minja;

/// <summary>
/// Evaluates minja expressions and renders templates, including if/elif/else, for, set, macro, call and filter
/// statements with minja-compatible scoping. Missing names, missing map keys and out-of-range indexes all
/// evaluate as none, mirroring minja's none/undefined equivalence; string and list indexes accept Python-style
/// negative offsets. Block bodies share the scope they are entered in, so <c>{% set %}</c> stays visible after
/// its block, while macro, call and filter invocations render in a child scope that closes over its defining
/// context for reads and keeps its own locals private. Filter chains, tests and function calls resolve names from
/// the built-in minja namespace in <see cref="MinjaBuiltins"/>; names bound by macros or set statements shadow
/// those built-ins, and unknown filters, tests and functions raise <see cref="MinjaRenderException"/> naming the
/// construct and its position. Inline list and map literals evaluate into value lists/maps preserving source and
/// insertion order; map literal keys must be strings in this library's string-keyed value model.
/// </summary>
public static class MinjaEvaluator
{
    /// <summary>Parses and evaluates a single expression, e.g. <c>"1 + 2 * 3"</c>.</summary>
    public static MinjaValue Evaluate(string expression, IReadOnlyDictionary<string, MinjaValue>? context = null)
    {
        ArgumentNullException.ThrowIfNull(expression);

        TemplateNode root = MinjaParser.Parse("{{ " + expression + " }}");
        OutputNode output = root.Nodes.OfType<OutputNode>().First();

        return Eval(output.Expression, TemplateScope.Root(context));
    }

    /// <summary>Evaluates an expression node from an already parsed AST.</summary>
    public static MinjaValue Evaluate(ExpressionNode expression, IReadOnlyDictionary<string, MinjaValue>? context = null) =>
        Eval(expression, TemplateScope.Root(context));

    /// <summary>Evaluates an expression node against an existing runtime scope, so callables evaluate their arguments in their invocation scope.</summary>
    internal static MinjaValue EvaluateInScope(ExpressionNode expression, TemplateScope scope) => Eval(expression, scope);

    /// <summary>
    /// Renders a template against a context. Text runs already carry the exact bytes to emit because the parser
    /// applies whitespace elision at parse time; statement bodies execute against the shared template scope.
    /// </summary>
    public static string Render(string template, IReadOnlyDictionary<string, MinjaValue>? context = null)
    {
        ArgumentNullException.ThrowIfNull(template);
        return Render(MinjaParser.Parse(template), context);
    }

    /// <summary>Renders an already parsed template.</summary>
    public static string Render(TemplateNode template, IReadOnlyDictionary<string, MinjaValue>? context = null)
    {
        ArgumentNullException.ThrowIfNull(template);

        return Render(template, TemplateScope.Root(context));
    }

    /// <summary>
    /// Renders an already parsed template against a prebuilt scope. The chat-template wrapper uses this to bind
    /// context callables (such as <c>strftime_now</c>) into the root scope before rendering; normal callers build
    /// their own root from a plain value context through the overload above.
    /// </summary>
    internal static string Render(TemplateNode template, TemplateScope scope)
    {
        ArgumentNullException.ThrowIfNull(template);

        StringBuilder builder = new();
        RenderFlow flow = RenderBody(template.Nodes, scope, builder);

        // A loop consumes its own break/continue; flow reaching the top level escaped every loop.
        if (flow != RenderFlow.None)
            throw FlowOutsideLoop(flow, -1);

        return builder.ToString();
    }

    // ---------------------------------------------------------------- statement execution

    /// <summary>
    /// Renders a body into the builder and reports control flow requested by break/continue. Bodies share the
    /// scope they are entered in, which is what keeps {% set %} inside an if or for block visible after the block.
    /// </summary>
    private static RenderFlow RenderBody(IReadOnlyList<MinjaNode> nodes, TemplateScope scope, StringBuilder builder)
    {
        foreach (MinjaNode node in nodes)
        {
            switch (node)
            {
                case TextNode text:
                    builder.Append(text.Text);
                    break;

                case CommentNode:
                    break;

                case OutputNode output:
                    builder.Append(Eval(output.Expression, scope).ToString());
                    break;

                case IfStatement statement:
                {
                    RenderFlow branchFlow = RenderIf(statement, scope, builder);

                    if (branchFlow != RenderFlow.None)
                        return branchFlow;

                    break;
                }

                case ForStatement statement:
                    RenderItems(statement, statement.Iterable, scope, builder);
                    break;

                case SetStatement statement:
                    RenderSet(statement, scope);
                    break;

                case MacroStatement statement:
                    // A macro defined inside an if or for block stays callable after the block like any other set.
                    scope.DefineCallable(statement.Name, new MacroValue(statement.Name, statement.Parameters, statement.Body, scope));
                    break;

                case CallStatement statement:
                    RenderCall(statement, scope, builder);
                    break;

                case FilterStatement statement:
                    RenderFilter(statement, scope, builder);
                    break;

                case BreakStatement:
                    return RenderFlow.Break;

                case ContinueStatement:
                    return RenderFlow.Continue;

                default:
                    throw MinjaRenderException.UnsupportedExpression(node.GetType().Name, node.Offset);
            }
        }

        return RenderFlow.None;
    }

    private static RenderFlow RenderIf(IfStatement statement, TemplateScope scope, StringBuilder builder)
    {
        foreach (IfBranch branch in statement.Branches)
        {
            if (Eval(branch.Condition, scope).IsTruthy)
                return RenderBody(branch.Body, scope, builder);
        }

        return statement.ElseBody is not null
            ? RenderBody(statement.ElseBody, scope, builder)
            : RenderFlow.None;
    }

    /// <summary>
    /// Runs a for statement over a chosen iterable expression. Recursive loops re-enter through this method with
    /// a nested iterable supplied by a loop(items) call, which is how templates flatten trees.
    /// </summary>
    private static void RenderItems(ForStatement statement, ExpressionNode iterable, TemplateScope scope, StringBuilder builder)
    {
        IReadOnlyList<MinjaValue> items = EnumerateIterable(Eval(iterable, scope), statement.Offset);

        // Loop targets and the loop object are restored when the loop ends. Jinja leaks the last loop value into
        // its frame; restoring instead keeps a loop from clobbering a same-named outer binding. This is the
        // deliberate choice here until oracle parity tests settle the question.
        (string Name, bool Had, MinjaValue Value)[] saved = CaptureBindings(TargetNames(statement.Target), scope);
        bool hadLoop = scope.TryGetValue("loop", out MinjaValue outerLoop);

        // minja binds the target while the loop condition runs, so the filter sees each candidate item.
        List<MinjaValue> filtered = [];

        foreach (MinjaValue item in items)
        {
            BindTarget(statement.Target, item, scope, statement.Offset);

            if (statement.LoopCondition is null || Eval(statement.LoopCondition, scope).IsTruthy)
                filtered.Add(item);
        }

        LoopState loop = new(filtered.Count);

        for (int i = 0; i < filtered.Count; i++)
        {
            loop.Index0 = i;
            scope.SetValue("loop", loop.ToMap());

            // A recursive for also binds the loop name as a callable so {{ loop(nested) }} renders the same body
            // over a nested iterable, with fresh loop bookkeeping for that nested pass.
            if (statement.Recursive)
                scope.DefineCallable("loop", new RecursiveLoopValue(statement, scope));

            BindTarget(statement.Target, filtered[i], scope, statement.Offset);

            if (RenderBody(statement.Body, scope, builder) == RenderFlow.Break)
                break;
        }

        if (hadLoop)
            scope.SetValue("loop", outerLoop);
        else
            scope.Remove("loop");

        RestoreBindings(saved, scope);
    }

    private static void RenderSet(SetStatement statement, TemplateScope scope)
    {
        if (statement.Targets.Count == 1)
        {
            // joiner() is the one builtin that yields a callable rather than a value, and {% set %} is where a
            // template binds it to a name; every other assignment stores an evaluated value.
            SetTarget target = statement.Targets[0];

            if (target.Attributes.Count == 1
                && statement.Value is CallExpression call
                && call.Callee is NameExpression callee
                && MinjaBuiltins.TryBindStateful(callee.Name, call.Arguments, scope, statement.Offset, out MinjaCallable? stateful)
                && stateful is not null)
            {
                scope.DefineCallable(target.Name, stateful);
                return;
            }

            AssignTarget(target, Eval(statement.Value, scope), scope, statement.Offset);
            return;
        }

        // Flat set destructuring binds the single list value positionally across the target names.
        MinjaValue value = Eval(statement.Value, scope);

        if (value.Kind != MinjaValueKind.List)
            throw new MinjaRenderException("set", statement.Offset, $"destructuring set requires a list value but found {value.Kind}");

        IReadOnlyList<MinjaValue> items = value.Items;

        if (items.Count != statement.Targets.Count)
            throw new MinjaRenderException("set", statement.Offset, $"expected {statement.Targets.Count} values to destructure but found {items.Count}");

        for (int i = 0; i < items.Count; i++)
            AssignTarget(statement.Targets[i], items[i], scope, statement.Offset);
    }

    private static void RenderCall(CallStatement statement, TemplateScope scope, StringBuilder builder)
    {
        if (!scope.TryGetCallable(CallableName(statement.Callable), out MinjaCallable? callable) || callable is null)
            throw new MinjaRenderException("call", statement.Offset, "call blocks need a macro or block callable in scope");

        CallerBlockValue caller = new(statement.Body, scope);
        MinjaValue rendered = callable.Invoke(statement.Arguments, scope, statement.Offset, caller, input: null);

        builder.Append(rendered.ToString());
    }

    private static void RenderFilter(FilterStatement statement, TemplateScope scope, StringBuilder builder)
    {
        StringBuilder body = new();
        RenderFlow bodyFlow = RenderBody(statement.Body, scope, body);

        if (bodyFlow != RenderFlow.None)
            throw FlowOutsideLoop(bodyFlow, statement.Offset);

        string text = body.ToString();

        // Jinja renders the else body when the filtered body is empty, then applies the callable to whichever text won.
        if (text.Length == 0 && statement.ElseBody is not null)
        {
            StringBuilder elseBody = new();
            RenderFlow elseFlow = RenderBody(statement.ElseBody, scope, elseBody);

            if (elseFlow != RenderFlow.None)
                throw FlowOutsideLoop(elseFlow, statement.Offset);

            text = elseBody.ToString();
        }

        if (!scope.TryGetCallable(statement.Name, out MinjaCallable? callable) || callable is null)
        {
            // A filter block applies the same built-in namespace an inline chain does, with the body text as input.
            FilterApplication application = new(statement.Name, statement.Arguments) { Offset = statement.Offset };

            builder.Append(MinjaBuiltins.ApplyFilter(application, MinjaValue.Of(text), scope).ToString());
            return;
        }

        MinjaValue result = callable.Invoke(statement.Arguments, scope, statement.Offset, caller: null, MinjaValue.Of(text));

        builder.Append(result.ToString());
    }

    /// <summary>
    /// Invokes a macro or captured block body in a child of its defining scope. Parameters and locals are defined
    /// in the child, so they shadow and never overwrite outer bindings; the child reads through the chain, which
    /// is the closure over the defining context. A macro invoked from a {% call %} block also receives the body
    /// as _caller (and as the Jinja name caller).
    /// </summary>
    internal static MinjaValue InvokeBody(
        IReadOnlyList<MinjaNode> body,
        IReadOnlyList<MacroParameter> parameters,
        TemplateScope definingScope,
        IReadOnlyList<CallArgument> arguments,
        int offset,
        CallerBlockValue? caller,
        MinjaValue? input)
    {
        TemplateScope local = TemplateScope.Child(definingScope);

        BindArguments(parameters, arguments, definingScope, local, offset, input);

        if (caller is not null)
        {
            local.DefineCallable("_caller", caller);
            local.DefineCallable("caller", caller);
        }

        StringBuilder builder = new();
        RenderFlow flow = RenderBody(body, local, builder);

        if (flow != RenderFlow.None)
            throw FlowOutsideLoop(flow, offset);

        return MinjaValue.Of(builder.ToString());
    }

    private static void BindArguments(
        IReadOnlyList<MacroParameter> parameters,
        IReadOnlyList<CallArgument> arguments,
        TemplateScope definingScope,
        TemplateScope local,
        int offset,
        MinjaValue? input)
    {
        List<CallArgument> positional = [.. arguments.Where(argument => argument.Name is null)];
        HashSet<string> bound = [];
        int nextParameter = 0;

        // A {% filter %} block passes the rendered body as the leading positional input value.
        if (input is not null)
        {
            if (parameters.Count == 0)
                throw new MinjaRenderException("filter", offset, "filter callables need at least one parameter to receive the body");

            local.SetValue(parameters[0].Name, input);
            bound.Add(parameters[0].Name);
            nextParameter = 1;
        }

        if (nextParameter + positional.Count > parameters.Count)
            throw new MinjaRenderException("call", offset, $"expected at most {parameters.Count - nextParameter} positional arguments but found {positional.Count}");

        foreach (CallArgument argument in positional)
        {
            MacroParameter parameter = parameters[nextParameter++];
            local.SetValue(parameter.Name, EvaluateInScope(argument.Value, local));
            bound.Add(parameter.Name);
        }

        foreach (CallArgument argument in arguments)
        {
            if (argument.Name is null)
                continue;

            MacroParameter? parameter = parameters.FirstOrDefault(candidate => candidate.Name == argument.Name);

            if (parameter is null)
                throw new MinjaRenderException("call", offset, $"'{argument.Name}' is not a parameter of this macro");

            if (!bound.Add(parameter.Name))
                throw new MinjaRenderException("call", offset, $"'{parameter.Name}' was passed more than once");

            local.SetValue(parameter.Name, EvaluateInScope(argument.Value, local));
        }

        foreach (MacroParameter parameter in parameters)
        {
            if (bound.Contains(parameter.Name))
                continue;

            if (parameter.DefaultValue is null)
            {
                local.SetValue(parameter.Name, MinjaValue.None);
                continue;
            }

            // Defaults evaluate against the defining scope, matching Jinja macro default semantics.
            local.SetValue(parameter.Name, EvaluateInScope(parameter.DefaultValue, definingScope));
        }
    }

    private static string CallableName(ExpressionNode callee) => callee is NameExpression name
        ? name.Name
        : throw new MinjaRenderException("call", callee.Offset, "only plain macro names are callable until built-in functions land");

    private static MinjaRenderException FlowOutsideLoop(RenderFlow flow, int offset) =>
        new(flow == RenderFlow.Break ? "break" : "continue", offset, "break and continue are only valid inside a for loop");

    // ---------------------------------------------------------------- targets and iteration

    /// <summary>Iteration order shared by for loops and the iterable built-ins: lists in order, maps by key, strings by character, undefined as empty.</summary>
    internal static IReadOnlyList<MinjaValue> EnumerateIterable(MinjaValue value, int offset) => value.Kind switch
    {
        MinjaValueKind.List => value.Items,

        // Iterating a map walks its keys in insertion order, like Jinja and minja.
        MinjaValueKind.Map => [.. value.MapValue.Keys.Select(MinjaValue.Of)],

        // Strings iterate as single-character strings, matching their indexing and slicing behavior.
        MinjaValueKind.String => [.. value.StringValue.Select(character => MinjaValue.Of(character.ToString()))],

        // Undefined iterates as an empty sequence, matching Jinja's sequence-like undefined.
        MinjaValueKind.None => [],

        _ => throw new MinjaRenderException("for", offset, $"cannot iterate a {value.Kind} value"),
    };

    private static IEnumerable<string> TargetNames(TargetPattern target) => target switch
    {
        NameTarget name => [name.Name],
        TupleTarget tuple => tuple.Items.SelectMany(TargetNames),
        _ => throw new MinjaRenderException("for", target.Offset, $"'{target.GetType().Name}' targets are not supported"),
    };

    private static (string Name, bool Had, MinjaValue Value)[] CaptureBindings(IEnumerable<string> names, TemplateScope scope) =>
        [.. names.Select(name => (name, scope.TryGetValue(name, out MinjaValue value), value))];

    private static void RestoreBindings((string Name, bool Had, MinjaValue Value)[] saved, TemplateScope scope)
    {
        foreach ((string name, bool had, MinjaValue value) in saved)
        {
            if (had)
                scope.SetValue(name, value);
            else
                scope.Remove(name);
        }
    }

    private static void BindTarget(TargetPattern target, MinjaValue item, TemplateScope scope, int offset)
    {
        switch (target)
        {
            case NameTarget name:
                scope.SetValue(name.Name, item);
                return;

            case TupleTarget tuple:
            {
                if (item.Kind != MinjaValueKind.List)
                    throw new MinjaRenderException("for", offset, $"destructuring targets require list items but found {item.Kind}");

                IReadOnlyList<MinjaValue> items = item.Items;

                if (items.Count != tuple.Items.Count)
                    throw new MinjaRenderException("for", offset, $"expected {tuple.Items.Count} values to destructure but found {items.Count}");

                for (int i = 0; i < items.Count; i++)
                    BindTarget(tuple.Items[i], items[i], scope, offset);

                return;
            }

            default:
                throw new MinjaRenderException("for", offset, $"'{target.GetType().Name}' targets are not supported");
        }
    }

    private static void AssignTarget(SetTarget target, MinjaValue value, TemplateScope scope, int offset)
    {
        if (target.Attributes.Count == 1)
        {
            scope.SetValue(target.Name, value);
            return;
        }

        // Namespaced dotted assignment walks to the owning map and mutates it in place, so every scope or value
        // holding that same map reference sees the change.
        if (!scope.TryGetValue(target.Name, out MinjaValue owner))
            throw new MinjaRenderException("set", offset, $"cannot set an attribute of undefined name '{target.Name}'");

        MinjaValue current = owner;

        for (int i = 1; i < target.Attributes.Count - 1; i++)
        {
            if (current.Kind != MinjaValueKind.Map)
                throw new MinjaRenderException("set", offset, $"cannot set an attribute on a {current.Kind} value");

            current = current.MapValue.Get(target.Attributes[i]);
        }

        if (current.Kind != MinjaValueKind.Map)
            throw new MinjaRenderException("set", offset, $"cannot set an attribute on a {current.Kind} value");

        current.MapValue.Set(target.Attributes[^1], value);
    }

    /// <summary>Callable bound as the loop name inside a recursive for: it re-enters the same for statement with a nested iterable.</summary>
    private sealed class RecursiveLoopValue(ForStatement statement, TemplateScope scope) : MinjaCallable
    {
        public MinjaValue Invoke(IReadOnlyList<CallArgument> arguments, TemplateScope callSite, int offset, CallerBlockValue? caller, MinjaValue? input)
        {
            if (arguments.Count != 1)
                throw new MinjaRenderException("loop", offset, "recursive loops take exactly one iterable per call");

            StringBuilder builder = new();
            RenderItems(statement, arguments[0].Value, scope, builder);

            return MinjaValue.Of(builder.ToString());
        }
    }

    // ---------------------------------------------------------------- expression evaluation

    private static MinjaValue Eval(ExpressionNode node, TemplateScope? scope)
    {
        switch (node)
        {
            case StringLiteral text:
                return MinjaValue.Of(text.Value);

            case IntegerLiteral number:
                return MinjaValue.Of(number.Value);

            case FloatLiteral number:
                return MinjaValue.Of(number.Value);

            case BoolLiteral flag:
                return MinjaValue.Of(flag.Value);

            case NoneLiteral:
                return MinjaValue.None;

            case ListLiteral list:
                return EvalListLiteral(list, scope);

            case MapLiteral map:
                return EvalMapLiteral(map, scope);

            case NameExpression name:
                // Undefined names are none: minja does not distinguish missing from null.
                if (scope is not null && scope.TryGetValue(name.Name, out MinjaValue value))
                    return value;

                return MinjaValue.None;

            case AttributeExpression attribute:
                return EvalAttribute(attribute, scope);

            case IndexExpression index:
                return EvalIndex(index, scope);

            case UnaryExpression unary:
                return EvalUnary(unary, scope);

            case BinaryExpression binary:
                return EvalBinary(binary, scope);

            case ComparisonExpression comparison:
                return EvalComparison(comparison, scope);

            case MembershipExpression membership:
                return EvalMembership(membership, scope);

            case ConditionalExpression conditional:
                return Eval(conditional.Condition, scope).IsTruthy
                    ? Eval(conditional.TrueBranch, scope)
                    : Eval(conditional.ElseBranch, scope);

            case CallExpression call:
                return EvalCall(call, scope);

            case FilterChainExpression chain:
                return EvalFilterChain(chain, scope);

            case TestExpression test:
                // Tests answer a predicate about the value they follow; 'is not' negates inside ApplyTest.
                if (test.Subject is null)
                    throw new MinjaRenderException("test", test.Offset, "the test has no subject expression to evaluate");

                return MinjaBuiltins.ApplyTest(test, Eval(test.Subject, scope), scope ?? TemplateScope.Root(null));

            case ValueExpression literal:
                // An already-evaluated value re-enters evaluation unchanged (used for built-in predicate arguments).
                return literal.Value;

            default:
                throw MinjaRenderException.UnsupportedExpression(node.GetType().Name, node.Offset);
        }
    }

    /// <summary>Evaluates an inline list literal into a value list preserving source order.</summary>
    private static MinjaValue EvalListLiteral(ListLiteral node, TemplateScope? scope) =>
        MinjaValue.List([.. node.Items.Select(item => Eval(item, scope))]);

    /// <summary>
    /// Evaluates an inline map literal into an ordered value map. This library's maps are string-keyed everywhere —
    /// attribute reads, key lookups and JSON dumps all work on text keys — so a non-string key expression is an
    /// actionable render error rather than a silent coercion; minja itself accepts primitive keys this model cannot
    /// hold without changing the public value type. Later entries for the same key update the value in place like
    /// MinjaMap.Set, keeping the original insertion position.
    /// </summary>
    private static MinjaValue EvalMapLiteral(MapLiteral node, TemplateScope? scope)
    {
        MinjaMap map = new();

        foreach (MapEntry entry in node.Entries)
        {
            MinjaValue key = Eval(entry.Key, scope);

            if (key.Kind != MinjaValueKind.String)
                throw new MinjaRenderException("map literal", node.Offset, $"map keys must be strings but found a {key.Kind} key");

            map.Set(key.StringValue, Eval(entry.Value, scope));
        }

        return MinjaValue.Map(map);
    }

    /// <summary>
    /// Applies a filter chain left to right. A name bound by a macro or set statement wins over the built-in
    /// namespace, which is how templates override a built-in with their own helper of the same name.
    /// </summary>
    private static MinjaValue EvalFilterChain(FilterChainExpression chain, TemplateScope? scope)
    {
        TemplateScope callSite = scope ?? TemplateScope.Root(null);
        MinjaValue value = Eval(chain.Source, scope);

        foreach (FilterApplication filter in chain.Filters)
        {
            if (callSite.TryGetCallable(filter.Name, out MinjaCallable? callable) && callable is not null)
                value = callable.Invoke(filter.Arguments, callSite, filter.Offset, caller: null, input: value);
            else
                value = MinjaBuiltins.ApplyFilter(filter, value, callSite);
        }

        return value;
    }

    /// <summary>
    /// Evaluates a call. Names bound to macros, captured block bodies or joiner callables invoke directly; plain
    /// names then resolve through the built-in function namespace, and anything else is an actionable render error.
    /// </summary>
    private static MinjaValue EvalCall(CallExpression node, TemplateScope? scope)
    {
        TemplateScope callSite = scope ?? TemplateScope.Root(null);

        if (node.Callee is NameExpression name && callSite.TryGetCallable(name.Name, out MinjaCallable? callable) && callable is not null)
            return callable.Invoke(node.Arguments, callSite, node.Offset, caller: null, input: null);

        if (node.Callee is NameExpression function && !callSite.TryGetValue(function.Name, out _))
            return MinjaBuiltins.InvokeFunction(function.Name, node.Arguments, callSite, node.Offset);

        MinjaValue callee = Eval(node.Callee, scope);

        throw new MinjaRenderException("call", node.Offset, $"cannot call a {callee.Kind} value; only macros, block callables and built-in functions are callable");
    }

    private static MinjaValue EvalAttribute(AttributeExpression node, TemplateScope? scope)
    {
        MinjaValue target = Eval(node.Target, scope);

        // Attribute access on a map reads the key; anything else (including a missing map key) is undefined, which is none.
        return target.Kind == MinjaValueKind.Map ? target.MapValue.Get(node.Name) : MinjaValue.None;
    }

    private static MinjaValue EvalIndex(IndexExpression node, TemplateScope? scope)
    {
        MinjaValue target = Eval(node.Target, scope);

        if (node.Slice)
            return EvalSlice(target, node, scope);

        MinjaValue key = Eval(node.Start!, scope);

        switch (target.Kind)
        {
            case MinjaValueKind.Map:
            {
                // Map keys are strings; a non-string key never matches, and a missing key is none.
                string lookup = key.Kind == MinjaValueKind.String ? key.StringValue : key.ToString();
                return target.MapValue.Get(lookup);
            }

            case MinjaValueKind.List:
            {
                int length = target.Items.Count;
                long position = RequireIndex(key, node);
                long resolved = position < 0 ? position + length : position;

                // Out-of-range indexes are undefined, which is none.
                return resolved < 0 || resolved >= length ? MinjaValue.None : target.Items[(int)resolved];
            }

            case MinjaValueKind.String:
            {
                int length = target.StringValue.Length;
                long position = RequireIndex(key, node);
                long resolved = position < 0 ? position + length : position;

                return resolved < 0 || resolved >= length ? MinjaValue.None : MinjaValue.Of(target.StringValue[(int)resolved].ToString());
            }

            default:
                // Indexing none (an undefined variable) is itself undefined.
                return MinjaValue.None;
        }
    }

    private static long RequireIndex(MinjaValue key, IndexExpression node)
    {
        if (key.Kind == MinjaValueKind.Int)
            return key.IntValue;

        if (key.Kind == MinjaValueKind.Bool)
            return key.BoolValue ? 1L : 0L;

        if (key.Kind == MinjaValueKind.Float && key.DoubleValue == Math.Truncate(key.DoubleValue))
            return (long)key.DoubleValue;

        throw new MinjaRenderException("index", node.Offset, $"index must be an integer but found {key.Kind}");
    }

    private static MinjaValue EvalSlice(MinjaValue target, IndexExpression node, TemplateScope? scope)
    {
        // Slicing an undefined value is itself undefined.
        if (target.Kind == MinjaValueKind.None)
            return MinjaValue.None;

        int length = target.Kind switch
        {
            MinjaValueKind.List => target.Items.Count,
            MinjaValueKind.String => target.StringValue.Length,
            MinjaValueKind.Map => throw new MinjaRenderException("slice", node.Offset, "maps cannot be sliced"),
            _ => throw new MinjaRenderException("slice", node.Offset, $"cannot slice a {target.Kind}"),
        };

        long? startRaw = node.Start is null ? null : RequireIndex(Eval(node.Start, scope), node);
        long? stopRaw = node.Stop is null ? null : RequireIndex(Eval(node.Stop, scope), node);
        long step = node.Step is null ? 1L : RequireIndex(Eval(node.Step, scope), node);

        if (step == 0)
            throw new MinjaRenderException("slice", node.Offset, "slice step cannot be zero");

        (long start, long stop) = AdjustSlice(length, startRaw, stopRaw, step);

        List<MinjaValue> picked = [];

        if (step > 0)
        {
            for (long i = start; i < stop; i += step)
                picked.Add(GetSequenceItem(target, (int)i));
        }
        else
        {
            for (long i = start; i > stop; i += step)
                picked.Add(GetSequenceItem(target, (int)i));
        }

        return target.Kind == MinjaValueKind.String
            ? MinjaValue.Of(string.Concat(picked.Select(v => v.StringValue)))
            : MinjaValue.List(picked);
    }

    private static MinjaValue GetSequenceItem(MinjaValue target, int index) =>
        target.Kind == MinjaValueKind.String ? MinjaValue.Of(target.StringValue[index].ToString()) : target.Items[index];

    /// <summary>
    /// Resolves Python slice bounds against a sequence length. Null marks an omitted bound; provided
    /// negative bounds count back from the end and out-of-range bounds clamp instead of failing.
    /// </summary>
    private static (long Start, long Stop) AdjustSlice(int length, long? startRaw, long? stopRaw, long step)
    {
        long n = length;

        if (step > 0)
        {
            long start = startRaw ?? 0;
            long stop = stopRaw ?? n;

            if (start < 0) start += n;
            if (start < 0) start = 0;
            if (start > n) start = n;

            if (stop < 0) stop += n;
            if (stop < 0) stop = 0;
            if (stop > n) stop = n;

            return (start, stop);
        }

        long backwardStart = startRaw ?? n - 1;
        long backwardStop = stopRaw ?? -1;

        // Only a provided negative bound counts back from the end; the omitted -1 sentinel stays before the start.
        if (startRaw < 0) backwardStart = startRaw.Value + n;
        if (stopRaw < 0) backwardStop = stopRaw.Value + n;

        if (backwardStart < -1) backwardStart = -1;
        if (backwardStart > n - 1) backwardStart = n - 1;

        if (backwardStop < -1) backwardStop = -1;
        if (backwardStop > n - 1) backwardStop = n - 1;

        return (backwardStart, backwardStop);
    }

    private static MinjaValue EvalUnary(UnaryExpression node, TemplateScope? scope)
    {
        MinjaValue operand = Eval(node.Operand, scope);

        if (node.Operator == UnaryOperator.Not)
            return MinjaValue.Of(!operand.IsTruthy);

        switch (operand.Kind)
        {
            case MinjaValueKind.Int:
                return MinjaValue.Of(-operand.IntValue);

            case MinjaValueKind.Float:
                return MinjaValue.Of(-operand.DoubleValue);

            case MinjaValueKind.Bool:
                return MinjaValue.Of(operand.BoolValue ? -1L : 0L);

            default:
                throw new MinjaRenderException("unary minus", node.Offset, $"cannot negate a {operand.Kind}");
        }
    }

    private static MinjaValue EvalBinary(BinaryExpression node, TemplateScope? scope)
    {
        MinjaValue left = Eval(node.Left, scope);

        // and/or short-circuit and return the operand itself, not a boolean, matching Python and minja.
        if (node.Operator == BinaryOperator.And)
            return left.IsTruthy ? Eval(node.Right, scope) : left;

        if (node.Operator == BinaryOperator.Or)
            return left.IsTruthy ? left : Eval(node.Right, scope);

        MinjaValue right = Eval(node.Right, scope);

        switch (node.Operator)
        {
            case BinaryOperator.Add:
                return Add(left, right, node.Offset);

            case BinaryOperator.Sub:
                return Numeric(node.Operator, left, right, node.Offset, (a, b) => a - b);

            case BinaryOperator.Mul:
                return Numeric(node.Operator, left, right, node.Offset, (a, b) => a * b);

            case BinaryOperator.Div:
            {
                // Python true division always yields a float.
                RequireNumbers(node.Operator, left, right, node.Offset);

                if (right.NumberValue == 0)
                    throw new MinjaRenderException("division", node.Offset, "division by zero");

                return MinjaValue.Of(left.NumberValue / right.NumberValue);
            }

            case BinaryOperator.FloorDiv:
            {
                RequireNumbers(node.Operator, left, right, node.Offset);

                if (right.NumberValue == 0)
                    throw new MinjaRenderException("floor division", node.Offset, "division by zero");

                double floored = Math.Floor(left.NumberValue / right.NumberValue);

                return BothIntegerish(left, right)
                    ? MinjaValue.Of((long)floored)
                    : MinjaValue.Of(floored);
            }

            case BinaryOperator.Mod:
            {
                RequireNumbers(node.Operator, left, right, node.Offset);

                if (right.NumberValue == 0)
                    throw new MinjaRenderException("modulo", node.Offset, "division by zero");

                double remainder = left.NumberValue % right.NumberValue;

                // Python modulo takes the sign of the divisor.
                if (remainder != 0 && (remainder < 0) != (right.NumberValue < 0))
                    remainder += right.NumberValue;

                return BothIntegerish(left, right)
                    ? MinjaValue.Of((long)remainder)
                    : MinjaValue.Of(remainder);
            }

            case BinaryOperator.Pow:
            {
                RequireNumbers(node.Operator, left, right, node.Offset);

                if (left.Kind == MinjaValueKind.Int && right.Kind == MinjaValueKind.Int && right.IntValue >= 0)
                    return MinjaValue.Of((long)Math.Pow(left.IntValue, right.IntValue));

                return MinjaValue.Of(Math.Pow(left.NumberValue, right.NumberValue));
            }

            default:
                throw MinjaRenderException.UnsupportedExpression(node.Operator.ToString(), node.Offset);
        }
    }

    private static MinjaValue Add(MinjaValue left, MinjaValue right, int offset)
    {
        if (left.Kind == MinjaValueKind.String && right.Kind == MinjaValueKind.String)
            return MinjaValue.Of(left.StringValue + right.StringValue);

        if (left.Kind == MinjaValueKind.List && right.Kind == MinjaValueKind.List)
            return MinjaValue.List(left.Items.Concat(right.Items));

        if (left.IsNumberOrBool() && right.IsNumberOrBool())
            return Numeric(BinaryOperator.Add, left, right, offset, (a, b) => a + b);

        throw new MinjaRenderException("addition", offset, $"cannot add {left.Kind} and {right.Kind}");
    }

    private static MinjaValue Numeric(BinaryOperator op, MinjaValue left, MinjaValue right, int offset, Func<double, double, double> apply)
    {
        RequireNumbers(op, left, right, offset);

        double result = apply(left.NumberValue, right.NumberValue);

        // Integer operands (bools count as ints, as in Python) keep integer results; any float operand makes the result a float.
        return BothIntegerish(left, right)
            ? MinjaValue.Of((long)result)
            : MinjaValue.Of(result);
    }

    private static bool BothIntegerish(MinjaValue left, MinjaValue right) =>
        left.Kind is MinjaValueKind.Int or MinjaValueKind.Bool && right.Kind is MinjaValueKind.Int or MinjaValueKind.Bool;

    private static void RequireNumbers(BinaryOperator op, MinjaValue left, MinjaValue right, int offset)
    {
        if (!left.IsNumberOrBool() || !right.IsNumberOrBool())
            throw new MinjaRenderException(op.ToString(), offset, $"cannot apply '{op}' to {left.Kind} and {right.Kind}");
    }

    private static MinjaValue EvalComparison(ComparisonExpression node, TemplateScope? scope)
    {
        MinjaValue left = Eval(node.Left, scope);
        MinjaValue right = Eval(node.Right, scope);

        switch (node.Operator)
        {
            case CompareOperator.Eq:
                return MinjaValue.Of(left.Equals(right));

            case CompareOperator.Ne:
                return MinjaValue.Of(!left.Equals(right));
        }

        int? order = CompareOrderValues(left, right, node.Offset);

        return node.Operator switch
        {
            CompareOperator.Lt => MinjaValue.Of(order < 0),
            CompareOperator.Lte => MinjaValue.Of(order <= 0),
            CompareOperator.Gt => MinjaValue.Of(order > 0),
            CompareOperator.Gte => MinjaValue.Of(order >= 0),
            _ => throw MinjaRenderException.UnsupportedExpression(node.Operator.ToString(), node.Offset),
        };
    }

    /// <summary>Total ordering for ordered comparisons and built-in sorting: numbers (including bools) compare numerically, strings ordinally, lists lexicographically. Anything else is a type error.</summary>
    internal static int CompareOrderValues(MinjaValue left, MinjaValue right, int offset)
    {
        if (left.IsNumberOrBool() && right.IsNumberOrBool())
            return left.NumberValue.CompareTo(right.NumberValue);

        if (left.Kind == MinjaValueKind.String && right.Kind == MinjaValueKind.String)
            return string.CompareOrdinal(left.StringValue, right.StringValue);

        if (left.Kind == MinjaValueKind.List && right.Kind == MinjaValueKind.List)
        {
            int shared = Math.Min(left.Items.Count, right.Items.Count);

            for (int i = 0; i < shared; i++)
            {
                int itemOrder = CompareOrderValues(left.Items[i], right.Items[i], offset);

                if (itemOrder != 0)
                    return itemOrder;
            }

            return left.Items.Count.CompareTo(right.Items.Count);
        }

        throw new MinjaRenderException("comparison", offset, $"cannot order-compare {left.Kind} with {right.Kind}");
    }

    private static MinjaValue EvalMembership(MembershipExpression node, TemplateScope? scope)
    {
        MinjaValue item = Eval(node.Item, scope);
        MinjaValue container = Eval(node.Container, scope);

        bool found = container.Kind switch
        {
            // String membership is substring containment.
            MinjaValueKind.String => item.Kind == MinjaValueKind.String && container.StringValue.Contains(item.StringValue, StringComparison.Ordinal),

            // List membership is deep element equality.
            MinjaValueKind.List => container.Items.Any(candidate => candidate.Equals(item)),

            // Map membership is key presence.
            MinjaValueKind.Map => item.Kind == MinjaValueKind.String && container.MapValue.ContainsKey(item.StringValue),

            // Membership in none (an undefined value) finds nothing.
            MinjaValueKind.None => false,

            _ => throw new MinjaRenderException("membership", node.Offset, $"cannot test membership in a {container.Kind}"),
        };

        return MinjaValue.Of(node.Negated ? !found : found);
    }
}

file static class MinjaValueKindExtensions
{
    public static bool IsNumberOrBool(this MinjaValue value) =>
        value.Kind is MinjaValueKind.Int or MinjaValueKind.Float or MinjaValueKind.Bool;
}

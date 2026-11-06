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
    public static MinjaValue Evaluate(string expression, IReadOnlyDictionary<string, MinjaValue>? context = null) =>
        Evaluate(expression, context, CancellationToken.None);

    /// <summary>
    /// Parses and evaluates a single expression with cooperative cancellation. The token is additive and does
    /// not widen any of the library's fixed resource limits.
    /// </summary>
    public static MinjaValue Evaluate(string expression, IReadOnlyDictionary<string, MinjaValue>? context, CancellationToken cancellation)
    {
        ArgumentNullException.ThrowIfNull(expression);

        TemplateNode root = MinjaParser.Parse("{{ " + expression + " }}", cancellation);
        OutputNode output = root.Nodes.OfType<OutputNode>().First();
        MinjaLimits limits = MinjaLimits.ForRender(cancellation);

        return Eval(output.Expression, limits, TemplateScope.Root(context, limits));
    }

    /// <summary>Evaluates an expression node from an already parsed AST.</summary>
    public static MinjaValue Evaluate(ExpressionNode expression, IReadOnlyDictionary<string, MinjaValue>? context = null) =>
        Evaluate(expression, context, CancellationToken.None);

    /// <summary>Evaluates an already parsed expression node with cooperative cancellation.</summary>
    public static MinjaValue Evaluate(ExpressionNode expression, IReadOnlyDictionary<string, MinjaValue>? context, CancellationToken cancellation)
    {
        ArgumentNullException.ThrowIfNull(expression);

        MinjaLimits limits = MinjaLimits.ForRender(cancellation);

        return Eval(expression, limits, TemplateScope.Root(context, limits));
    }

    /// <summary>Evaluates an expression node against an existing runtime scope, so callables evaluate their arguments in their invocation scope.</summary>
    internal static MinjaValue EvaluateInScope(ExpressionNode expression, TemplateScope scope) =>
        Eval(expression, MinjaLimits.ForScope(scope, CancellationToken.None), scope);

    /// <summary>Evaluates against an existing scope with the render's active guard, so built-in argument evaluation charges the same budget.</summary>
    internal static MinjaValue EvaluateInScope(ExpressionNode expression, TemplateScope scope, MinjaLimits limits) =>
        Eval(expression, limits, scope);

    /// <summary>
    /// Renders a template against a context. Text runs already carry the exact bytes to emit because the parser
    /// applies whitespace elision at parse time; statement bodies execute against the shared template scope.
    /// </summary>
    public static string Render(string template, IReadOnlyDictionary<string, MinjaValue>? context = null) =>
        Render(template, context, CancellationToken.None);

    /// <summary>Renders a template with cooperative cancellation.</summary>
    public static string Render(string template, IReadOnlyDictionary<string, MinjaValue>? context, CancellationToken cancellation)
    {
        ArgumentNullException.ThrowIfNull(template);
        return Render(MinjaParser.Parse(template, cancellation), context, cancellation);
    }

    /// <summary>Renders an already parsed template.</summary>
    public static string Render(TemplateNode template, IReadOnlyDictionary<string, MinjaValue>? context = null) =>
        Render(template, context, CancellationToken.None);

    /// <summary>Renders an already parsed template with cooperative cancellation.</summary>
    public static string Render(TemplateNode template, IReadOnlyDictionary<string, MinjaValue>? context, CancellationToken cancellation)
    {
        ArgumentNullException.ThrowIfNull(template);

        MinjaLimits limits = MinjaLimits.ForRender(cancellation);

        return Render(template, TemplateScope.Root(context, limits), limits);
    }

    /// <summary>
    /// Renders an already parsed template against a prebuilt scope. The chat-template wrapper uses this to bind
    /// context callables (such as <c>strftime_now</c>) into the root scope before rendering; normal callers build
    /// their own root from a plain value context through the overload above. The guard that owns this render's
    /// budgets is taken from the scope chain when the scope already carries one, and created here otherwise.
    /// </summary>
    internal static string Render(TemplateNode template, TemplateScope scope) =>
        Render(template, scope, CancellationToken.None);

    internal static string Render(TemplateNode template, TemplateScope scope, CancellationToken cancellation)
    {
        ArgumentNullException.ThrowIfNull(template);
        ArgumentNullException.ThrowIfNull(scope);

        return Render(template, scope, MinjaLimits.ForScope(scope, cancellation));
    }

    /// <summary>Body of every render path: one guard, one builder, one flow check at the top level.</summary>
    private static string Render(TemplateNode template, TemplateScope scope, MinjaLimits limits)
    {
        StringBuilder builder = new();
        RenderFlow flow = RenderBody(template.Nodes, limits, scope, builder);

        // A loop consumes its own break/continue; flow reaching the top level escaped every loop.
        if (flow != RenderFlow.None)
            throw FlowOutsideLoop(flow, -1);

        return builder.ToString();
    }

    // ---------------------------------------------------------------- statement execution

    /// <summary>
    /// Renders a body into the builder and reports control flow requested by break/continue. Bodies share the
    /// scope they are entered in, which is what keeps {% set %} inside an if or for block visible after the block.
    /// Every append passes through the guard so a render cannot grow past its output budget.
    /// </summary>
    private static RenderFlow RenderBody(IReadOnlyList<MinjaNode> nodes, MinjaLimits limits, TemplateScope scope, StringBuilder builder)
    {
        foreach (MinjaNode node in nodes)
        {
            limits.CheckCancellation(NodeConstruct(node), node.Offset);

            switch (node)
            {
                case TextNode text:
                    limits.Append(builder, text.Text, "template text", text.Offset);
                    break;

                case CommentNode:
                    break;

                case OutputNode output:
                    limits.Append(builder, Eval(output.Expression, limits, scope).ToString(), "output expression", output.Offset);
                    break;

                case IfStatement statement:
                {
                    RenderFlow branchFlow = RenderIf(statement, limits, scope, builder);

                    if (branchFlow != RenderFlow.None)
                        return branchFlow;

                    break;
                }

                case ForStatement statement:
                    RenderItems(statement, statement.Iterable, limits, scope, builder);
                    break;

                case SetStatement statement:
                    RenderSet(statement, limits, scope);
                    break;

                case MacroStatement statement:
                    // A macro defined inside an if or for block stays callable after the block like any other set.
                    scope.DefineCallable(statement.Name, new MacroValue(statement.Name, statement.Parameters, statement.Body, scope));
                    break;

                case CallStatement statement:
                    RenderCall(statement, limits, scope, builder);
                    break;

                case FilterStatement statement:
                    RenderFilter(statement, limits, scope, builder);
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

    private static string NodeConstruct(MinjaNode node) => node switch
    {
        TextNode => "template text",
        OutputNode => "output expression",
        IfStatement => "if",
        ForStatement => "for",
        SetStatement => "set",
        MacroStatement => "macro",
        CallStatement => "call",
        FilterStatement => "filter",
        _ => "template body",
    };

    private static RenderFlow RenderIf(IfStatement statement, MinjaLimits limits, TemplateScope scope, StringBuilder builder)
    {
        foreach (IfBranch branch in statement.Branches)
        {
            if (Eval(branch.Condition, limits, scope).IsTruthy)
                return RenderBody(branch.Body, limits, scope, builder);
        }

        return statement.ElseBody is not null
            ? RenderBody(statement.ElseBody, limits, scope, builder)
            : RenderFlow.None;
    }

    /// <summary>
    /// Runs a for statement over a chosen iterable expression. Recursive loops re-enter through this method with
    /// a nested iterable supplied by a loop(items) call, which is how templates flatten trees. The iterable is
    /// bounded before it is materialized and every body execution is charged against the render's loop budget,
    /// so an unbounded loop fails as an actionable limit error instead of running until the process dies.
    /// </summary>
    private static void RenderItems(ForStatement statement, ExpressionNode iterable, MinjaLimits limits, TemplateScope scope, StringBuilder builder)
    {
        IReadOnlyList<MinjaValue> items = EnumerateIterable(Eval(iterable, limits, scope), limits, statement.Offset);

        // Loop targets and the loop object are restored when the loop ends. Jinja leaks the last loop value into
        // its frame; restoring instead keeps a loop from clobbering a same-named outer binding. This is the
        // deliberate choice here until oracle parity tests settle the question.
        (string Name, bool Had, MinjaValue Value)[] saved = CaptureBindings(TargetNames(statement.Target), scope);
        bool hadLoop = scope.TryGetValue("loop", out MinjaValue outerLoop);

        // minja binds the target while the loop condition runs, so the filter sees each candidate item.
        List<MinjaValue> filtered = [];

        foreach (MinjaValue item in items)
        {
            limits.CheckCancellation("for", statement.Offset);

            BindTarget(statement.Target, item, scope, statement.Offset);

            if (statement.LoopCondition is null || Eval(statement.LoopCondition, limits, scope).IsTruthy)
                filtered.Add(item);
        }

        LoopState loop = new(filtered.Count);

        for (int i = 0; i < filtered.Count; i++)
        {
            limits.ClaimLoopIteration("for", statement.Offset);

            loop.Index0 = i;
            scope.SetValue("loop", loop.ToMap());

            // A recursive for also binds the loop name as a callable so {{ loop(nested) }} renders the same body
            // over a nested iterable, with fresh loop bookkeeping for that nested pass.
            if (statement.Recursive)
                scope.DefineCallable("loop", new RecursiveLoopValue(statement, scope));

            BindTarget(statement.Target, filtered[i], scope, statement.Offset);

            if (RenderBody(statement.Body, limits, scope, builder) == RenderFlow.Break)
                break;
        }

        if (hadLoop)
            scope.SetValue("loop", outerLoop);
        else
            scope.Remove("loop");

        RestoreBindings(saved, scope);
    }

    private static void RenderSet(SetStatement statement, MinjaLimits limits, TemplateScope scope)
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

            AssignTarget(target, Eval(statement.Value, limits, scope), scope, statement.Offset);
            return;
        }

        // Flat set destructuring binds the single list value positionally across the target names.
        MinjaValue value = Eval(statement.Value, limits, scope);

        if (value.Kind != MinjaValueKind.List)
            throw new MinjaRenderException("set", statement.Offset, $"destructuring set requires a list value but found {value.Kind}");

        IReadOnlyList<MinjaValue> items = value.Items;

        if (items.Count != statement.Targets.Count)
            throw new MinjaRenderException("set", statement.Offset, $"expected {statement.Targets.Count} values to destructure but found {items.Count}");

        for (int i = 0; i < items.Count; i++)
            AssignTarget(statement.Targets[i], items[i], scope, statement.Offset);
    }

    private static void RenderCall(CallStatement statement, MinjaLimits limits, TemplateScope scope, StringBuilder builder)
    {
        if (!scope.TryGetCallable(CallableName(statement.Callable), out MinjaCallable? callable) || callable is null)
            throw new MinjaRenderException("call", statement.Offset, "call blocks need a macro or block callable in scope");

        limits.CheckCancellation("call", statement.Offset);

        CallerBlockValue caller = new(statement.Body, scope);
        MinjaValue rendered = callable.Invoke(statement.Arguments, scope, statement.Offset, caller, input: null);

        limits.Append(builder, rendered.ToString(), "call", statement.Offset);
    }

    private static void RenderFilter(FilterStatement statement, MinjaLimits limits, TemplateScope scope, StringBuilder builder)
    {
        limits.CheckCancellation("filter", statement.Offset);

        StringBuilder body = new();
        RenderFlow bodyFlow = RenderBody(statement.Body, limits, scope, body);

        if (bodyFlow != RenderFlow.None)
            throw FlowOutsideLoop(bodyFlow, statement.Offset);

        string text = body.ToString();

        // Jinja renders the else body when the filtered body is empty, then applies the callable to whichever text won.
        if (text.Length == 0 && statement.ElseBody is not null)
        {
            StringBuilder elseBody = new();
            RenderFlow elseFlow = RenderBody(statement.ElseBody, limits, scope, elseBody);

            if (elseFlow != RenderFlow.None)
                throw FlowOutsideLoop(elseFlow, statement.Offset);

            text = elseBody.ToString();
        }

        if (!scope.TryGetCallable(statement.Name, out MinjaCallable? callable) || callable is null)
        {
            // A filter block applies the same built-in namespace an inline chain does, with the body text as input.
            FilterApplication application = new(statement.Name, statement.Arguments) { Offset = statement.Offset };

            limits.Append(builder, MinjaBuiltins.ApplyFilter(application, MinjaValue.Of(text), scope, limits).ToString(), "filter", statement.Offset);
            return;
        }

        MinjaValue result = callable.Invoke(statement.Arguments, scope, statement.Offset, caller: null, MinjaValue.Of(text));

        limits.Append(builder, result.ToString(), "filter", statement.Offset);
    }

    /// <summary>
    /// Invokes a macro or captured block body in a child of its defining scope. Parameters and locals are defined
    /// in the child, so they shadow and never overwrite outer bindings; the child reads through the chain, which
    /// is the closure over the defining context. Call arguments are evaluated in the invocation scope, matching
    /// the pinned llama.cpp jinja engine, which gathers every call_expression argument from the caller's context
    /// before invoking the callee. A macro invoked from a {% call %} block also receives the body as _caller (and
    /// as the Jinja name caller).
    /// </summary>
    internal static MinjaValue InvokeBody(
        IReadOnlyList<MinjaNode> body,
        IReadOnlyList<MacroParameter> parameters,
        TemplateScope definingScope,
        IReadOnlyList<CallArgument> arguments,
        int offset,
        CallerBlockValue? caller,
        MinjaValue? input,
        TemplateScope callSite)
    {
        MinjaLimits limits = MinjaLimits.ForScope(callSite, CancellationToken.None);

        limits.EnterCall(offset);

        try
        {
            TemplateScope local = TemplateScope.Child(definingScope);

            BindArguments(parameters, arguments, callSite, local, limits, offset, input);

            if (caller is not null)
            {
                local.DefineCallable("_caller", caller);
                local.DefineCallable("caller", caller);
            }

            StringBuilder builder = new();
            RenderFlow flow = RenderBody(body, limits, local, builder);

            if (flow != RenderFlow.None)
                throw FlowOutsideLoop(flow, offset);

            limits.RequireOutputLength(builder.Length, "macro body", offset);

            return MinjaValue.Of(builder.ToString());
        }
        finally
        {
            limits.ExitCall();
        }
    }

    private static void BindArguments(
        IReadOnlyList<MacroParameter> parameters,
        IReadOnlyList<CallArgument> arguments,
        TemplateScope callSite,
        TemplateScope local,
        MinjaLimits limits,
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
            local.SetValue(parameter.Name, Eval(argument.Value, limits, callSite));
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

            local.SetValue(parameter.Name, Eval(argument.Value, limits, callSite));
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

            // Defaults evaluate against the invocation scope, matching the pinned llama.cpp jinja engine,
            // which executes a default argument expression in the call expression's context.
            local.SetValue(parameter.Name, Eval(parameter.DefaultValue, limits, callSite));
        }
    }

    private static string CallableName(ExpressionNode callee) => callee is NameExpression name
        ? name.Name
        : throw new MinjaRenderException("call", callee.Offset, "only plain macro names are callable until built-in functions land");

    private static MinjaRenderException FlowOutsideLoop(RenderFlow flow, int offset) =>
        new(flow == RenderFlow.Break ? "break" : "continue", offset, "break and continue are only valid inside a for loop");

    // ---------------------------------------------------------------- targets and iteration

    /// <summary>
    /// Iteration order shared by for loops and the iterable built-ins: lists in order, maps by key, strings by
    /// character, undefined as empty. Maps and strings are expanded into item sequences here, so their length is
    /// checked against the sequence budget before any list is allocated.
    /// </summary>
    internal static IReadOnlyList<MinjaValue> EnumerateIterable(MinjaValue value, MinjaLimits limits, int offset) => value.Kind switch
    {
        MinjaValueKind.List => value.Items,

        // Iterating a map walks its keys in insertion order, like Jinja and minja.
        MinjaValueKind.Map => MapKeys(value, limits, offset),

        // Strings iterate as single-character strings, matching their indexing and slicing behavior.
        MinjaValueKind.String => StringCharacters(value, limits, offset),

        // Undefined iterates as an empty sequence, matching Jinja's sequence-like undefined.
        MinjaValueKind.None => [],

        _ => throw new MinjaRenderException("for", offset, $"cannot iterate a {value.Kind} value"),
    };

    private static IReadOnlyList<MinjaValue> MapKeys(MinjaValue value, MinjaLimits limits, int offset)
    {
        limits.RequireSequenceLength(value.MapValue.Count, "iterating a map", offset);

        return [.. value.MapValue.Keys.Select(MinjaValue.Of)];
    }

    private static IReadOnlyList<MinjaValue> StringCharacters(MinjaValue value, MinjaLimits limits, int offset)
    {
        limits.RequireSequenceLength(value.StringValue.Length, "iterating a string", offset);

        return [.. value.StringValue.Select(character => MinjaValue.Of(character.ToString()))];
    }

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

            MinjaLimits limits = MinjaLimits.ForScope(scope, CancellationToken.None);

            limits.EnterCall(offset);

            try
            {
                StringBuilder builder = new();
                RenderItems(statement, arguments[0].Value, limits, scope, builder);

                limits.RequireOutputLength(builder.Length, "recursive loop", offset);

                return MinjaValue.Of(builder.ToString());
            }
            finally
            {
                limits.ExitCall();
            }
        }
    }

    // ---------------------------------------------------------------- expression evaluation

    private static MinjaValue Eval(ExpressionNode node, MinjaLimits limits, TemplateScope? scope)
    {
        limits.EnterExpression(node.Offset);

        try
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
                    return EvalListLiteral(list, limits, scope);

                case MapLiteral map:
                    return EvalMapLiteral(map, limits, scope);

                case NameExpression name:
                    // Undefined names are none: minja does not distinguish missing from null.
                    if (scope is not null && scope.TryGetValue(name.Name, out MinjaValue value))
                        return value;

                    return MinjaValue.None;

                case AttributeExpression attribute:
                    return EvalAttribute(attribute, limits, scope);

                case IndexExpression index:
                    return EvalIndex(index, limits, scope);

                case UnaryExpression unary:
                    return EvalUnary(unary, limits, scope);

                case BinaryExpression binary:
                    return EvalBinary(binary, limits, scope);

                case ComparisonExpression comparison:
                    return EvalComparison(comparison, limits, scope);

                case MembershipExpression membership:
                    return EvalMembership(membership, limits, scope);

                case ConditionalExpression conditional:
                    return Eval(conditional.Condition, limits, scope).IsTruthy
                        ? Eval(conditional.TrueBranch, limits, scope)
                        : Eval(conditional.ElseBranch, limits, scope);

                case CallExpression call:
                    return EvalCall(call, limits, scope);

                case FilterChainExpression chain:
                    return EvalFilterChain(chain, limits, scope);

                case TestExpression test:
                    // Tests answer a predicate about the value they follow; 'is not' negates inside ApplyTest.
                    if (test.Subject is null)
                        throw new MinjaRenderException("test", test.Offset, "the test has no subject expression to evaluate");

                    return MinjaBuiltins.ApplyTest(test, Eval(test.Subject, limits, scope), scope ?? TemplateScope.Root(null, limits), limits);

                case ValueExpression literal:
                    // An already-evaluated value re-enters evaluation unchanged (used for built-in predicate arguments).
                    return literal.Value;

                default:
                    throw MinjaRenderException.UnsupportedExpression(node.GetType().Name, node.Offset);
            }
        }
        finally
        {
            limits.ExitExpression();
        }
    }

    /// <summary>
    /// Evaluates an inline list literal into a value list preserving source order. The literal's item count is
    /// checked before evaluation so a template packed with literals fails as a sequence limit instead of
    /// allocating the list first.
    /// </summary>
    private static MinjaValue EvalListLiteral(ListLiteral node, MinjaLimits limits, TemplateScope? scope)
    {
        limits.RequireSequenceLength(node.Items.Count, "list literal", node.Offset);

        return MinjaValue.List([.. node.Items.Select(item => Eval(item, limits, scope))]);
    }

    /// <summary>
    /// Evaluates an inline map literal into an ordered value map. This library's maps are string-keyed everywhere —
    /// attribute reads, key lookups and JSON dumps all work on text keys — so a non-string key expression is an
    /// actionable render error rather than a silent coercion; minja itself accepts primitive keys this model cannot
    /// hold without changing the public value type. Later entries for the same key update the value in place like
    /// MinjaMap.Set, keeping the original insertion position.
    /// </summary>
    private static MinjaValue EvalMapLiteral(MapLiteral node, MinjaLimits limits, TemplateScope? scope)
    {
        limits.RequireSequenceLength(node.Entries.Count, "map literal", node.Offset);

        MinjaMap map = new();

        foreach (MapEntry entry in node.Entries)
        {
            MinjaValue key = Eval(entry.Key, limits, scope);

            if (key.Kind != MinjaValueKind.String)
                throw new MinjaRenderException("map literal", node.Offset, $"map keys must be strings but found a {key.Kind} key");

            map.Set(key.StringValue, Eval(entry.Value, limits, scope));
        }

        return MinjaValue.Map(map);
    }

    /// <summary>
    /// Applies a filter chain left to right. A name bound by a macro or set statement wins over the built-in
    /// namespace, which is how templates override a built-in with their own helper of the same name.
    /// </summary>
    private static MinjaValue EvalFilterChain(FilterChainExpression chain, MinjaLimits limits, TemplateScope? scope)
    {
        TemplateScope callSite = scope ?? TemplateScope.Root(null, limits);
        MinjaValue value = Eval(chain.Source, limits, scope);

        foreach (FilterApplication filter in chain.Filters)
        {
            limits.CheckCancellation("filter", filter.Offset);

            if (callSite.TryGetCallable(filter.Name, out MinjaCallable? callable) && callable is not null)
                value = callable.Invoke(filter.Arguments, callSite, filter.Offset, caller: null, input: value);
            else
                value = MinjaBuiltins.ApplyFilter(filter, value, callSite, limits);
        }

        return value;
    }

    /// <summary>
    /// Evaluates a call. Names bound to macros, captured block bodies or joiner callables invoke directly; plain
    /// names then resolve through the built-in function namespace, and anything else is an actionable render error.
    /// </summary>
    private static MinjaValue EvalCall(CallExpression node, MinjaLimits limits, TemplateScope? scope)
    {
        TemplateScope callSite = scope ?? TemplateScope.Root(null, limits);

        if (node.Callee is NameExpression name && callSite.TryGetCallable(name.Name, out MinjaCallable? callable) && callable is not null)
            return callable.Invoke(node.Arguments, callSite, node.Offset, caller: null, input: null);

        if (node.Callee is NameExpression function && !callSite.TryGetValue(function.Name, out _))
            return MinjaBuiltins.InvokeFunction(function.Name, node.Arguments, callSite, node.Offset, limits);

        MinjaValue callee = Eval(node.Callee, limits, scope);

        throw new MinjaRenderException("call", node.Offset, $"cannot call a {callee.Kind} value; only macros, block callables and built-in functions are callable");
    }

    private static MinjaValue EvalAttribute(AttributeExpression node, MinjaLimits limits, TemplateScope? scope)
    {
        MinjaValue target = Eval(node.Target, limits, scope);

        // Attribute access on a map reads the key; anything else (including a missing map key) is undefined, which is none.
        return target.Kind == MinjaValueKind.Map ? target.MapValue.Get(node.Name) : MinjaValue.None;
    }

    private static MinjaValue EvalIndex(IndexExpression node, MinjaLimits limits, TemplateScope? scope)
    {
        MinjaValue target = Eval(node.Target, limits, scope);

        if (node.Slice)
            return EvalSlice(target, node, limits, scope);

        MinjaValue key = Eval(node.Start!, limits, scope);

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

    private static MinjaValue EvalSlice(MinjaValue target, IndexExpression node, MinjaLimits limits, TemplateScope? scope)
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

        long? startRaw = node.Start is null ? null : RequireIndex(Eval(node.Start, limits, scope), node);
        long? stopRaw = node.Stop is null ? null : RequireIndex(Eval(node.Stop, limits, scope), node);
        long step = node.Step is null ? 1L : RequireIndex(Eval(node.Step, limits, scope), node);

        if (step == 0)
            throw new MinjaRenderException("slice", node.Offset, "slice step cannot be zero");

        (long start, long stop) = AdjustSlice(length, startRaw, stopRaw, step);

        // The number of items a slice can produce is computable from its bounds, so an over-budget slice is
        // rejected before any element is copied.
        long span = step > 0 ? stop - start : start - stop;
        long stride = Math.Abs(step);
        long count = span <= 0 ? 0 : span <= stride ? 1 : (span + stride - 1) / stride;
        limits.RequireSequenceLength(count, "slice", node.Offset);

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

        if (target.Kind == MinjaValueKind.String)
        {
            string sliced = string.Concat(picked.Select(v => v.StringValue));
            limits.RequireOutputLength(sliced.Length, "slice", node.Offset);

            return MinjaValue.Of(sliced);
        }

        return MinjaValue.List(picked);
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

    private static MinjaValue EvalUnary(UnaryExpression node, MinjaLimits limits, TemplateScope? scope)
    {
        MinjaValue operand = Eval(node.Operand, limits, scope);

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

    private static MinjaValue EvalBinary(BinaryExpression node, MinjaLimits limits, TemplateScope? scope)
    {
        MinjaValue left = Eval(node.Left, limits, scope);

        // and/or short-circuit and return the operand itself, not a boolean, matching Python and minja.
        if (node.Operator == BinaryOperator.And)
            return left.IsTruthy ? Eval(node.Right, limits, scope) : left;

        if (node.Operator == BinaryOperator.Or)
            return left.IsTruthy ? left : Eval(node.Right, limits, scope);

        MinjaValue right = Eval(node.Right, limits, scope);

        switch (node.Operator)
        {
            case BinaryOperator.Add:
                return Add(left, right, limits, node.Offset);

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

    private static MinjaValue Add(MinjaValue left, MinjaValue right, MinjaLimits limits, int offset)
    {
        if (left.Kind == MinjaValueKind.String && right.Kind == MinjaValueKind.String)
        {
            string combined = left.StringValue + right.StringValue;
            limits.RequireOutputLength(combined.Length, "string addition", offset);

            return MinjaValue.Of(combined);
        }

        if (left.Kind == MinjaValueKind.List && right.Kind == MinjaValueKind.List)
        {
            limits.RequireSequenceLength(left.Items.Count + right.Items.Count, "list addition", offset);

            return MinjaValue.List(left.Items.Concat(right.Items));
        }

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

    private static MinjaValue EvalComparison(ComparisonExpression node, MinjaLimits limits, TemplateScope? scope)
    {
        MinjaValue left = Eval(node.Left, limits, scope);
        MinjaValue right = Eval(node.Right, limits, scope);

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

    private static MinjaValue EvalMembership(MembershipExpression node, MinjaLimits limits, TemplateScope? scope)
    {
        MinjaValue item = Eval(node.Item, limits, scope);
        MinjaValue container = Eval(node.Container, limits, scope);

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

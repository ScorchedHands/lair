using System.Globalization;
using System.Text;

namespace Lair.Templating.Minja;

/// <summary>
/// Evaluates minja expressions and renders expression-only templates. Missing names, missing map keys and
/// out-of-range indexes all evaluate as none, mirroring minja's none/undefined equivalence; string and list
/// indexes accept Python-style negative offsets. Statement tags, filters, tests and calls raise
/// <see cref="MinjaRenderException"/> until their runtime is implemented.
/// </summary>
public static class MinjaEvaluator
{
    /// <summary>Parses and evaluates a single expression, e.g. <c>"1 + 2 * 3"</c>.</summary>
    public static MinjaValue Evaluate(string expression, IReadOnlyDictionary<string, MinjaValue>? context = null)
    {
        ArgumentNullException.ThrowIfNull(expression);

        TemplateNode root = MinjaParser.Parse("{{ " + expression + " }}");
        OutputNode output = root.Nodes.OfType<OutputNode>().First();

        return Eval(output.Expression, context);
    }

    /// <summary>Evaluates an expression node from an already parsed AST.</summary>
    public static MinjaValue Evaluate(ExpressionNode expression, IReadOnlyDictionary<string, MinjaValue>? context = null) =>
        Eval(expression, context);

    /// <summary>
    /// Renders a template that contains only text, comments and expression outputs. Text runs already carry
    /// the exact bytes to emit because the parser applies whitespace elision at parse time.
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

        StringBuilder builder = new();
        RenderNodes(template.Nodes, context, builder);
        return builder.ToString();
    }

    private static void RenderNodes(IReadOnlyList<MinjaNode> nodes, IReadOnlyDictionary<string, MinjaValue>? context, StringBuilder builder)
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
                    builder.Append(Eval(output.Expression, context).ToString());
                    break;

                case StatementNode statement:
                    throw MinjaRenderException.UnsupportedStatement(StatementName(statement), statement.Offset);

                default:
                    throw MinjaRenderException.UnsupportedExpression(node.GetType().Name, node.Offset);
            }
        }
    }

    private static string StatementName(StatementNode statement) => statement switch
    {
        IfStatement => "if",
        ForStatement => "for",
        SetStatement => "set",
        MacroStatement => "macro",
        CallStatement => "call",
        FilterStatement => "filter",
        BreakStatement => "break",
        ContinueStatement => "continue",
        _ => statement.GetType().Name,
    };

    // ---------------------------------------------------------------- expression evaluation

    private static MinjaValue Eval(ExpressionNode node, IReadOnlyDictionary<string, MinjaValue>? context)
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

            case NameExpression name:
                // Undefined names are none: minja does not distinguish missing from null.
                if (context is not null && context.TryGetValue(name.Name, out MinjaValue? value))
                    return value;

                return MinjaValue.None;

            case AttributeExpression attribute:
                return EvalAttribute(attribute, context);

            case IndexExpression index:
                return EvalIndex(index, context);

            case UnaryExpression unary:
                return EvalUnary(unary, context);

            case BinaryExpression binary:
                return EvalBinary(binary, context);

            case ComparisonExpression comparison:
                return EvalComparison(comparison, context);

            case MembershipExpression membership:
                return EvalMembership(membership, context);

            case ConditionalExpression conditional:
                return Eval(conditional.Condition, context).IsTruthy
                    ? Eval(conditional.TrueBranch, context)
                    : Eval(conditional.ElseBranch, context);

            case FilterChainExpression:
                throw MinjaRenderException.UnsupportedExpression("filter", node.Offset);

            case TestExpression:
                throw MinjaRenderException.UnsupportedExpression("test", node.Offset);

            case CallExpression:
                throw MinjaRenderException.UnsupportedExpression("call", node.Offset);

            default:
                throw MinjaRenderException.UnsupportedExpression(node.GetType().Name, node.Offset);
        }
    }

    private static MinjaValue EvalAttribute(AttributeExpression node, IReadOnlyDictionary<string, MinjaValue>? context)
    {
        MinjaValue target = Eval(node.Target, context);

        // Attribute access on a map reads the key; anything else (including a missing map key) is undefined, which is none.
        return target.Kind == MinjaValueKind.Map ? target.MapValue.Get(node.Name) : MinjaValue.None;
    }

    private static MinjaValue EvalIndex(IndexExpression node, IReadOnlyDictionary<string, MinjaValue>? context)
    {
        MinjaValue target = Eval(node.Target, context);

        if (node.Slice)
            return EvalSlice(target, node, context);

        MinjaValue key = Eval(node.Start!, context);

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

    private static MinjaValue EvalSlice(MinjaValue target, IndexExpression node, IReadOnlyDictionary<string, MinjaValue>? context)
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

        long? startRaw = node.Start is null ? null : RequireIndex(Eval(node.Start, context), node);
        long? stopRaw = node.Stop is null ? null : RequireIndex(Eval(node.Stop, context), node);
        long step = node.Step is null ? 1L : RequireIndex(Eval(node.Step, context), node);

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

    private static MinjaValue EvalUnary(UnaryExpression node, IReadOnlyDictionary<string, MinjaValue>? context)
    {
        MinjaValue operand = Eval(node.Operand, context);

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

    private static MinjaValue EvalBinary(BinaryExpression node, IReadOnlyDictionary<string, MinjaValue>? context)
    {
        MinjaValue left = Eval(node.Left, context);

        // and/or short-circuit and return the operand itself, not a boolean, matching Python and minja.
        if (node.Operator == BinaryOperator.And)
            return left.IsTruthy ? Eval(node.Right, context) : left;

        if (node.Operator == BinaryOperator.Or)
            return left.IsTruthy ? left : Eval(node.Right, context);

        MinjaValue right = Eval(node.Right, context);

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

    private static MinjaValue EvalComparison(ComparisonExpression node, IReadOnlyDictionary<string, MinjaValue>? context)
    {
        MinjaValue left = Eval(node.Left, context);
        MinjaValue right = Eval(node.Right, context);

        switch (node.Operator)
        {
            case CompareOperator.Eq:
                return MinjaValue.Of(left.Equals(right));

            case CompareOperator.Ne:
                return MinjaValue.Of(!left.Equals(right));
        }

        int? order = CompareOrder(left, right, node.Offset);

        return node.Operator switch
        {
            CompareOperator.Lt => MinjaValue.Of(order < 0),
            CompareOperator.Lte => MinjaValue.Of(order <= 0),
            CompareOperator.Gt => MinjaValue.Of(order > 0),
            CompareOperator.Gte => MinjaValue.Of(order >= 0),
            _ => throw MinjaRenderException.UnsupportedExpression(node.Operator.ToString(), node.Offset),
        };
    }

    /// <summary>Total ordering for ordered comparisons: numbers (including bools) compare numerically, strings ordinally, lists lexicographically. Anything else is a type error.</summary>
    private static int CompareOrder(MinjaValue left, MinjaValue right, int offset)
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
                int itemOrder = CompareOrder(left.Items[i], right.Items[i], offset);

                if (itemOrder != 0)
                    return itemOrder;
            }

            return left.Items.Count.CompareTo(right.Items.Count);
        }

        throw new MinjaRenderException("comparison", offset, $"cannot order-compare {left.Kind} with {right.Kind}");
    }

    private static MinjaValue EvalMembership(MembershipExpression node, IReadOnlyDictionary<string, MinjaValue>? context)
    {
        MinjaValue item = Eval(node.Item, context);
        MinjaValue container = Eval(node.Container, context);

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

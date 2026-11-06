namespace Lair.Templating.Minja;

/// <summary>An expression node inside an output tag or statement tag.</summary>
public abstract record ExpressionNode : MinjaNode;

public sealed record StringLiteral(string Value) : ExpressionNode;

public sealed record IntegerLiteral(long Value) : ExpressionNode;

public sealed record FloatLiteral(double Value) : ExpressionNode;

public sealed record BoolLiteral(bool Value) : ExpressionNode;

/// <summary>The None literal. minja treats none and undefined alike, but the AST keeps the written form.</summary>
public sealed record NoneLiteral() : ExpressionNode;

/// <summary>An inline list literal: {{ ['user', 'assistant'] }}. Items evaluate left to right into a MinjaValue list.</summary>
public sealed record ListLiteral(IReadOnlyList<ExpressionNode> Items) : ExpressionNode;

/// <summary>One key/value pair of an inline map literal.</summary>
public sealed record MapEntry(ExpressionNode Key, ExpressionNode Value);

/// <summary>An inline map/object literal: {{ {'role': 'system'} }}. Keys evaluate to strings and entries keep insertion order.</summary>
public sealed record MapLiteral(IReadOnlyList<MapEntry> Entries) : ExpressionNode;

public sealed record NameExpression(string Name) : ExpressionNode;

/// <summary>Attribute access such as {{ message.role }}.</summary>
public sealed record AttributeExpression(ExpressionNode Target, string Name) : ExpressionNode;

/// <summary>Index or slice access. A plain index has Stop, Step and Slice unset; slices fill whichever parts they provide.</summary>
public sealed record IndexExpression(
    ExpressionNode Target,
    ExpressionNode? Start,
    ExpressionNode? Stop,
    ExpressionNode? Step,
    bool Slice) : ExpressionNode;

public sealed record UnaryExpression(UnaryOperator Operator, ExpressionNode Operand) : ExpressionNode;

public sealed record BinaryExpression(BinaryOperator Operator, ExpressionNode Left, ExpressionNode Right) : ExpressionNode;

public sealed record ComparisonExpression(ExpressionNode Left, CompareOperator Operator, ExpressionNode Right) : ExpressionNode;

/// <summary>Membership test: {{ item in sequence }} with Negated for 'not in'.</summary>
public sealed record MembershipExpression(ExpressionNode Item, ExpressionNode Container, bool Negated) : ExpressionNode;

/// <summary>Test expression: {{ value is name(args) }} with Negated for 'is not'. Subject is the value the test answers about.</summary>
public sealed record TestExpression(string Name, IReadOnlyList<CallArgument> Arguments, bool Negated, ExpressionNode? Subject = null) : ExpressionNode;

/// <summary>Filter chain: {{ value | a | b(x) }}. Filters apply left to right starting from Source.</summary>
public sealed record FilterChainExpression(ExpressionNode Source, IReadOnlyList<FilterApplication> Filters) : ExpressionNode;

/// <summary>One filter in a chain. Offset is the position of the pipe that introduces it.</summary>
public sealed record FilterApplication(string Name, IReadOnlyList<CallArgument> Arguments) : MinjaNode;

/// <summary>Ternary conditional: {{ a if b else c }}. Ternary-less 'a if b' is rejected at parse time.</summary>
public sealed record ConditionalExpression(ExpressionNode TrueBranch, ExpressionNode Condition, ExpressionNode ElseBranch) : ExpressionNode;

public sealed record CallExpression(ExpressionNode Callee, IReadOnlyList<CallArgument> Arguments) : ExpressionNode;

/// <summary>
/// A value produced by the evaluator rather than written in template source. Built-in predicates that receive an
/// already-evaluated list or map pass it back through a call as this node so the value keeps its exact shape.
/// </summary>
internal sealed record ValueExpression(MinjaValue Value) : ExpressionNode;

/// <summary>A call or filter argument. Name is null for positional arguments.</summary>
public sealed record CallArgument(string? Name, ExpressionNode Value);

public enum UnaryOperator { Not, Negate }

public enum BinaryOperator { Add, Sub, Mul, FloorDiv, Div, Mod, Pow, And, Or }

public enum CompareOperator { Eq, Ne, Lt, Lte, Gt, Gte }

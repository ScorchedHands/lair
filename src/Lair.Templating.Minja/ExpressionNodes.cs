namespace Lair.Templating.Minja;

/// <summary>An expression node inside an output tag or statement tag.</summary>
public abstract record ExpressionNode : MinjaNode;

public sealed record StringLiteral(string Value) : ExpressionNode;

public sealed record IntegerLiteral(long Value) : ExpressionNode;

public sealed record FloatLiteral(double Value) : ExpressionNode;

public sealed record BoolLiteral(bool Value) : ExpressionNode;

/// <summary>The None literal. minja treats none and undefined alike, but the AST keeps the written form.</summary>
public sealed record NoneLiteral() : ExpressionNode;

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

/// <summary>Test expression: {{ value is name(args) }} with Negated for 'is not'.</summary>
public sealed record TestExpression(string Name, IReadOnlyList<CallArgument> Arguments, bool Negated) : ExpressionNode;

/// <summary>Filter chain: {{ value | a | b(x) }}. Filters apply left to right starting from Source.</summary>
public sealed record FilterChainExpression(ExpressionNode Source, IReadOnlyList<FilterApplication> Filters) : ExpressionNode;

public sealed record FilterApplication(string Name, IReadOnlyList<CallArgument> Arguments);

/// <summary>Ternary conditional: {{ a if b else c }}. Ternary-less 'a if b' is rejected at parse time.</summary>
public sealed record ConditionalExpression(ExpressionNode TrueBranch, ExpressionNode Condition, ExpressionNode ElseBranch) : ExpressionNode;

public sealed record CallExpression(ExpressionNode Callee, IReadOnlyList<CallArgument> Arguments) : ExpressionNode;

/// <summary>A call or filter argument. Name is null for positional arguments.</summary>
public sealed record CallArgument(string? Name, ExpressionNode Value);

public enum UnaryOperator { Not, Negate }

public enum BinaryOperator { Add, Sub, Mul, FloorDiv, Div, Mod, Pow, And, Or }

public enum CompareOperator { Eq, Ne, Lt, Lte, Gt, Gte }

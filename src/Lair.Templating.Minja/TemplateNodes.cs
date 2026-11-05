namespace Lair.Templating.Minja;

/// <summary>Base for every AST node. Offset is the 0-based position of the construct in the template source.</summary>
public abstract record MinjaNode
{
    public int Offset { get; init; }

    /// <summary>Length in characters where meaningful (tags and text runs); zero or unset otherwise.</summary>
    public int Length { get; init; }
}

/// <summary>Root of a parsed minja template.</summary>
public sealed record TemplateNode(IReadOnlyList<MinjaNode> Nodes) : MinjaNode;

/// <summary>A run of literal text to emit. Whitespace elision requested by adjacent tags is already applied.</summary>
public sealed record TextNode(string Text) : MinjaNode;

/// <summary>A comment tag {# ... #}. It never emits output but stays in the tree so structure stays inspectable.</summary>
public sealed record CommentNode : TaggedNode;

/// <summary>An expression output tag {{ ... }}.</summary>
public sealed record OutputNode(ExpressionNode Expression) : TaggedNode;

/// <summary>
/// Base for nodes produced by {% %} tags and comments. TrimLeft records {%- on the opening marker (whitespace before
/// this node is elided); TrimRight records -%} or -#} on the closing marker (whitespace after this node is elided).
/// Both effects are already applied to neighbouring TextNode content; the flags record which marker requested them.
/// </summary>
public abstract record TaggedNode : MinjaNode
{
    public bool TrimLeft { get; init; }

    public bool TrimRight { get; init; }
}

public abstract record StatementNode : TaggedNode;

public sealed record IfStatement(IReadOnlyList<IfBranch> Branches, IReadOnlyList<MinjaNode>? ElseBody) : StatementNode;

public sealed record IfBranch(ExpressionNode Condition, IReadOnlyList<MinjaNode> Body);

public sealed record ForStatement(
    TargetPattern Target,
    ExpressionNode Iterable,
    IReadOnlyList<MinjaNode> Body,
    bool Recursive,
    ExpressionNode? LoopCondition) : StatementNode;

/// <summary>A set statement. One target for plain assignment, several targets for destructuring such as {% set a, b = value %}.</summary>
public sealed record SetStatement(IReadOnlyList<SetTarget> Targets, ExpressionNode Value) : StatementNode;

/// <summary>A set target: a name or dotted namespaced assignment such as ns.attr. Attributes lists the dot-separated parts.</summary>
public sealed record SetTarget(string Name, IReadOnlyList<string> Attributes);

public sealed record MacroStatement(string Name, IReadOnlyList<MacroParameter> Parameters, IReadOnlyList<MinjaNode> Body) : StatementNode;

public sealed record MacroParameter(string Name, ExpressionNode? DefaultValue);

/// <summary>A {% call callable(args) %} block: the body is captured and passed to the callee as caller() content.</summary>
public sealed record CallStatement(
    ExpressionNode Callable,
    IReadOnlyList<CallArgument> Arguments,
    IReadOnlyList<MinjaNode> Body) : StatementNode;

/// <summary>A {% filter name(args) %} block with optional else branch.</summary>
public sealed record FilterStatement(
    string Name,
    IReadOnlyList<CallArgument> Arguments,
    IReadOnlyList<MinjaNode> Body,
    IReadOnlyList<MinjaNode>? ElseBody) : StatementNode;

public sealed record BreakStatement() : StatementNode;

public sealed record ContinueStatement() : StatementNode;

namespace Lair.Templating.Minja;

/// <summary>Destructuring target of a for loop: a single name or a parenthesized tuple of names.</summary>
public abstract record TargetPattern : MinjaNode;

public sealed record NameTarget(string Name) : TargetPattern;

public sealed record TupleTarget(IReadOnlyList<TargetPattern> Items) : TargetPattern;

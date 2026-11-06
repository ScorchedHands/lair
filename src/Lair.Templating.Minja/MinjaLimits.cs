using System.Diagnostics;
using System.Text;

namespace Lair.Templating.Minja;

/// <summary>
/// Hard, non-configurable resource limits the library applies to every template it parses or renders.
/// GGUF chat templates arrive from model files and are treated as untrusted input, so these defaults are
/// library-internal guards rather than options a caller can widen. They are published as read-only
/// constants (there is no setter and no per-call override) so tests and callers can reason about the exact
/// budget a template must stay inside.
/// </summary>
public static class MinjaLimitDefaults
{
    /// <summary>
    /// Largest template source the parser accepts, measured in UTF-16 code units (about 1 MiB of text).
    /// Enforced in <see cref="MinjaParser.Parse"/> before any lexing or parsing work happens.
    /// </summary>
    public const int MaxTemplateSourceLength = 1_048_576;

    /// <summary>
    /// Recursion budget for one parse. Each body, expression entry, primary, literal body, index body,
    /// tuple target and argument list counts one frame, so a parenthesized or literal nest burns roughly
    /// two to four frames per level of nesting. 128 frames keeps the deepest accepted parse orders of
    /// magnitude below the depth that would threaten the .NET stack while staying far above anything a real
    /// chat template nests.
    /// </summary>
    public const int MaxParserFrames = 128;

    /// <summary>
    /// Recursion budget for one expression evaluation: each nested expression node counts one frame.
    /// The parser already bounds AST depth, so this is defense in depth for ASTs built through other entry
    /// points (for example <see cref="MinjaEvaluator.Evaluate(ExpressionNode, IReadOnlyDictionary{string, MinjaValue})"/>).
    /// </summary>
    public const int MaxExpressionDepth = 160;

    /// <summary>
    /// Maximum nesting of macro, call, filter and recursive-loop invocations inside one render. This is the
    /// guard against self-recursive templates such as <c>{% macro m %}{{ m() }}{% endmacro %}{{ m() }}</c>,
    /// which would otherwise exhaust the stack.
    /// </summary>
    public const int MaxCallDepth = 48;

    /// <summary>
    /// Maximum length of any sequence the library materializes: a <c>range</c> result, a list or map literal,
    /// a map/select/reject/dictsort/items result, a slice, or a for-loop iterable (including the character
    /// expansion of a string). <c>range</c> computes its count and rejects an over-budget span before
    /// allocating; other paths reject a source that is already too large before materializing it.
    /// </summary>
    public const int MaxSequenceLength = 100_000;

    /// <summary>
    /// Total for-loop body executions allowed in one render, counting nested loops and recursive-loop
    /// re-entries. A loop aborts on the iteration that would cross this budget.
    /// </summary>
    public const int MaxLoopIterationsPerRender = 200_000;

    /// <summary>
    /// Maximum rendered text a single render may produce, in UTF-16 code units (4 MiB). Every output append
    /// path — text runs, expression outputs, macro/call/filter bodies, and builtins that build strings such
    /// as <c>join</c>, <c>tojson</c> and <c>escape</c> — is checked against this budget.
    /// </summary>
    public const int MaxRenderedOutputCharacters = 4_194_304;

    /// <summary>
    /// Wall-clock budget for one render. The deadline is checked cooperatively at loop-iteration and
    /// expression boundaries, so a template that keeps producing output stops here instead of running for
    /// as long as its other budgets allow.
    /// </summary>
    public static TimeSpan MaxRenderDuration => TimeSpan.FromSeconds(30);

    /// <summary>
    /// Number of loop iterations evaluated between two cancellation/deadline probes inside one loop, so the
    /// hot path pays for the clock only occasionally.
    /// </summary>
    public const int LoopCancellationCheckInterval = 64;
}

/// <summary>
/// Per-operation guard that carries the budgets in <see cref="MinjaLimitDefaults"/> through a parse or a
/// render: recursion frames, materialized sequence lengths, loop iterations, rendered output characters,
/// a stopwatch deadline and the caller's <see cref="CancellationToken"/>. One guard belongs to one
/// operation; nested evaluation reaches it through the scope chain so every append and allocation path is
/// checked against the same budget.
/// </summary>
internal sealed class MinjaLimits
{
    private readonly Stopwatch _clock = Stopwatch.StartNew();

    private int _expressionDepth;
    private int _callDepth;
    private long _loopIterations;
    private long _outputChars;

    private MinjaLimits(CancellationToken cancellation)
    {
        Cancellation = cancellation;
    }

    /// <summary>Caller-supplied cooperative cancellation for this operation.</summary>
    public CancellationToken Cancellation { get; }

    /// <summary>Guard for a parse operation (source-size and parser-frame budgets).</summary>
    public static MinjaLimits ForParse(CancellationToken cancellation) => new(cancellation);

    /// <summary>Guard for a render or evaluation operation (all runtime budgets).</summary>
    public static MinjaLimits ForRender(CancellationToken cancellation) => new(cancellation);

    /// <summary>
    /// Finds the guard that owns a scope chain, so evaluation reached from any entry point charges the same
    /// budget. A scope built outside a guarded render (for example a caller-supplied scope handed straight
    /// to an internal render path) gets a fresh guard rooted at that render instead.
    /// </summary>
    public static MinjaLimits ForScope(TemplateScope scope, CancellationToken cancellation)
    {
        for (TemplateScope? candidate = scope; candidate is not null; candidate = candidate.Parent)
        {
            if (candidate.Limits is MinjaLimits guard)
                return guard;
        }

        return ForRender(cancellation);
    }

    // ---------------------------------------------------------------- recursion budgets

    /// <summary>Claims one parser recursion frame; throws when the parse nesting budget is exceeded.</summary>
    public void EnterParserFrame(int offset)
    {
        if (++ParserFrames > MinjaLimitDefaults.MaxParserFrames)
            throw MinjaSyntaxException.LimitExceeded("parser depth", offset);
    }

    public void ExitParserFrame() => ParserFrames--;

    /// <summary>Parser recursion frames currently open; owned by the parser instance.</summary>
    public int ParserFrames { get; set; }

    /// <summary>Claims one expression-evaluation frame for a nested expression node.</summary>
    public void EnterExpression(int offset)
    {
        if (++_expressionDepth > MinjaLimitDefaults.MaxExpressionDepth)
            throw new RenderLimitsException("evaluator depth", "expression", offset);

        // Expression boundaries are the cheapest place to notice a cancelled caller: a volatile read of the
        // token costs far less than the stopwatch, so the deadline is left to the coarser boundaries.
        ThrowIfCancelled("expression", offset);
    }

    public void ExitExpression() => _expressionDepth--;

    /// <summary>Claims one macro/call/filter/recursive-loop invocation frame.</summary>
    public void EnterCall(int offset)
    {
        if (++_callDepth > MinjaLimitDefaults.MaxCallDepth)
            throw new RenderLimitsException("call depth", "macro", offset);
    }

    public void ExitCall() => _callDepth--;

    // ---------------------------------------------------------------- sequence budget

    /// <summary>
    /// Validates the length of a sequence about to be materialized or iterated. Over-budget sources fail
    /// with the limit named before any list is built, which is what keeps a huge <c>range</c> or a huge
    /// string iteration from allocating first and failing later.
    /// </summary>
    public void RequireSequenceLength(long length, string construct, int offset)
    {
        if (length > MinjaLimitDefaults.MaxSequenceLength)
            throw new RenderLimitsException("sequence length", construct, offset, $"at most {MinjaLimitDefaults.MaxSequenceLength} items are allowed but this {construct} would produce {length}");
    }

    // ---------------------------------------------------------------- loop budget

    /// <summary>
    /// Claims the next loop iteration and probes cancellation/deadline every
    /// <see cref="MinjaLimitDefaults.LoopCancellationCheckInterval"/> iterations.
    /// </summary>
    public void ClaimLoopIteration(string construct, int offset)
    {
        if (++_loopIterations > MinjaLimitDefaults.MaxLoopIterationsPerRender)
            throw new RenderLimitsException("loop iterations", construct, offset, $"a single render may execute at most {MinjaLimitDefaults.MaxLoopIterationsPerRender} loop iterations");

        if (_loopIterations % MinjaLimitDefaults.LoopCancellationCheckInterval == 0)
            CheckDeadline(construct, offset);
    }

    // ---------------------------------------------------------------- output budget

    /// <summary>
    /// Appends rendered text to a builder under the output budget. Text runs are the only path that may
    /// cross the limit in one step (the template source itself is size-bounded), so a run that fits is
    /// charged whole while any other append that would overflow aborts with the limit named.
    /// </summary>
    public void Append(StringBuilder builder, string text, string construct, int offset)
    {
        long projected = _outputChars + text.Length;

        if (projected > MinjaLimitDefaults.MaxRenderedOutputCharacters)
        {
            bool templateText = construct == "template text";

            if (!templateText || projected > MinjaLimitDefaults.MaxRenderedOutputCharacters + MinjaLimitDefaults.MaxTemplateSourceLength)
                throw OutputSizeExceeded(construct, offset);
        }

        builder.Append(text);
        _outputChars = projected;
    }

    /// <summary>
    /// Validates the length of a string a builtin is about to produce (join, tojson, escape, string
    /// concatenation, string slices) before it is returned to the render.
    /// </summary>
    public void RequireOutputLength(long length, string construct, int offset)
    {
        if (length > MinjaLimitDefaults.MaxRenderedOutputCharacters)
            throw OutputSizeExceeded(construct, offset);
    }

    private RenderLimitsException OutputSizeExceeded(string construct, int offset) =>
        new("output size", construct, offset, $"a single render may produce at most {MinjaLimitDefaults.MaxRenderedOutputCharacters} characters");

    // ---------------------------------------------------------------- deadline and cancellation

    /// <summary>Probes the caller's cancellation token and the render deadline at an expression boundary.</summary>
    public void CheckCancellation(string construct, int offset)
    {
        ThrowIfCancelled(construct, offset);
        CheckDeadline(construct, offset);
    }

    /// <summary>
    /// Throws when the caller's token is cancelled: a token read only, without the stopwatch probe, so the
    /// cheapest boundaries (expression entries) can afford the check while the deadline stays with the coarser
    /// boundaries. Surfaced as an actionable cancellation render error rather than a hang or a bare
    /// OperationCanceledException.
    /// </summary>
    private void ThrowIfCancelled(string construct, int offset)
    {
        if (Cancellation.IsCancellationRequested)
            throw new RenderLimitsException("cancellation", construct, offset, "rendering was cancelled through the caller's CancellationToken");
    }

    /// <summary>Probes only the stopwatch deadline (used on loop hot paths).</summary>
    public void CheckDeadline(string construct, int offset)
    {
        if (_clock.Elapsed > MinjaLimitDefaults.MaxRenderDuration)
            throw new RenderLimitsException("render deadline", construct, offset, $"rendering exceeded the {MinjaLimitDefaults.MaxRenderDuration.TotalSeconds:0} second deadline");
    }
}

/// <summary>
/// Thrown when a template stays inside minja's grammar but asks the library for more resources than
/// <see cref="MinjaLimitDefaults"/> allows: generated sequence length, loop iterations, rendered output
/// size, evaluator or call recursion depth, the render deadline, or a caller cancellation. It is a
/// <see cref="MinjaRenderException"/> so existing render-error handling keeps working, while
/// <see cref="Limit"/> names the specific guard that fired. Parse-time guards (template size, parser
/// depth) stay <see cref="MinjaSyntaxException"/> instead.
/// </summary>
public sealed class RenderLimitsException : MinjaRenderException
{
    /// <summary>Name of the guard that was exceeded: sequence length, loop iterations, output size, call depth, evaluator depth, render deadline or cancellation.</summary>
    public string Limit { get; }

    public RenderLimitsException(string limit, string construct, int position)
        : base(construct, position, $"{limit} limit exceeded")
    {
        Limit = limit;
    }

    public RenderLimitsException(string limit, string construct, int position, string detail)
        : base(construct, position, $"{limit} limit exceeded; {detail}")
    {
        Limit = limit;
    }
}

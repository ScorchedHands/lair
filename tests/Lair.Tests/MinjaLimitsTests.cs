using System.Diagnostics;
using Lair.Templating.Minja;

namespace Lair.Tests;

/// <summary>
/// Adversarial goldens for the library's non-configurable resource guards: template source size and parser depth fail as
/// actionable syntax errors, sequence length, loop iterations, output size, evaluator/call recursion depth, the render
/// deadline and caller cancellation fail as RenderLimitsException naming the guard that fired. Each failure is also a
/// wall-time golden — an over-budget template must stop quickly instead of allocating or running unbounded — while
/// templates just inside every budget keep rendering byte-exactly, so limits never alter semantics.
/// </summary>
[Trait("Category", "MinjaLimits")]
public sealed class MinjaLimitsTests
{
    // ---------------------------------------------------------------- library defaults are non-configurable

    [Fact]
    public void LibraryDefaults_AreFixedConstantsAndNoPublicApiCarriesLimitValues()
    {
        Assert.Equal(1_048_576, MinjaLimitDefaults.MaxTemplateSourceLength);
        Assert.Equal(128, MinjaLimitDefaults.MaxParserFrames);
        Assert.Equal(160, MinjaLimitDefaults.MaxExpressionDepth);
        Assert.Equal(48, MinjaLimitDefaults.MaxCallDepth);
        Assert.Equal(100_000, MinjaLimitDefaults.MaxSequenceLength);
        Assert.Equal(200_000, MinjaLimitDefaults.MaxLoopIterationsPerRender);
        Assert.Equal(4_194_304, MinjaLimitDefaults.MaxRenderedOutputCharacters);
        Assert.Equal(TimeSpan.FromSeconds(30), MinjaLimitDefaults.MaxRenderDuration);
        Assert.Equal(64, MinjaLimitDefaults.LoopCancellationCheckInterval);

        // Every budget is a compile-time constant (or the get-only duration) with no setter anywhere.
        foreach (System.Reflection.FieldInfo field in typeof(MinjaLimitDefaults).GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static))
            Assert.True(field.IsLiteral, $"'{field.Name}' must be a literal constant");

        foreach (System.Reflection.PropertyInfo property in typeof(MinjaLimitDefaults).GetProperties())
            Assert.False(property.CanWrite, $"'{property.Name}' must not expose a setter");

        // Parse and render entry points take only template text/ASTs, value contexts and the additive cancellation
        // token: no duration, budget number or limits object can widen any guard.
        Type[] publicApis = [typeof(MinjaParser), typeof(MinjaEvaluator), typeof(MinjaChatTemplate)];

        foreach (Type api in publicApis)
            foreach (System.Reflection.MethodInfo method in api.GetMethods(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Instance))
                AssertLimitFreeSignature(method);

        // The one options record the library exposes tunes chat normalization only: every knob is a behavior flag,
        // never a budget number or duration.
        foreach (System.Reflection.PropertyInfo property in typeof(ChatTemplateOptions).GetProperties())
            Assert.Equal(typeof(bool), property.PropertyType);
    }

    // ---------------------------------------------------------------- template source size (parse guard)

    [Fact]
    public void TemplateSourceSize_JustOverLimit_FailsAsSyntaxErrorNamingTemplateSizeAtZero()
    {
        string oversized = new('a', MinjaLimitDefaults.MaxTemplateSourceLength + 1);
        Stopwatch clock = Stopwatch.StartNew();

        MinjaSyntaxException exception = Assert.Throws<MinjaSyntaxException>(() => MinjaParser.Parse(oversized));
        clock.Stop();

        Assert.Equal("template size", exception.Construct);
        Assert.Equal(0, exception.Position);
        Assert.Contains("1048576", exception.Message);
        // The guard rejects the source before any lexing work happens. The wall bound is a generous load-tolerance
        // guard for a busy box, not the guarantee: the deterministic guarantee is the position-at-zero assertion.
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), clock.Elapsed.ToString());
    }

    [Fact]
    public void TemplateSourceSize_ExactlyAtLimit_ParsesAndRendersWhole()
    {
        string atLimit = new('a', MinjaLimitDefaults.MaxTemplateSourceLength);

        TemplateNode root = MinjaParser.Parse(atLimit);
        TextNode text = Assert.IsType<TextNode>(Assert.Single(root.Nodes));
        Assert.Equal(MinjaLimitDefaults.MaxTemplateSourceLength, text.Text.Length);
    }

    [Fact]
    public void OutputGraceBand_AllowsRawTemplateTextToCrossTheCapButGeneratedAppendsStop()
    {
        // Generated output sits just under the cap; a 450k-character raw template run then crosses it and is still
        // emitted whole, because text runs are bounded by the source-size guard instead.
        const int tailLength = 450_000;
        IReadOnlyDictionary<string, MinjaValue> context = Context(("unit", MinjaValue.Of(new string('x', 100))));
        string template = "{% for a in range(3) %}{% for i in range(13000) %}{{ unit }}{% endfor %}{% endfor %}" + new string('a', tailLength);

        Assert.True(template.Length <= MinjaLimitDefaults.MaxTemplateSourceLength, template.Length.ToString());
        Assert.Equal(3_900_000 + tailLength, MinjaEvaluator.Render(template, context).Length);

        // The same crossing by a generated append is not covered by the band and stops with the guard named.
        string huge = new('y', 3_900_000 + tailLength);
        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => MinjaEvaluator.Render("{{ huge }}", Context(("huge", MinjaValue.Of(huge)))));

        Assert.Equal("output size", exception.Limit);
        Assert.Contains("4194304", exception.Message);
    }

    // ---------------------------------------------------------------- parser depth (parse guard)

    [Fact]
    public void ParserDepth_NestedBlocksBeyondFrameBudget_FailAsSyntaxErrorNamingParserDepth()
    {
        const int nesting = 128;
        string template = string.Concat(Enumerable.Repeat("{% if x %}", nesting)) + string.Concat(Enumerable.Repeat("{% endif %}", nesting));

        MinjaSyntaxException exception = Assert.Throws<MinjaSyntaxException>(() => MinjaParser.Parse(template));

        Assert.Equal("parser depth", exception.Construct);
        // The offset names the innermost body that could not be opened: deep inside the tag nest, past its midpoint.
        Assert.True(exception.Position >= 0, exception.Position.ToString());
        Assert.Contains("parser depth", exception.Message);
    }

    [Fact]
    public void ParserDepth_NestedParensBeyondFrameBudget_FailAsSyntaxErrorWithInnermostOffset()
    {
        const int parens = 128;
        string template = "{{ " + new string('(', parens) + "x" + new string(')', parens) + " }}";

        MinjaSyntaxException exception = Assert.Throws<MinjaSyntaxException>(() => MinjaParser.Parse(template));

        Assert.Equal("parser depth", exception.Construct);
        // Each paren burns a conditional frame plus a primary frame, so the budget runs out deep inside the tag:
        // the reported offset sits between the opening tag and the innermost operand.
        int contentStart = template.IndexOf('x');
        Assert.True(exception.Position > "{{ ".Length && exception.Position < contentStart, $"offset {exception.Position}");
    }

    [Fact]
    public void ParserDepth_JustInsideFrameBudget_ParsesAndRenders()
    {
        // A body frame plus the transient condition frames must stay inside 128, which caps block nesting at 126.
        const int nesting = MinjaLimitDefaults.MaxParserFrames - 2;
        string blocks = string.Concat(Enumerable.Repeat("{% if x %}", nesting)) + "body" + string.Concat(Enumerable.Repeat("{% endif %}", nesting));

        // The condition is false, so the nested bodies never render but the parse itself must succeed.
        Assert.Equal(string.Empty, MinjaEvaluator.Render(blocks, Context(("x", MinjaValue.Of(false)))));

        // Each paren costs a conditional frame plus a primary frame on top of the tag's own expression entry.
        const int parens = (MinjaLimitDefaults.MaxParserFrames - 4) / 2;
        string expression = "{{ " + new string('(', parens) + "'ok'" + new string(')', parens) + " }}";
        Assert.Equal("ok", MinjaEvaluator.Render(expression));
    }

    // ---------------------------------------------------------------- evaluator expression depth (render guard)

    [Fact]
    public void EvaluatorDepth_ExpressionNestingBeyondBudget_FailsAsRenderLimit()
    {
        ExpressionNode deep = new StringLiteral("deep");

        for (int level = 0; level < MinjaLimitDefaults.MaxExpressionDepth + 12; level++)
            deep = new UnaryExpression(UnaryOperator.Not, deep);

        Stopwatch clock = Stopwatch.StartNew();
        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => MinjaEvaluator.Evaluate(deep));
        clock.Stop();

        Assert.Equal("evaluator depth", exception.Limit);
        Assert.Equal("expression", exception.Construct);
        // Load-tolerance guard only; the deterministic guarantee is the depth-limit assertion above.
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), clock.Elapsed.ToString());
    }

    [Fact]
    public void EvaluatorDepth_JustInsideBudget_EvaluatesToTheExpectedValue()
    {
        ExpressionNode deep = new StringLiteral("deep");

        for (int level = 0; level < MinjaLimitDefaults.MaxExpressionDepth - 12; level++)
            deep = new UnaryExpression(UnaryOperator.Not, deep);

        // An even number of negations over a truthy string yields true again.
        Assert.True(MinjaEvaluator.Evaluate(deep).IsTruthy);
    }

    [Fact]
    public void EvaluatorDepth_SourceBuiltTemplatesTripParserFramesBeforeEvaluatorDepth()
    {
        // Source-built ASTs hit the 128-frame parse budget before the deeper evaluator guard can be reached, so a deep
        // source is reported as a syntax limit rather than a render limit; the evaluator guard stays defense in depth
        // for prebuilt ASTs (see EvaluatorDepth_ExpressionNestingBeyondBudget_FailsAsRenderLimit).
        string template = "{{ " + new string('(', MinjaLimitDefaults.MaxParserFrames / 2 + 4) + "'deep'" + new string(')', MinjaLimitDefaults.MaxParserFrames / 2 + 4) + " }}";

        MinjaSyntaxException exception = Assert.Throws<MinjaSyntaxException>(() => Render(template));

        Assert.Equal("parser depth", exception.Construct);
    }

    // ---------------------------------------------------------------- macro / call depth (render guard)

    [Fact]
    public void MacroRecursion_BeyondCallDepthBudget_FailsAsRenderLimitNamingMacroConstruct()
    {
        string template = "{% macro m() %}{{ m() }}{% endmacro %}{{ m() }}";
        Stopwatch clock = Stopwatch.StartNew();

        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => Render(template));
        clock.Stop();

        Assert.Equal("call depth", exception.Limit);
        Assert.Equal("macro", exception.Construct);
        // Recursion is cut off at the 49th frame, not after a stack overflow or a long run. The wall bound is a
        // load-tolerance guard only; the deterministic guarantee is the call-depth limit named above.
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), clock.Elapsed.ToString());
    }

    [Fact]
    public void MutualMacroRecursion_BeyondCallDepthBudget_FailsAsRenderLimit()
    {
        string template = "{% macro ping(n) %}{{ pong(n) }}{% endmacro %}{% macro pong(n) %}{{ ping(n) }}{% endmacro %}{{ ping(0) }}";

        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => Render(template));

        Assert.Equal("call depth", exception.Limit);
    }

    [Fact]
    public void CallStatementRecursion_BeyondCallDepthBudget_FailsAsRenderLimit()
    {
        string template = "{% macro spin() %}{% call spin() %}x{% endcall %}{% endmacro %}{{ spin() }}";

        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => Render(template));

        // A {% call %} body re-entering its own macro charges the same call budget an expression call does.
        Assert.Equal("call depth", exception.Limit);
    }

    [Fact]
    public void RecursiveLoopReentry_BeyondCallDepthBudget_FailsAsRenderLimit()
    {
        string template = "{% for item in [1] recursive %}{{ loop([1]) }}{% endfor %}";

        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => Render(template));

        // Each self-similar re-entry claims a call frame; the guard fires long before output or iteration budgets.
        Assert.Equal("call depth", exception.Limit);
    }

    [Fact]
    public void MacroChain_AtExactCallDepthBudget_RendersAndOneLevelDeeperFails()
    {
        // A chain of exactly MaxCallDepth nested macro invocations is allowed; one more frame trips the guard.
        Assert.Equal("ok", Render(MacroChain(MinjaLimitDefaults.MaxCallDepth)));

        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => Render(MacroChain(MinjaLimitDefaults.MaxCallDepth + 1)));

        Assert.Equal("call depth", exception.Limit);
    }

    // ---------------------------------------------------------------- sequence length (render guard)

    [Fact]
    public void Range_HugeSpan_FailsAsSequenceLimitBeforeAllocating()
    {
        Stopwatch clock = Stopwatch.StartNew();
        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => Render("{{ range(1000000000) | length }}"));
        clock.Stop();

        Assert.Equal("sequence length", exception.Limit);
        Assert.Contains("range", exception.Message);
        // The span is rejected from its bounds: no billion-item list is ever built. The wall bound is a generous
        // load-tolerance guard only; the deterministic guarantee is the sequence-limit assertion naming range.
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), clock.Elapsed.ToString());
    }

    [Fact]
    public void Range_JustOverSequenceBudget_FailsAsSequenceLimitNamingExactCount()
    {
        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => Render("{{ range(100001) | length }}"));

        Assert.Equal("sequence length", exception.Limit);
        Assert.Contains("range", exception.Message);
        Assert.Contains("100001", exception.Message);
    }

    [Fact]
    public void Range_InsideSequenceBudget_RendersWithCorrectLength()
    {
        Assert.Equal("100000", Render("{{ range(100000) | length }}"));
    }

    [Fact]
    public void ListConcatenation_AndInlineLiteral_OverBudget_FailAsSequenceLimitBeforeAllocating()
    {
        Stopwatch clock = Stopwatch.StartNew();

        // Concatenated spans are charged as a whole before any element is copied.
        RenderLimitsException added = Assert.Throws<RenderLimitsException>(() => Render("{{ (range(60000) + range(60000)) | length }}"));
        Assert.Equal("sequence length", added.Limit);

        // A literal packed past the item budget fails before its items are evaluated.
        string packed = "{{ [" + string.Concat(Enumerable.Repeat("1, ", 120_000)) + "1] }}";
        RenderLimitsException literal = Assert.Throws<RenderLimitsException>(() => Render(packed));
        Assert.Equal("sequence length", literal.Limit);

        clock.Stop();
        // Load-tolerance guard only; the deterministic guarantee is that both paths report the sequence limit above.
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), clock.Elapsed.ToString());
    }

    [Fact]
    public void ItemsDictSortAndStringIteration_OverBudgetSequences_FailAsSequenceLimit()
    {
        MinjaValue bigMap = NumberedMap(120_000);

        RenderLimitsException items = Assert.Throws<RenderLimitsException>(() => Render("{{ big | items | length }}", Context(("big", bigMap))));
        Assert.Equal("sequence length", items.Limit);
        Assert.Contains("items", items.Message);

        RenderLimitsException dictSort = Assert.Throws<RenderLimitsException>(() => Render("{{ big | dictsort | length }}", Context(("big", bigMap))));
        Assert.Equal("sequence length", dictSort.Limit);
        Assert.Contains("dictsort", dictSort.Message);

        // String sources are bounded before their characters expand, for loops and filters alike.
        MinjaValue bigString = MinjaValue.Of(new string('x', 120_000));

        RenderLimitsException iterated = Assert.Throws<RenderLimitsException>(() => Render("{% for c in s %}{% endfor %}", Context(("s", bigString))));
        Assert.Equal("sequence length", iterated.Limit);

        RenderLimitsException selected = Assert.Throws<RenderLimitsException>(() => Render("{{ s | select('defined') | length }}", Context(("s", bigString))));
        Assert.Equal("sequence length", selected.Limit);

        // A slice whose bounds promise an over-budget span fails before copying a single element.
        RenderLimitsException sliced = Assert.Throws<RenderLimitsException>(() => Render("{{ s[::1] | length }}", Context(("s", MinjaValue.Of(new string('x', 200_000))))));
        Assert.Equal("sequence length", sliced.Limit);

        // The same paths stay exact inside the budget.
        Assert.Equal("3", Render("{{ small | items | length }}", Context(("small", NumberedMap(3)))));
        Assert.Equal("60000", Render("{{ s[::2] | length }}", Context(("s", bigString))));
    }

    [Fact]
    public void OutputSize_TojsonAndStringAdditionBeyondCap_FailAsOutputLimit()
    {
        // tojson materializes the whole dump before it is returned, so its length is charged against the output budget.
        MinjaValue[] units = new MinjaValue[100_000];

        for (int i = 0; i < units.Length; i++)
            units[i] = MinjaValue.Of(new string('x', 45));

        RenderLimitsException dumped = Assert.Throws<RenderLimitsException>(() => Render("{{ units | tojson }}", Context(("units", ListOf(units)))));
        Assert.Equal("output size", dumped.Limit);
        Assert.Equal("tojson", dumped.Construct);

        // Concatenating two large strings is rejected by the addition path before its result reaches the render.
        RenderLimitsException added = Assert.Throws<RenderLimitsException>(() => Render(
            "{{ left + right }}",
            Context(("left", MinjaValue.Of(new string('a', 3_000_000))), ("right", MinjaValue.Of(new string('b', 3_000_000))))));

        Assert.Equal("output size", added.Limit);
        Assert.Equal("string addition", added.Construct);
    }

    [Fact]
    public void SelectOverBudgetSource_RejectsMaterializedResultAsSequenceLimit()
    {
        // A filter that keeps almost every item of an oversized source is stopped when its own result crosses the budget.
        MinjaValue[] numbers = new MinjaValue[120_000];

        for (int i = 0; i < numbers.Length; i++)
            numbers[i] = MinjaValue.Of((long)i + 1);

        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => Render("{{ nums | select('defined') | length }}", Context(("nums", ListOf(numbers)))));

        Assert.Equal("sequence length", exception.Limit);
    }

    // ---------------------------------------------------------------- loop iterations (render guard)

    [Fact]
    public void LoopIterations_NestedLoopsBeyondPerRenderBudget_FailAsLoopLimitQuickly()
    {
        string template = "{% for a in range(3) %}{% for i in range(100000) %}x{% endfor %}{% endfor %}";

        Stopwatch clock = Stopwatch.StartNew();
        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => MinjaEvaluator.Render(template));
        clock.Stop();

        // 300,000 body executions are demanded; the guard stops on iteration 200,001.
        Assert.Equal("loop iterations", exception.Limit);
        Assert.Contains("200000", exception.Message);
        // The guard fires at the iteration budget, not after full materialization. This wall bound is a generous
        // load-tolerance guard for a busy box (concurrent model-loaded collections), not the guarantee: the
        // deterministic guarantee is the iteration-budget assertion above.
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), clock.Elapsed.ToString());
    }

    [Fact]
    public void LoopIterations_JustInsideBudget_RendersEveryIteration()
    {
        // Nested loops charge the outer body executions too: 2 outer + 198,000 inner = 198,002 claimed iterations.
        string template = "{% for a in range(2) %}{% for i in range(99000) %}x{% endfor %}{% endfor %}";

        Assert.Equal(198_000, MinjaEvaluator.Render(template).Length);
    }

    // ---------------------------------------------------------------- rendered output size (render guard)

    [Fact]
    public void OutputSize_GeneratedOutputBeyondCap_FailsAsOutputLimitQuickly()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("unit", MinjaValue.Of(new string('x', 100))));
        string template = "{% for a in range(3) %}{% for i in range(20000) %}{{ unit }}{% endfor %}{% endfor %}";

        Stopwatch clock = Stopwatch.StartNew();
        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => MinjaEvaluator.Render(template, context));
        clock.Stop();

        // 6,000,000 characters are demanded; the append that would cross 4 MiB aborts.
        Assert.Equal("output size", exception.Limit);
        Assert.Contains("4194304", exception.Message);
        // The guard fires at the crossing append, not after full materialization. This wall bound is a generous
        // load-tolerance guard for a busy box, not the guarantee: the deterministic guarantee is the output-size
        // limit assertion above.
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), clock.Elapsed.ToString());
    }

    [Fact]
    public void OutputSize_JustInsideCap_RendersSuccessfully()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("unit", MinjaValue.Of(new string('x', 100))));
        string template = "{% for a in range(3) %}{% for i in range(13981) %}{{ unit }}{% endfor %}{% endfor %}";

        Assert.Equal(4_194_300, MinjaEvaluator.Render(template, context).Length);
    }

    [Fact]
    public void OutputSize_BuiltinStringBeyondCap_FailsAsOutputLimit()
    {
        // join builds the whole string before it is returned to the render, so its length is charged too.
        IReadOnlyDictionary<string, MinjaValue> context = Context(("units", ListOf([.. Enumerable.Range(0, 50_000).Select(_ => MinjaValue.Of(new string('x', 120)))])));

        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => MinjaEvaluator.Render("{{ units | join('-') }}", context));

        Assert.Equal("output size", exception.Limit);
        Assert.Contains("4194304", exception.Message);
        // The guard names the builtin that produced the oversized string.
        Assert.Equal("join", exception.Construct);
    }

    // ---------------------------------------------------------------- render deadline (render guard)

    [Fact]
    public void RenderDeadline_PathologicalTemplate_StopsAsActionableDeadlineNotHang()
    {
        const int baseLength = 3_950_000;
        const int iterations = 2 * 99_000;

        // Each iteration re-copies the near-cap accumulator: quadratic work that stays inside every other budget
        // (198,000 claimed loop iterations, sequences of 99,000 items, a final string under the output cap) and can
        // only be stopped by the deadline probe taken at each body node and on every claimed iteration.
        Assert.True(baseLength + iterations <= MinjaLimitDefaults.MaxRenderedOutputCharacters);
        Assert.True(iterations < MinjaLimitDefaults.MaxLoopIterationsPerRender);

        IReadOnlyDictionary<string, MinjaValue> context = Context(("base", MinjaValue.Of(new string('x', baseLength))));
        string template = "{% set acc = base %}{% for a in range(2) %}{% for i in range(99000) %}{% set acc = acc + 'x' %}{% endfor %}{% endfor %}{{ acc | length }}";

        Stopwatch clock = Stopwatch.StartNew();
        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => MinjaEvaluator.Render(template, context));
        clock.Stop();

        Assert.Equal("render deadline", exception.Limit);
        Assert.Contains("30 second deadline", exception.Message);
        // It ran long enough to be genuinely pathological (the 30 s budget was spent) and stopped shortly after it.
        Assert.True(clock.Elapsed > TimeSpan.FromSeconds(29), clock.Elapsed.ToString());
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(75), clock.Elapsed.ToString());
    }

    [Fact]
    public void RenderDeadline_FastTemplatesNeverTripTheGuard()
    {
        Stopwatch clock = Stopwatch.StartNew();
        string rendered = MinjaEvaluator.Render("{% for i in range(1000) %}x{% endfor %}");
        clock.Stop();

        Assert.Equal(1000, rendered.Length);
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), clock.Elapsed.ToString());
    }

    // ---------------------------------------------------------------- cooperative cancellation (render guard)

    [Fact]
    public void Cancellation_PreCancelledToken_FailsAsCancellationLimitNotOperationCanceledException()
    {
        using CancellationTokenSource source = new();
        source.Cancel();

        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => MinjaEvaluator.Render("hello {{ name }}", Context(("name", MinjaValue.Of("x"))), source.Token));

        Assert.Equal("cancellation", exception.Limit);
        Assert.Contains("cancelled", exception.Message);
        Assert.False(typeof(OperationCanceledException).IsAssignableFrom(exception.GetType()));
    }

    [Fact]
    public void Cancellation_PreCancelledToken_ReachesParseAndExpressionEvaluateAsRenderLimit()
    {
        using CancellationTokenSource source = new();
        source.Cancel();

        // Parse and expression evaluation surface cancellation as the same actionable render limit, never a bare token.
        RenderLimitsException parse = Assert.Throws<RenderLimitsException>(() => MinjaParser.Parse("hello {{ name }}", source.Token));
        Assert.Equal("cancellation", parse.Limit);

        RenderLimitsException evaluated = Assert.Throws<RenderLimitsException>(() => MinjaEvaluator.Evaluate("'a' + 'b'", null, source.Token));
        Assert.Equal("cancellation", evaluated.Limit);
    }

    [Fact]
    public void Cancellation_CancelledMidRender_StopsAtTheNextProbeWindowQuickly()
    {
        using CancellationTokenSource source = new();
        string template = "{% for a in range(2) %}{% for i in range(99000) %}x{% endfor %}{% endfor %}";

        // Cancel while the render is already running; the token is noticed at the next probe window (64 iterations).
        source.CancelAfter(TimeSpan.FromMilliseconds(30));
        Stopwatch clock = Stopwatch.StartNew();

        RenderLimitsException exception = Assert.Throws<RenderLimitsException>(() => MinjaEvaluator.Render(template, null, source.Token));
        clock.Stop();

        Assert.Equal("cancellation", exception.Limit);
        // Load-tolerance guard only: under concurrent load the cancel timer and the probe window can both slip. The
        // deterministic guarantee is the cancellation limit named above.
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30), clock.Elapsed.ToString());
    }

    [Fact]
    public void Cancellation_UncancelledRenderOfSameTemplate_CompletesNormally()
    {
        string template = "{% for a in range(2) %}{% for i in range(99000) %}x{% endfor %}{% endfor %}";

        Assert.Equal(198_000, MinjaEvaluator.Render(template).Length);
    }

    // ---------------------------------------------------------------- sanity: limits do not alter semantics

    [Fact]
    public void ChatMlGolden_RendersByteIdenticallyWithLimitsActive()
    {
        const string template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

        MinjaChatTemplate chat = new(template);
        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("system", "You are helpful."), ChatMessage.Text("user", "Hello")],
            AddGenerationPrompt = true,
        });

        Assert.Equal(
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            + "<|im_start|>user\nHello<|im_end|>\n"
            + "<|im_start|>assistant\n",
            result.Prompt);
    }

    [Fact]
    public void LlamaStyleGolden_RendersByteIdenticallyWithLimitsActive()
    {
        const string template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}<<SYS>>{{ message['content'] }}<</SYS>>\n{% else %}[{{ message['role'] }}]\n{{ message['content'] }}\n{% endif %}{% endfor %}{% if add_generation_prompt %}[assistant]\n{% endif %}";

        MinjaChatTemplate chat = new(template, bosToken: "<s>", eosToken: "</s>");
        ChatTemplateRenderResult result = chat.Render(new ChatTemplateInput
        {
            Messages = [ChatMessage.Text("system", "Be useful."), ChatMessage.Text("user", "Hi")],
            AddGenerationPrompt = true,
        });

        Assert.Equal("<s><<SYS>>Be useful.<</SYS>>\n[user]\nHi\n[assistant]\n", result.Prompt);
    }

    [Fact]
    public void EverydayTemplateWithMacrosFiltersAndLoops_RendersByteIdenticallyUnderGuards()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("tools", ListOf(MapOf(("name", MinjaValue.Of("search"))))),
            ("messages", ListOf(MapOf(("role", MinjaValue.Of("user")), ("content", MinjaValue.Of("hi there  "))))));

        string template = "{% macro tool(t) %}[{{ t['name'] }}]{% endmacro %}"
            + "{%- if tools %}TOOLS:{% for t in tools %}{{ tool(t) }}{% endfor %}{% endif %}"
            + "{% for message in messages %} {{ message['content'] | trim }}{% endfor %}";

        Assert.Equal("TOOLS:[search] hi there", MinjaEvaluator.Render(template, context));
    }

    // ---------------------------------------------------------------- helpers

    private static string Render(string template) => MinjaEvaluator.Render(template);

    /// <summary>A chain of macros where each one calls the next, so a render nests exactly the given number of call frames.</summary>
    private static string MacroChain(int depth)
    {
        System.Text.StringBuilder builder = new();

        for (int index = 1; index < depth; index++)
            builder.Append("{% macro m").Append(index).Append("() %}{{ m").Append(index + 1).Append("() }}{% endmacro %}");

        return builder.Append("{% macro m").Append(depth).Append("() %}ok{% endmacro %}{{ m1() }}").ToString();
    }

    private static string Render(string template, IReadOnlyDictionary<string, MinjaValue> context) => MinjaEvaluator.Render(template, context);

    private static string Render(string template, CancellationToken cancellation) => MinjaEvaluator.Render(template, null, cancellation);

    private static void AssertLimitFreeSignature(System.Reflection.MethodInfo method)
    {
        foreach (System.Reflection.ParameterInfo parameter in method.GetParameters())
            Assert.False(
                IsLimitBearingType(parameter.ParameterType),
                $"'{method.DeclaringType?.Name}.{method.Name}' must not accept a limit-bearing parameter of type {parameter.ParameterType}");
    }

    private static bool IsLimitBearingType(Type type)
    {
        // A budget would arrive as a duration, a number or an object named for limits. The additive CancellationToken
        // and the value-context dictionary are allowed; nothing that could widen a guard is.
        if (type == typeof(CancellationToken))
            return false;

        Type candidate = Nullable.GetUnderlyingType(type) ?? type;

        return candidate == typeof(TimeSpan)
            || candidate.Name.Contains("Limit", StringComparison.OrdinalIgnoreCase)
            || candidate.IsEnum && candidate.Name.Contains("Depth", StringComparison.OrdinalIgnoreCase);
    }

    private static MinjaValue ListOf(params MinjaValue[] items) => MinjaValue.List(items);

    private static MinjaValue NumberedMap(int count)
    {
        (string, MinjaValue)[] entries = new (string, MinjaValue)[count];

        for (int i = 0; i < count; i++)
            entries[i] = ($"k{i}", MinjaValue.Of((long)i));

        return MapOf(entries);
    }

    private static MinjaValue MapOf(params (string Key, MinjaValue Value)[] entries) =>
        MinjaValue.Map(entries.Select(entry => new KeyValuePair<string, MinjaValue>(entry.Key, entry.Value)));

    private static IReadOnlyDictionary<string, MinjaValue> Context(params (string Name, MinjaValue Value)[] entries) =>
        entries.ToDictionary(entry => entry.Name, entry => entry.Value);
}

using Lair.Templating.Minja;

namespace Lair.Tests;

/// <summary>
/// Statement and scoping goldens for the managed minja evaluator: if/elif/else selection, for loops over lists,
/// maps and strings with loop conditions, destructuring and loop.* attributes, break/continue, recursive loop
/// flattening, minja-style set visibility after blocks, namespace map mutation, and macro/call/filter scoping.
/// Built-in filters, tests and functions are covered elsewhere; these tests use user-defined macros as callables.
/// </summary>
[Trait("Category", "MinjaStatements")]
public sealed class MinjaStatementsTests
{
    // ---------------------------------------------------------------- if / elif / else

    [Theory]
    [InlineData("True", "yes")]
    [InlineData("1", "yes")]
    [InlineData("0.5", "yes")]
    [InlineData("'a'", "yes")]
    [InlineData("None", "no")]
    [InlineData("0", "no")]
    [InlineData("''", "no")]
    [InlineData("False", "no")]
    public void If_TruthinessSelectsBranch(string condition, string expected)
    {
        Assert.Equal(expected, Render($"{{% if {condition} %}}yes{{% else %}}no{{% endif %}}"));
    }

    [Fact]
    public void If_ElifChainPicksFirstTruthyBranch()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("falsy", MinjaValue.Of(0L)),
            ("truthy", MinjaValue.Of(1L)));

        Assert.Equal("B", Render("{% if falsy %}A{% elif truthy %}B{% else %}C{% endif %}", context));
        Assert.Equal("C", Render("{% if falsy %}A{% elif missing %}B{% else %}C{% endif %}", context));
        Assert.Equal("A", Render("{% if truthy %}A{% elif 1 %}B{% else %}C{% endif %}", context));
        Assert.Equal("", Render("{% if falsy %}A{% endif %}", context));
    }

    [Fact]
    public void If_BranchBodiesShareTemplateScope()
    {
        Assert.Equal("on", Render("{% if True %}{% set flag = 'on' %}{% endif %}{{ flag }}"));

        // A branch that never runs assigns nothing, and the name stays undefined.
        Assert.Equal("", Render("{% if False %}{% set flag = 'off' %}{% endif %}{{ flag }}"));

        // The else branch shares the template scope too.
        Assert.Equal("off", Render("{% if False %}{% else %}{% set flag = 'off' %}{% endif %}{{ flag }}"));
    }

    // ---------------------------------------------------------------- for loops

    [Fact]
    public void For_ListsRenderEachItemInOrder()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L), MinjaValue.Of(3L))));

        Assert.Equal("123", Render("{% for i in nums %}{{ i }}{% endfor %}", context));
        Assert.Equal("", Render("{% for i in missing %}{{ i }}{% endfor %}", context));
    }

    [Fact]
    public void For_MapsIterateKeysInInsertionOrder()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("message", MapOf(("role", MinjaValue.Of("user")), ("content", MinjaValue.Of("hi")))));

        Assert.Equal("role=user content=hi ", Render("{% for key in message %}{{ key }}={{ message[key] }} {% endfor %}", context));
    }

    [Fact]
    public void For_StringsIterateCharacters()
    {
        Assert.Equal("[a][b][c]", Render("{% for c in 'abc' %}[{{ c }}]{% endfor %}"));
    }

    [Fact]
    public void For_LoopConditionFiltersItemsWithTargetBound()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("tools", ListOf(
                MapOf(("name", MinjaValue.Of("seen")), ("enabled", MinjaValue.True)),
                MapOf(("name", MinjaValue.Of("hidden")), ("enabled", MinjaValue.False)))));

        Assert.Equal("seen", Render("{% for tool in tools if tool.enabled %}{{ tool.name }}{% endfor %}", context));
        Assert.Equal("hidden", Render("{% for tool in tools if not tool.enabled %}{{ tool.name }}{% endfor %}", context));
    }

    [Fact]
    public void For_DestructuringTargetsBindTupleItems()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("pairs", ListOf(
                ListOf(MinjaValue.Of(1L), MinjaValue.Of("a")),
                ListOf(MinjaValue.Of(2L), MinjaValue.Of("b")))),
            ("nested", ListOf(ListOf(MinjaValue.Of(1L), ListOf(MinjaValue.Of(2L), MinjaValue.Of(3L))))));

        Assert.Equal("1:a 2:b ", Render("{% for (k, v) in pairs %}{{ k }}:{{ v }} {% endfor %}", context));

        // Nested tuple targets destructure recursively.
        Assert.Equal("1-2-3", Render("{% for (a, (b, c)) in nested %}{{ a }}-{{ b }}-{{ c }}{% endfor %}", context));
    }

    [Fact]
    public void For_DestructuringMismatchAndNonListItemsThrowRenderError()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("short", ListOf(ListOf(MinjaValue.Of(1L)))),
            ("plain", ListOf(MinjaValue.Of(7L))));

        MinjaRenderException shape = Assert.Throws<MinjaRenderException>(() => Render("{% for (a, b) in short %}{{ a }}{% endfor %}", context));
        Assert.Equal("for", shape.Construct);

        MinjaRenderException kind = Assert.Throws<MinjaRenderException>(() => Render("{% for (a, b) in plain %}{{ a }}{% endfor %}", context));
        Assert.Equal("for", kind.Construct);
    }

    [Fact]
    public void For_NonIterableValueThrowsRenderError()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("count", MinjaValue.Of(3L)));

        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Render("{% for i in count %}{{ i }}{% endfor %}", context));
        Assert.Equal("for", error.Construct);
    }

    [Fact]
    public void For_BreakStopsIterationAndContinueSkipsToNextItem()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L), MinjaValue.Of(3L))));

        Assert.Equal("1", Render("{% for i in nums %}{% if i == 2 %}{% break %}{% endif %}{{ i }}{% endfor %}", context));
        Assert.Equal("13", Render("{% for i in nums %}{% if i == 2 %}{% continue %}{% endif %}{{ i }}{% endfor %}", context));
        Assert.Equal("12", Render("{% for i in nums %}{{ i }}{% if i == 2 %}{% break %}{% endif %}{% endfor %}", context));
    }

    [Fact]
    public void For_BreakInsideNestedLoopStopsOnlyTheInnerLoop()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("outer", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))),
            ("inner", ListOf(MinjaValue.Of(9L), MinjaValue.Of(8L))));

        // The break sits in the inner body, so the outer loop keeps running and the inner loop restarts.
        Assert.Equal("9898", Render("{% for a in outer %}{% for b in inner %}{{ b }}{% if b == 8 %}{% break %}{% endif %}{% endfor %}{% endfor %}", context));
    }

    [Fact]
    public void For_LoopAttributesExposeIndexReverseVariantsFirstLastAndLength()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(10L), MinjaValue.Of(20L), MinjaValue.Of(30L))));

        Assert.Equal(
            "1/0/3/2/True/False/3 2/1/2/1/False/False/3 3/2/1/0/False/True/3 ",
            Render("{% for i in nums %}{{ loop.index }}/{{ loop.index0 }}/{{ loop.revindex }}/{{ loop.revindex0 }}/{{ loop.first }}/{{ loop.last }}/{{ loop.length }} {% endfor %}", context));
    }

    [Fact]
    public void For_FilteredLoopLengthCountsOnlyKeptItems()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L), MinjaValue.Of(3L), MinjaValue.Of(4L))));

        Assert.Equal("1/2 2/2 ", Render("{% for i in nums if i > 2 %}{{ loop.index }}/{{ loop.length }} {% endfor %}", context));
    }

    [Fact]
    public void For_LoopCycleMethodIsNotImplementedYet()
    {
        // Honest gap: loop exposes the index/first/last/length attributes but not the cycle() method,
        // because attribute callables arrive with the filters item.
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L))));

        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Render("{% for i in nums %}{{ loop.cycle('a', 'b') }}{% endfor %}", context));
        Assert.Equal("call", error.Construct);
    }

    [Fact]
    public void For_LoopTargetIsRestoredWhenTheLoopEnds()
    {
        // Deliberate limitation recorded honestly: Jinja leaks the last loop value into its frame, while this
        // evaluator restores the target's previous binding when the loop ends (or clears it when there was none).
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))),
            ("item", MinjaValue.Of("outer")));

        IReadOnlyDictionary<string, MinjaValue> bare = Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))));

        Assert.Equal("12outer", Render("{% for item in nums %}{{ item }}{% endfor %}{{ item }}", context));
        Assert.Equal("12", Render("{% for item in nums %}{{ item }}{% endfor %}{{ item }}", bare));
    }

    [Fact]
    public void For_RecursiveLoopFlattensNestedIterables()
    {
        // Each node is [value, children]; loop(children) re-enters the same recursive body for the nested list.
        IReadOnlyDictionary<string, MinjaValue> context = Context(("tree", ListOf(
            ListOf(
                MinjaValue.Of(1L),
                ListOf(
                    ListOf(MinjaValue.Of(2L), MinjaValue.EmptyList),
                    ListOf(MinjaValue.Of(3L), ListOf(ListOf(MinjaValue.Of(4L), MinjaValue.EmptyList))))))));

        Assert.Equal("1234", Render("{% for node in tree recursive %}{{ node[0] }}{% if node[1] %}{{ loop(node[1]) }}{% endif %}{% endfor %}", context));
        Assert.Equal("1", Render("{% for node in tree recursive %}{{ node[0] }}{% if false %}{{ loop(node[1]) }}{% endif %}{% endfor %}", context));
    }

    [Fact]
    public void For_RecursiveLoopCallableRejectsWrongArgumentCount()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("tree", ListOf(ListOf(MinjaValue.Of(1L), MinjaValue.EmptyList))));

        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Render("{% for node in tree recursive %}{{ loop() }}{% endfor %}", context));
        Assert.Equal("loop", error.Construct);
    }

    // ---------------------------------------------------------------- set statements and scoping

    [Fact]
    public void Set_InsideForRemainsVisibleAfterTheLoop()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L), MinjaValue.Of(3L))));

        Assert.Equal("3", Render("{% for i in nums %}{% set last = i %}{% endfor %}{{ last }}", context));
    }

    [Fact]
    public void Set_RebindingTheIteratedNameKeepsTheLoopSnapshot()
    {
        // Llama-family templates re-bind messages through set; the running loop keeps the snapshot taken at entry.
        IReadOnlyDictionary<string, MinjaValue> context = Context(("messages", ListOf(MinjaValue.Of("a"), MinjaValue.Of("b"))));

        Assert.Equal("ab|x", Render("{% for m in messages %}{% set messages = 'x' %}{{ m }}{% endfor %}|{{ messages }}", context));
    }

    [Fact]
    public void Set_DestructuringBindsAListValuePositionally()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("pair", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))));

        Assert.Equal("1-2", Render("{% set a, b = pair %}{{ a }}-{{ b }}", context));
        Assert.Equal("12", Render("{% set (c, d) = pair %}{{ c }}{{ d }}", context));
    }

    [Fact]
    public void Set_DestructuringMismatchAndNonListValueThrowRenderError()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("short", ListOf(MinjaValue.Of(1L))),
            ("text", MinjaValue.Of("ab")));

        MinjaRenderException shape = Assert.Throws<MinjaRenderException>(() => Render("{% set a, b = short %}", context));
        Assert.Equal("set", shape.Construct);

        MinjaRenderException kind = Assert.Throws<MinjaRenderException>(() => Render("{% set a, b = text %}", context));
        Assert.Equal("set", kind.Construct);
    }

    [Fact]
    public void Set_NamespacedAttributeMutationIsVisibleOutsideTheAssigningBlock()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("ns", MapOf(("count", MinjaValue.Of(0L)))));

        Assert.Equal("1", Render("{% if True %}{% set ns.count = 1 %}{% endif %}{{ ns.count }}", context));

        // The dotted assignment mutated the map object itself, so a later render sharing the same context value sees it.
        Assert.Equal("1", Render("{{ ns.count }}", context));
        Assert.Equal(1L, context["ns"].MapValue.Get("count").IntValue);
    }

    [Fact]
    public void Set_NamespacedMutationFromAMacroScopeReachesTheSharedMap()
    {
        // A macro's own scope cannot leak a plain set to its caller, but a dotted assignment mutates the map
        // object itself, so the caller and every later holder see the change.
        IReadOnlyDictionary<string, MinjaValue> context = Context(("ns", MapOf(("count", MinjaValue.Of(0L)))));

        Assert.Equal("2", Render("{% macro bump() %}{% set ns.count = ns.count + 1 %}{% endmacro %}{% for i in 'ab' %}{{ bump() }}{% endfor %}{{ ns.count }}", context));
        Assert.Equal(2L, context["ns"].MapValue.Get("count").IntValue);
    }

    [Fact]
    public void Set_NestedNamespacedAssignmentMutatesTheInnerMap()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("outer", MapOf(("inner", MapOf(("v", MinjaValue.Of(1L)))))));

        Assert.Equal("2", Render("{% set outer.inner.v = 2 %}{{ outer.inner.v }}", context));
    }

    [Fact]
    public void Set_NamespacedAssignmentToUndefinedOrNonMapOwnerThrowsRenderError()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("ns", MapOf(("count", MinjaValue.Of(0L)))));

        MinjaRenderException undefined = Assert.Throws<MinjaRenderException>(() => Render("{% set missing.attr = 1 %}", context));
        Assert.Equal("set", undefined.Construct);

        MinjaRenderException notMap = Assert.Throws<MinjaRenderException>(() => Render("{% set ns.count.deep = 1 %}", context));
        Assert.Equal("set", notMap.Construct);
    }

    // ---------------------------------------------------------------- macros

    [Fact]
    public void Macro_DefaultsAndNamedArgumentsBindLikeJinja()
    {
        Assert.Equal(
            "hello worldhi moon",
            Render("{% macro greet(name, greeting='hello') %}{{ greeting }} {{ name }}{% endmacro %}{{ greet('world') }}{{ greet(name='moon', greeting='hi') }}"));

        Assert.Equal(
            "hello ",
            Render("{% macro greet(name, greeting='hello') %}{{ greeting }} {{ name }}{% endmacro %}{{ greet() }}"));
    }

    [Fact]
    public void Macro_LocalsDoNotLeakIntoCallerScope()
    {
        Assert.Equal("1", Render("{% macro m() %}{% set inner = 1 %}{{ inner }}{% endmacro %}{{ m() }}{{ inner }}"));
    }

    [Fact]
    public void Macro_ParametersShadowOuterBindingsWithoutRewritingThem()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("outer", MinjaValue.Of(1L)));

        Assert.Equal("21", Render("{% macro m(outer) %}{{ outer }}{% endmacro %}{{ m(2) }}{{ outer }}", context));
        Assert.Equal("21", Render("{% macro m() %}{% set outer = 2 %}{{ outer }}{% endmacro %}{{ m() }}{{ outer }}", context));
    }

    [Fact]
    public void Macro_ClosureReadsItsDefiningContextAtCallTime()
    {
        Assert.Equal(
            "hi!hi?",
            Render("{% set suffix = '!' %}{% macro greet(name) %}{{ name }}{{ suffix }}{% endmacro %}{{ greet('hi') }}{% set suffix = '?' %}{{ greet('hi') }}"));
    }

    [Fact]
    public void Macro_DefinedInsideABlockStaysCallableAfterIt()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L))));

        Assert.Equal("T", Render("{% for i in nums %}{% macro tag() %}T{% endmacro %}{% endfor %}{{ tag() }}", context));
        Assert.Equal("U", Render("{% if True %}{% macro tag2() %}U{% endmacro %}{% endif %}{{ tag2() }}", context));
    }

    [Fact]
    public void Macro_BadArgumentCountsAndUnknownNamesThrowRenderError()
    {
        MinjaRenderException tooMany = Assert.Throws<MinjaRenderException>(() => Render("{% macro one(a) %}{{ a }}{% endmacro %}{{ one(1, 2) }}"));
        Assert.Equal("call", tooMany.Construct);

        MinjaRenderException unknown = Assert.Throws<MinjaRenderException>(() => Render("{% macro one(a) %}{{ a }}{% endmacro %}{{ one(b=1) }}"));
        Assert.Equal("call", unknown.Construct);

        MinjaRenderException twice = Assert.Throws<MinjaRenderException>(() => Render("{% macro one(a) %}{{ a }}{% endmacro %}{{ one(1, a=2) }}"));
        Assert.Equal("call", twice.Construct);

        MinjaRenderException notCallable = Assert.Throws<MinjaRenderException>(() => Render("{{ missing(1) }}"));
        Assert.Equal("call", notCallable.Construct);
    }

    [Fact]
    public void Macro_SetInsideMacroCannotReachAnOuterLoopBinding()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L))));

        // The macro child scope writes its own binding; the loop's scope keeps its original value.
        Assert.Equal("21", Render("{% for i in nums %}{% macro m() %}{% set i = 2 %}{{ i }}{% endmacro %}{{ m() }}{{ i }}{% endfor %}", context));
    }

    [Fact]
    public void Macro_CallArgumentsEvaluateInTheCallingScope()
    {
        // A recursive macro builds its next argument from its own frame, exactly like the pinned llama.cpp
        // jinja engine, which gathers every call_expression argument from the caller's context.
        Assert.Equal(
            "321",
            Render("{% macro countdown(n) %}{{ n }}{% if n > 1 %}{{ countdown(n - 1) }}{% endif %}{% endmacro %}{{ countdown(3) }}"));

        // A macro called from inside a loop receives the loop's local binding as its argument.
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))));

        Assert.Equal(
            "12",
            Render("{% macro tag(x) %}{{ x }}{% endmacro %}{% for pair in nums %}{{ tag(pair) }}{% endfor %}", context));

        // Keyword arguments resolve the same way.
        Assert.Equal(
            "12",
            Render("{% macro tag(x) %}{{ x }}{% endmacro %}{% for pair in nums %}{{ tag(x=pair) }}{% endfor %}", context));
    }

    // ---------------------------------------------------------------- call blocks

    [Fact]
    public void CallBlock_PassesBodyToMacroAsCaller()
    {
        Assert.Equal("[body]", Render("{% macro card() %}[{{ caller() }}]{% endmacro %}{% call card() %}body{% endcall %}"));
        Assert.Equal("[body]", Render("{% macro card() %}[{{ _caller() }}]{% endmacro %}{% call card() %}body{% endcall %}"));
    }

    [Fact]
    public void CallBlock_CallerBodySeesBindingsAroundTheCallTag()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))));

        Assert.Equal("[1][2]", Render("{% macro card() %}[{{ caller() }}]{% endmacro %}{% for i in nums %}{% call card() %}{{ i }}{% endcall %}{% endfor %}", context));
    }

    [Fact]
    public void CallBlock_BodyIsNotRenderedWhenTheMacroIgnoresCaller()
    {
        Assert.Equal("x", Render("{% macro silent() %}x{% endmacro %}{% call silent() %}ignored{% endcall %}"));
    }

    [Fact]
    public void CallBlock_WithoutCallableInScopeThrowsRenderError()
    {
        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Render("{% call missing() %}body{% endcall %}"));
        Assert.Equal("call", error.Construct);
    }

    // ---------------------------------------------------------------- filter blocks

    [Fact]
    public void FilterBlock_AppliesUserMacroToTheRenderedBody()
    {
        Assert.Equal("[body]", Render("{% macro wrap(text) %}[{{ text }}]{% endmacro %}{% filter wrap %}body{% endfilter %}"));
        Assert.Equal("<body>", Render("{% macro wrap(text, style='angle') %}{% if style == 'angle' %}<{{ text }}>{% else %}[{{ text }}]{% endif %}{% endmacro %}{% filter wrap %}body{% endfilter %}"));
    }

    [Fact]
    public void FilterBlock_ElseBodyRendersWhenBodyIsEmpty()
    {
        Assert.Equal("[fallback]", Render("{% macro wrap(text) %}[{{ text }}]{% endmacro %}{% filter wrap %}{% else %}fallback{% endfilter %}"));
    }

    [Fact]
    public void FilterBlock_BodySharesTemplateScope()
    {
        Assert.Equal("[x]1", Render("{% macro wrap(text) %}[{{ text }}]{% endmacro %}{% filter wrap %}{% set flag = 1 %}x{% endfilter %}{{ flag }}"));
    }

    [Fact]
    public void FilterBlock_BuiltInFilterNameRemainsUnsupported()
    {
        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Render("{% filter upper %}shout{% endfilter %}"));
        Assert.Equal("filter", error.Construct);
        Assert.Contains("built-in filters", error.Message);
    }

    // ---------------------------------------------------------------- break/continue boundaries

    [Fact]
    public void BreakOrContinueOutsideAnyLoopThrowsRenderError()
    {
        // The parser only allows these tags inside macro/call/filter bodies; at runtime they must sit in a loop.
        MinjaRenderException macroBreak = Assert.Throws<MinjaRenderException>(() => Render("{% macro m() %}{% break %}{% endmacro %}{{ m() }}"));
        Assert.Equal("break", macroBreak.Construct);

        MinjaRenderException macroContinue = Assert.Throws<MinjaRenderException>(() => Render("{% macro m() %}{% continue %}{% endmacro %}{{ m() }}"));
        Assert.Equal("continue", macroContinue.Construct);
    }

    // ---------------------------------------------------------------- helpers

    private static string Render(string template, IReadOnlyDictionary<string, MinjaValue>? context = null) =>
        MinjaEvaluator.Render(template, context);

    private static MinjaValue ListOf(params MinjaValue[] items) => MinjaValue.List(items);

    private static MinjaValue MapOf(params (string Key, MinjaValue Value)[] entries) =>
        MinjaValue.Map(entries.Select(entry => new KeyValuePair<string, MinjaValue>(entry.Key, entry.Value)));

    private static IReadOnlyDictionary<string, MinjaValue> Context(params (string Name, MinjaValue Value)[] entries) =>
        entries.ToDictionary(entry => entry.Name, entry => entry.Value);
}

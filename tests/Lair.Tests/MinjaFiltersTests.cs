using Lair.Templating.Minja;

namespace Lair.Tests;

/// <summary>
/// Goldens for the built-in minja filter, test and function namespace: collection shape and transformation
/// (length/count, items, dictsort, join, map, sort), selection filters with truthiness, named tests, attribute
/// paths and extra arguments, conversion and escaping (tojson, trim, e/escape), chat-template helpers (range,
/// namespace, joiner) and the deliberate template failure raise_exception. Each group also pins how bad or
/// unknown built-ins fail actionably instead of silently succeeding.
/// </summary>
[Trait("Category", "MinjaFilters")]
public sealed class MinjaFiltersTests
{
    // ---------------------------------------------------------------- length / count

    [Theory]
    [InlineData("'abc' | length", 3)]
    [InlineData("'abc' | count", 3)]
    [InlineData("'' | length", 0)]
    [InlineData("nums | length", 3)]
    [InlineData("nums | count", 3)]
    [InlineData("record | length", 2)]
    [InlineData("missing | length", 0)]
    public void Length_Count_MatchesElementCount(string expression, int expected)
    {
        Assert.Equal(expected, (int)Eval(expression, CollectionContext()).IntValue);
    }

    [Fact]
    public void Length_UndefinedTakesTheEmptySequenceReadingForLoopsAlreadyGiveIt()
    {
        // Undefined is none here and a for loop iterates it as empty, so its length is zero rather than an error.
        Assert.Equal(0L, Eval("missing | count").IntValue);
        Assert.Equal(string.Empty, Render("{% for item in missing %}x{% endfor %}"));
    }

    [Theory]
    [InlineData("5 | length")]
    [InlineData("'a' | length('extra')")]
    public void Length_BadInputOrArgumentsFailActionably(string expression)
    {
        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Eval(expression));

        Assert.Equal("length", error.Construct);
        Assert.True(error.Position >= 0);
    }

    // ---------------------------------------------------------------- tojson

    [Fact]
    public void ToJson_KeepsMapInsertionOrderAndNestedShapes()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("record", MapOf(
            ("b", MinjaValue.Of(2L)),
            ("a", ListOf(MinjaValue.Of(1L), MapOf(("k", MinjaValue.Of("v"))))),
            ("c", MinjaValue.EmptyList))));

        Assert.Equal("{\"b\": 2, \"a\": [1, {\"k\": \"v\"}], \"c\": []}", Render("{{ record | tojson }}", context));
    }

    [Fact]
    public void ToJson_NoneUndefinedAndBooleansFollowTheValueModel()
    {
        // The value model has no distinct undefined: an unbound name dumps as null exactly like the None literal.
        Assert.Equal("null null true false", Render("{{ missing | tojson }} {{ None | tojson }} {{ True | tojson }} {{ False | tojson }}"));
    }

    [Fact]
    public void ToJson_UsesPythonStringAndNumberEscaping()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("text", MinjaValue.Of("caf\u00e9 \"x\"\n")),
            ("number", MinjaValue.Of(1e-05)));

        Assert.Equal("\"caf\\u00e9 \\\"x\\\"\\n\"", Render("{{ text | tojson }}", context));
        Assert.Equal("1e-05", Render("{{ number | tojson }}", context));
    }

    [Fact]
    public void ToJson_RejectsArguments()
    {
        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Eval("'a' | tojson(indent=2)"));

        Assert.Equal("tojson", error.Construct);
        Assert.Contains("takes no arguments", error.Message);
    }

    // ---------------------------------------------------------------- e / escape

    [Fact]
    public void Escape_AndItsAliasProduceHtmlEntities()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("text", MinjaValue.Of("<a href=\"x\">'b' & c")));
        string expected = "&lt;a href=&#34;x&#34;&gt;&#39;b&#39; &amp; c";

        Assert.Equal(expected, Render("{{ text | e }}", context));
        Assert.Equal(expected, Render("{{ text | escape }}", context));
    }

    [Fact]
    public void Escape_PassesOtherTextThroughAndRejectsArguments()
    {
        Assert.Equal("plain 123", Render("{{ 'plain 123' | e }}"));

        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Eval("'a' | escape(quote=True)"));

        Assert.Contains("escape", error.Message);
    }

    // ---------------------------------------------------------------- trim

    [Theory]
    [InlineData("'  hi  ' | trim", "hi")]
    [InlineData("'\\t hi \\n' | trim", "hi")]
    [InlineData("'xhix' | trim('x')", "hi")]
    [InlineData("'abcbac' | trim('abc')", "")]
    public void Trim_DefaultWhitespaceAndCustomCharacterSet(string expression, string expected) => Assert.Equal(expected, Eval(expression).StringValue);

    [Fact]
    public void Trim_RejectsExtraOrUnknownArguments()
    {
        MinjaRenderException extra = Assert.Throws<MinjaRenderException>(() => Eval("'a' | trim('x', 'y')"));
        Assert.Contains("trim", extra.Message);

        MinjaRenderException named = Assert.Throws<MinjaRenderException>(() => Eval("'a' | trim(chars='x')"));
        Assert.Contains("named argument", named.Message);
    }

    // ---------------------------------------------------------------- items / dictsort

    [Fact]
    public void Items_MapBecomesKeyValuePairLists()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("record", MapOf(("b", MinjaValue.Of(2L)), ("a", MinjaValue.Of("v")))));

        Assert.Equal("[['b', 2], ['a', 'v']]", Render("{{ record | items }}", context));
        Assert.Equal("[]", Render("{{ missing | items }}"));
    }

    [Fact]
    public void Items_NonMapInputFailsActionably()
    {
        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Render("{{ nums | items }}", CollectionContext()));

        Assert.Equal("filter", error.Construct);
        Assert.Contains("items needs a map value", error.Message);
    }

    [Fact]
    public void DictSort_SortsPairsByKeyValueAndCaseMode()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("record", MapOf(
            ("B", MinjaValue.Of(2L)),
            ("a", MinjaValue.Of(1L)))));

        // Key sorting is case-insensitive unless the caller asks for exact ordering.
        Assert.Equal("[['a', 1], ['B', 2]]", Render("{{ record | dictsort }}", context));
        Assert.Equal("[['B', 2], ['a', 1]]", Render("{{ record | dictsort(case_sensitive=True) }}", context));
        Assert.Equal("[['a', 1], ['B', 2]]", Render("{{ record | dictsort(by='value') }}", context));
    }

    [Fact]
    public void DictSort_RejectsUnknownSortModeAndArguments()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("record", MapOf(("k", MinjaValue.Of(1L)))));

        MinjaRenderException mode = Assert.Throws<MinjaRenderException>(() => Render("{{ record | dictsort(by='nope') }}", context));
        Assert.Contains("dictsort sorts by 'key' or 'value'", mode.Message);

        MinjaRenderException named = Assert.Throws<MinjaRenderException>(() => Render("{{ record | dictsort(reverse=True) }}", context));
        Assert.Contains("named argument 'reverse'", named.Message);

        MinjaRenderException notMap = Assert.Throws<MinjaRenderException>(() => Render("{{ nums | dictsort }}", CollectionContext()));
        Assert.Contains("dictsort needs a map value", notMap.Message);
    }

    // ---------------------------------------------------------------- join

    [Fact]
    public void Join_DefaultAndExplicitDelimiterOverListsStringsAndMaps()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))),
            ("record", MapOf(("a", MinjaValue.Of(1L)), ("b", MinjaValue.Of(2L)))));

        Assert.Equal("12", Render("{{ nums | join }}", context));
        Assert.Equal("1, 2", Render("{{ nums | join(', ') }}", context));
        Assert.Equal("a-b-c", Render("{{ 'abc' | join('-') }}"));
        Assert.Equal("ab", Render("{{ 'ab' | join }}"));

        // Iterating a map joins its keys in insertion order, matching for-loop iteration.
        Assert.Equal("a,b", Render("{{ record | join(',') }}", context));
    }

    [Fact]
    public void Join_ReadsAttributePathsIncludingDottedAndNegativeIndexSegments()
    {
        IReadOnlyDictionary<string, MinjaValue> messages = Context(("messages", ListOf(
            MapOf(("role", MinjaValue.Of("user")), ("content", MinjaValue.Of("hi"))),
            MapOf(("role", MinjaValue.Of("assistant")), ("content", MinjaValue.Of("bye"))))));

        Assert.Equal("hi, bye", Render("{{ messages | join(', ', attribute='content') }}", messages));
        Assert.Equal("user/assistant", Render("{{ messages | join('/', attribute='role') }}", messages));

        IReadOnlyDictionary<string, MinjaValue> pairs = Context(("pairs", ListOf(
            ListOf(MinjaValue.Of("a"), MinjaValue.Of("b")),
            ListOf(MinjaValue.Of("c"), MinjaValue.Of("d")))));

        Assert.Equal("b,d", Render("{{ pairs | join(',', attribute='-1') }}", pairs));

        IReadOnlyDictionary<string, MinjaValue> nested = Context(("records", ListOf(
            MapOf(("meta", MapOf(("label", MinjaValue.Of("x"))))),
            MapOf(("meta", MapOf(("label", MinjaValue.Of("y"))))))));

        Assert.Equal("x|y", Render("{{ records | join('|', attribute='meta.label') }}", nested));
    }

    [Fact]
    public void Join_MissingAttributeRendersAsEmptyTextAndBadArgumentsFail()
    {
        // A missing attribute is none in this value model, and none renders as empty text.
        IReadOnlyDictionary<string, MinjaValue> records = Context(("records", ListOf(
            MapOf(("label", MinjaValue.Of("x"))),
            MapOf(("other", MinjaValue.Of(1L))))));

        Assert.Equal("x,", Render("{{ records | join(',', attribute='label') }}", records));

        IReadOnlyDictionary<string, MinjaValue> nums = Context(("nums", ListOf(MinjaValue.Of(1L))));

        MinjaRenderException extra = Assert.Throws<MinjaRenderException>(() => Render("{{ nums | join(',', ';') }}", nums));
        Assert.Contains("join takes at most 1 positional argument", extra.Message);

        MinjaRenderException named = Assert.Throws<MinjaRenderException>(() => Render("{{ nums | join(sep=',') }}", nums));
        Assert.Contains("named argument", named.Message);
    }

    // ---------------------------------------------------------------- joiner

    [Fact]
    public void Joiner_BoundBySetEmitsEmptyTextFirstThenTheSeparator()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L), MinjaValue.Of(3L))));

        Assert.Equal("1, 2, 3", Render("{% set comma = joiner(', ') %}{% for n in nums %}{{ comma() }}{{ n }}{% endfor %}", context));
        Assert.Equal("", Render("{% set comma = joiner(', ') %}{% for n in empty %}{{ comma() }}{{ n }}{% endfor %}", Context(("empty", MinjaValue.EmptyList))));
    }

    [Fact]
    public void Joiner_EachBindingStartsItsOwnSequence()
    {
        Assert.Equal("a-b", Render("{% set dash = joiner('-') %}{{ dash() }}a{{ dash() }}b"));
        Assert.Equal("c", Render("{% set dash = joiner('-') %}{{ dash() }}c"));
    }

    [Fact]
    public void Joiner_FailsActionablyWhenCalledWithoutABinding()
    {
        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Render("{{ joiner(', ')() }}"));

        Assert.Equal("call", error.Construct);
        Assert.Contains("unknown function 'joiner'", error.Message);
    }

    // ---------------------------------------------------------------- map / sort

    [Fact]
    public void Map_ReadsAttributePathsAndAppliesDefaultValues()
    {
        IReadOnlyDictionary<string, MinjaValue> records = Context(("records", ListOf(
            MapOf(("label", MinjaValue.Of("x"))),
            MapOf(("other", MinjaValue.Of(1L))))));

        Assert.Equal("x,", Render("{{ records | map(attribute='label') | join(',') }}", records));
        Assert.Equal("x,?", Render("{{ records | map(attribute='label', default='?') | join(',') }}", records));
        Assert.Equal(2L, Eval("records | map(attribute='label') | length", records).IntValue);

        IReadOnlyDictionary<string, MinjaValue> pairs = Context(("pairs", ListOf(
            ListOf(MinjaValue.Of("a"), MinjaValue.Of("b")),
            ListOf(MinjaValue.Of("c"), MinjaValue.Of("d")))));

        Assert.Equal("b,d", Render("{{ pairs | map(attribute='-1') | join(',') }}", pairs));
    }

    [Fact]
    public void Map_AppliesNamedBuiltinsAndTemplateMacrosToEachItem()
    {
        IReadOnlyDictionary<string, MinjaValue> words = Context(("words", ListOf(MinjaValue.Of(" a "), MinjaValue.Of("b "))));

        Assert.Equal("a|b", Render("{{ words | map('trim') | join('|') }}", words));

        // A macro bound in scope works as the per-item callable, receiving each item as its leading value.
        IReadOnlyDictionary<string, MinjaValue> nums = Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))));

        Assert.Equal("2,4", Render("{% macro twice(n) %}{{ n * 2 }}{% endmacro %}{{ nums | map('twice') | join(',') }}", nums));
    }

    [Fact]
    public void Map_RequiresAnAttributeOrCallableAndRejectsUnknownNames()
    {
        IReadOnlyDictionary<string, MinjaValue> nums = Context(("nums", ListOf(MinjaValue.Of(1L))));

        MinjaRenderException missingMode = Assert.Throws<MinjaRenderException>(() => Render("{{ nums | map() }}", nums));
        Assert.Contains("map needs attribute=", missingMode.Message);

        MinjaRenderException unknown = Assert.Throws<MinjaRenderException>(() => Render("{{ nums | map('nope') }}", nums));
        Assert.Equal("filter", unknown.Construct);
        Assert.Contains("'nope' is neither a built-in test nor a callable in scope", unknown.Message);

        MinjaRenderException named = Assert.Throws<MinjaRenderException>(() => Render("{{ nums | map(attribute='x', reverse=True) }}", nums));
        Assert.Contains("named argument 'reverse'", named.Message);
    }

    [Fact]
    public void Sort_OrdersNumbersStringsAndAttributeKeysWithReverseOption()
    {
        IReadOnlyDictionary<string, MinjaValue> nums = Context(("nums", ListOf(MinjaValue.Of(3L), MinjaValue.Of(1L), MinjaValue.Of(2L))));

        Assert.Equal("[1, 2, 3]", Render("{{ nums | sort }}", nums));
        Assert.Equal("[3, 2, 1]", Render("{{ nums | sort(reverse=True) }}", nums));

        IReadOnlyDictionary<string, MinjaValue> words = Context(("words", ListOf(MinjaValue.Of("pear"), MinjaValue.Of("apple"))));

        Assert.Equal("['apple', 'pear']", Render("{{ words | sort }}", words));
    }

    [Fact]
    public void Sort_UsesAttributePathsAndKeepsInputOrderForEqualKeys()
    {
        IReadOnlyDictionary<string, MinjaValue> messages = Context(("messages", ListOf(
            MapOf(("role", MinjaValue.Of("user")), ("content", MinjaValue.Of("hi"))),
            MapOf(("role", MinjaValue.Of("assistant")), ("content", MinjaValue.Of("bye"))))));

        Assert.Equal("bye, hi", Render("{{ messages | sort(attribute='role') | join(', ', attribute='content') }}", messages));

        IReadOnlyDictionary<string, MinjaValue> records = Context(("records", ListOf(
            MapOf(("label", MinjaValue.Of("same")), ("name", MinjaValue.Of("first"))),
            MapOf(("label", MinjaValue.Of("same")), ("name", MinjaValue.Of("second"))))));

        Assert.Equal("first,second", Render("{{ records | sort(attribute='label') | join(',', attribute='name') }}", records));

        // Reverse keeps the input order for equal keys, the way Python's stable reverse sort does.
        Assert.Equal("first,second", Render("{{ records | sort(attribute='label', reverse=True) | join(',', attribute='name') }}", records));
    }

    [Fact]
    public void Sort_IncomparableKeysAndBadArgumentsFailActionably()
    {
        IReadOnlyDictionary<string, MinjaValue> maps = Context(("maps", ListOf(MapOf(("k", MinjaValue.Of(1L))), MapOf(("j", MinjaValue.Of(2L))))));

        MinjaRenderException order = Assert.Throws<MinjaRenderException>(() => Render("{{ maps | sort }}", maps));
        Assert.Contains("cannot order-compare", order.Message);

        IReadOnlyDictionary<string, MinjaValue> nums = Context(("nums", ListOf(MinjaValue.Of(1L))));

        MinjaRenderException named = Assert.Throws<MinjaRenderException>(() => Render("{{ nums | sort(case_sensitive=True) }}", nums));
        Assert.Contains("named argument 'case_sensitive'", named.Message);
    }

    // ---------------------------------------------------------------- select / reject variants

    [Fact]
    public void Select_Reject_TruthinessSelectionWithoutATest()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("flags", ListOf(
            MinjaValue.Of(0L),
            MinjaValue.EmptyString,
            MinjaValue.Of("keep"),
            MinjaValue.None)));

        Assert.Equal(1L, Eval("flags | select | length", context).IntValue);
        Assert.Equal(3L, Eval("flags | reject | length", context).IntValue);
        Assert.Equal("keep", Render("{{ flags | select | join(',') }}", context));
    }

    [Fact]
    public void Select_Reject_NamedTestsWithExtraArguments()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L), MinjaValue.Of(3L), MinjaValue.Of(4L))),
            ("allowed", ListOf(MinjaValue.Of(1L), MinjaValue.Of(3L))));

        Assert.Equal("2, 4", Render("{{ nums | select('even') | join(', ') }}", context));
        Assert.Equal("2, 4", Render("{{ nums | reject('odd') | join(', ') }}", context));
        Assert.Equal("1,3", Render("{{ nums | select('in', allowed) | join(',') }}", context));
        Assert.Equal("2,4", Render("{{ nums | reject('in', allowed) | join(',') }}", context));

        IReadOnlyDictionary<string, MinjaValue> many = Context(("many", ListOf(
            MinjaValue.Of(1L), MinjaValue.Of(3L), MinjaValue.Of(6L), MinjaValue.Of(5L))));

        Assert.Equal("3,6", Render("{{ many | select('divisibleby', 3) | join(',') }}", many));
    }

    [Fact]
    public void Select_Reject_AcceptTemplateMacrosAsPredicates()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L), MinjaValue.Of(3L))));

        Assert.Equal("3", Render("{% macro big(n) %}{% if n > 2 %}1{% endif %}{% endmacro %}{{ nums | select('big') | join(',') }}", context));
    }

    [Fact]
    public void SelectAttr_RejectAttr_UseAttributePathsNamedTestsAndExtraArguments()
    {
        IReadOnlyDictionary<string, MinjaValue> messages = Context(("messages", ListOf(
            MapOf(("role", MinjaValue.Of("user")), ("content", MinjaValue.Of("hi"))),
            MapOf(("role", MinjaValue.Of("assistant")), ("content", MinjaValue.Of("bye"))))));

        Assert.Equal("hi", Render("{{ messages | selectattr('role', 'equalto', 'user') | join(',', attribute='content') }}", messages));
        Assert.Equal("bye", Render("{{ messages | rejectattr('role', 'equalto', 'user') | join(',', attribute='content') }}", messages));

        IReadOnlyDictionary<string, MinjaValue> records = Context(("records", ListOf(
            MapOf(("meta", MapOf(("ok", MinjaValue.True)))),
            MapOf(("meta", MapOf(("ok", MinjaValue.False)))),
            MapOf(("other", MinjaValue.Of(1L))))));

        Assert.Equal(1L, Eval("records | selectattr('meta.ok') | length", records).IntValue);
        Assert.Equal(2L, Eval("records | rejectattr('meta.ok') | length", records).IntValue);

        IReadOnlyDictionary<string, MinjaValue> flags = Context(("flags", ListOf(
            MapOf(("ok", MinjaValue.True)),
            MapOf(("ok", MinjaValue.False)))));

        Assert.Equal(1L, Eval("flags | selectattr('ok') | length", flags).IntValue);
    }

    [Fact]
    public void Select_Reject_UnknownPredicatesAndMissingAttributeArgumentFailActionably()
    {
        IReadOnlyDictionary<string, MinjaValue> nums = Context(("nums", ListOf(MinjaValue.Of(1L))));

        MinjaRenderException unknownTest = Assert.Throws<MinjaRenderException>(() => Render("{{ nums | select('nope') }}", nums));
        Assert.Equal("filter", unknownTest.Construct);
        Assert.Contains("'nope' is neither a built-in test nor a callable in scope", unknownTest.Message);

        MinjaRenderException missingPath = Assert.Throws<MinjaRenderException>(() => Render("{{ nums | selectattr() }}", nums));
        Assert.Contains("selectattr needs an attribute path", missingPath.Message);

        MinjaRenderException namedArg = Assert.Throws<MinjaRenderException>(() => Render("{{ nums | select(reverse=True) }}", nums));
        Assert.Contains("named argument 'reverse'", namedArg.Message);
    }

    // ---------------------------------------------------------------- tests

    [Theory]
    [InlineData("flag is boolean", true)]
    [InlineData("flag is integer", false)]
    [InlineData("flag is number", true)]
    [InlineData("one is integer", true)]
    [InlineData("one is float", false)]
    [InlineData("frac is float", true)]
    [InlineData("frac is integer", false)]
    [InlineData("text is string", true)]
    [InlineData("text is sequence", true)]
    [InlineData("record is mapping", true)]
    [InlineData("record is sequence", false)]
    [InlineData("nums is sequence", true)]
    [InlineData("missing is iterable", true)]
    [InlineData("flag is not boolean", false)]
    public void Tests_KindAndShapePredicates(string expression, bool expected)
    {
        Assert.Equal(expected, Eval(expression, CollectionContext()).IsTruthy);
    }

    [Fact]
    public void Tests_DefinedUndefinedNoneCollapseToTheSingleValueModel()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("x", MinjaValue.Of(1L)),
            ("nothing", MinjaValue.None));

        Assert.True(Eval("x is defined", context).IsTruthy);
        Assert.False(Eval("missing is defined", context).IsTruthy);
        Assert.True(Eval("missing is undefined", context).IsTruthy);

        // None and undefined are the same value here, so an explicit None reads as undefined to a test.
        Assert.False(Eval("nothing is defined", context).IsTruthy);
        Assert.True(Eval("None is none").IsTruthy);
        Assert.True(Eval("missing is none").IsTruthy);
        Assert.False(Eval("x is not defined", context).IsTruthy);
    }

    [Fact]
    public void Tests_NumberPredicatesAndMembership()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("four", MinjaValue.Of(4L)),
            ("neg", MinjaValue.Of(-3L)),
            ("text", MinjaValue.Of("abc")),
            ("allowed", ListOf(MinjaValue.Of("user"), MinjaValue.Of("system"))));

        Assert.True(Eval("four is even", context).IsTruthy);
        Assert.False(Eval("four is odd", context).IsTruthy);
        Assert.True(Eval("neg is odd", context).IsTruthy);
        Assert.False(Eval("text is even", context).IsTruthy);
        Assert.True(Eval("four is divisibleby(2)", context).IsTruthy);
        Assert.False(Eval("four is divisibleby(3)", context).IsTruthy);
        Assert.True(Eval("'user' in allowed", context).IsTruthy);

        // The test form of the membership predicate takes its container as a call argument.
        Assert.True(Eval("'user' is in(allowed)", context).IsTruthy);
        Assert.False(Eval("'tool' is in(allowed)", context).IsTruthy);
    }

    [Fact]
    public void Tests_EqualtoWorksAsATestAndAFilter()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("one", MinjaValue.Of(1L)));

        Assert.True(Eval("one is equalto(1)", context).IsTruthy);
        Assert.False(Eval("one is equalto(2)", context).IsTruthy);
        Assert.True(Eval("one is not equalto(2)", context).IsTruthy);
        Assert.Equal("True", Render("{{ 1 | equalto(1) }}"));
    }

    [Fact]
    public void Tests_ParameterizedTestsWithoutParenthesesFailActionably()
    {
        // The parser only attaches arguments to a test in call form, so the bare parameterized form is a syntax
        // error naming the position rather than a silent partial parse.
        MinjaSyntaxException error = Assert.Throws<MinjaSyntaxException>(() => Eval("1 is equalto 1"));

        Assert.Contains("expected '}}'", error.Message);
        Assert.True(error.Position >= 0);
    }

    [Fact]
    public void Tests_UnknownTestsAndDivideByZeroFailActionably()
    {
        MinjaRenderException unknown = Assert.Throws<MinjaRenderException>(() => Eval("'a' is not_a_test"));

        Assert.Equal("test", unknown.Construct);
        Assert.Contains("unknown test 'not_a_test'", unknown.Message);
        Assert.True(unknown.Position >= 0);

        IReadOnlyDictionary<string, MinjaValue> context = Context(("four", MinjaValue.Of(4L)));

        MinjaRenderException zero = Assert.Throws<MinjaRenderException>(() => Eval("four is divisibleby(0)", context));
        Assert.Equal("test", zero.Construct);
        Assert.Contains("division by zero", zero.Message);
    }

    // ---------------------------------------------------------------- range / namespace

    [Theory]
    [InlineData("range(3)", "[0, 1, 2]")]
    [InlineData("range(1, 4)", "[1, 2, 3]")]
    [InlineData("range(0, 6, 2)", "[0, 2, 4]")]
    [InlineData("range(3, 0, -1)", "[3, 2, 1]")]
    [InlineData("range(5, 1, -2)", "[5, 3]")]
    [InlineData("range(0)", "[]")]
    [InlineData("range(4, 4)", "[]")]
    public void Range_OneTwoAndThreeArgumentForms(string expression, string expected) => Assert.Equal(expected, Render("{{ " + expression + " }}"));

    [Fact]
    public void Range_BadArgumentsFailActionably()
    {
        MinjaRenderException zeroStep = Assert.Throws<MinjaRenderException>(() => Eval("range(0, 3, 0)"));
        Assert.Equal("range", zeroStep.Construct);
        Assert.Contains("step cannot be zero", zeroStep.Message);

        MinjaRenderException tooMany = Assert.Throws<MinjaRenderException>(() => Eval("range(1, 2, 3, 4)"));
        Assert.Contains("one to three arguments", tooMany.Message);

        IReadOnlyDictionary<string, MinjaValue> context = Context(("text", MinjaValue.Of("abc")));

        MinjaRenderException notInteger = Assert.Throws<MinjaRenderException>(() => Eval("range(text)", context));
        Assert.Contains("needs an integer", notInteger.Message);
    }

    [Fact]
    public void Namespace_CreatesAMutableMapThatDottedSetUpdates()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L), MinjaValue.Of(3L))));

        Assert.Equal("3", Render("{% set ns = namespace(count=0) %}{% for n in nums %}{% set ns.count = ns.count + 1 %}{% endfor %}{{ ns.count }}", context));

        IReadOnlyDictionary<string, MinjaValue> nested = Context(("meta", MapOf(("count", MinjaValue.Of(0L)))));

        Assert.Equal("2", Render("{% set ns = namespace(meta=meta) %}{% set ns.meta.count = 2 %}{{ ns.meta.count }}", nested));
    }

    [Fact]
    public void Namespace_VisibleThroughMacrosAndRejectsPositionalArguments()
    {
        Assert.Equal("7", Render("{% set ns = namespace(value=0) %}{% macro bump() %}{% set ns.value = 7 %}{% endmacro %}{{ bump() }}{{ ns.value }}"));

        MinjaRenderException positional = Assert.Throws<MinjaRenderException>(() => Eval("namespace(1)"));
        Assert.Equal("namespace", positional.Construct);
        Assert.Contains("name=value arguments", positional.Message);

        // Creating a missing intermediate map is not supported by dotted assignment, and it fails honestly.
        MinjaRenderException missingOwner = Assert.Throws<MinjaRenderException>(() => Render("{% set ns = namespace() %}{% set ns.a.b = 1 %}"));
        Assert.Equal("set", missingOwner.Construct);
    }

    // ---------------------------------------------------------------- raise_exception

    [Fact]
    public void RaiseException_CarriesTheTemplateMessageAndPositionAsARenderError()
    {
        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Render("{% set x = 1 %}{{ raise_exception('Only system message is not supported.') }}"));

        Assert.Equal("raise_exception", error.Construct);
        Assert.Contains("Only system message is not supported.", error.Message);
        Assert.True(error.Position > "{% set x = 1 %}".Length);
    }

    [Fact]
    public void RaiseException_FiresFromChatTemplateShapesAndFilterForm()
    {
        string template = "{{ bos_token }}{%- if messages[0]['role'] == 'system' %}{{ raise_exception('Only system message is not supported.') }}{% endif %}";

        IReadOnlyDictionary<string, MinjaValue> userFirst = Context(
            ("bos_token", MinjaValue.EmptyString),
            ("messages", ListOf(MapOf(("role", MinjaValue.Of("user")), ("content", MinjaValue.Of("hi"))))));

        Assert.Equal(string.Empty, Render(template, userFirst));

        IReadOnlyDictionary<string, MinjaValue> systemFirst = Context(
            ("bos_token", MinjaValue.EmptyString),
            ("messages", ListOf(MapOf(("role", MinjaValue.Of("system")), ("content", MinjaValue.Of("be helpful"))))));

        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Render(template, systemFirst));
        Assert.Equal("raise_exception", error.Construct);
        Assert.Contains("Only system message is not supported.", error.Message);

        // The filter form raises too: the filtered value reaches it but nothing renders.
        MinjaRenderException asFilter = Assert.Throws<MinjaRenderException>(() => Render("{{ 'x' | raise_exception('tool call without tools') }}"));
        Assert.Contains("tool call without tools", asFilter.Message);
    }

    [Fact]
    public void RaiseException_StopsRenderingInsteadOfEmittingText()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))));

        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Render("{% for n in nums %}{{ n }}{% if n == 2 %}{{ raise_exception('nope') }}{% endif %}{% endfor %}", context));

        Assert.Equal("raise_exception", error.Construct);
        Assert.Contains("nope", error.Message);
    }

    // ---------------------------------------------------------------- integration in templates

    [Fact]
    public void Templates_ComposeFiltersTestsAndCallsInStatementShapes()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L), MinjaValue.Of(3L))),
            ("allowed", ListOf(MinjaValue.Of(1L), MinjaValue.Of(3L))));

        Assert.Equal("1,3,", Render("{% for item in nums | select('in', allowed) %}{{ item }},{% endfor %}", context));
        Assert.Equal("2", Render("{% for n in nums | select('even') %}{{ n }}{% endfor %}", context));
        Assert.Equal("yes", Render("{% if value is defined %}yes{% else %}no{% endif %}", Context(("value", MinjaValue.Of(1L)))));
        Assert.Equal("no", Render("{% if missing is defined %}yes{% else %}no{% endif %}"));

        // Filter blocks apply the same built-in namespace to their rendered body.
        Assert.Equal("padded", Render("{% filter trim %}  padded  {% endfilter %}"));
        Assert.Equal("\"body\"", Render("{% filter tojson %}body{% endfilter %}"));
    }

    [Fact]
    public void Templates_UserBindingsShadowBuiltinsAndValueBindingsBlockFunctionCalls()
    {
        // A macro of a built-in name wins, which is how templates override built-ins with their own helper.
        Assert.Equal("custom", Render("{% macro length(x) %}custom{% endmacro %}{{ 'abc' | length }}"));

        MinjaRenderException shadowed = Assert.Throws<MinjaRenderException>(() => Render("{% set range = 3 %}{{ range(2) }}"));
        Assert.Equal("call", shadowed.Construct);
        Assert.Contains("cannot call a Int value", shadowed.Message);
    }

    // ---------------------------------------------------------------- inline literal containers

    [Fact]
    public void Templates_BuiltinsAcceptInlineListAndMapLiterals()
    {
        // Parity shape from llama-family templates: membership containers written inline instead of injected.
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("messages", ListOf(
                MapOf(("role", MinjaValue.Of("user")), ("content", MinjaValue.Of("hi"))),
                MapOf(("role", MinjaValue.Of("assistant")), ("content", MinjaValue.Of("yo"))),
                MapOf(("role", MinjaValue.Of("tool")), ("content", MinjaValue.Of("x"))))));

        Assert.Equal("user,assistant", Render("{{ messages | map(attribute='role') | select('in', ['user', 'assistant']) | join(',') }}", context));
        Assert.Equal("assistant,tool", Render("{{ messages | map(attribute='role') | reject('in', ['user']) | join(',') }}", context));

        // The inline container also reaches the predicate through the selectattr extra-argument position.
        Assert.Equal("hi|yo", Render("{{ messages | selectattr('role', 'in', ['user', 'assistant']) | map(attribute='content') | join('|') }}", context));

        // Membership operators take literal containers directly.
        Assert.Equal("True", Render("{{ message['role'] in ['user', 'assistant'] }}", Context(("message", MapOf(("role", MinjaValue.Of("user")))))));
        Assert.Equal("False", Render("{{ message['role'] in ['user', 'assistant'] }}", Context(("message", MapOf(("role", MinjaValue.Of("system")))))));

        // A literal survives the whole pipeline: tojson dumps it like any context-built map.
        Assert.Equal("{\"tools\": [{\"name\": \"x\"}]}", Render("{{ {'tools': [{'name': 'x'}]} | tojson }}"));
    }

    // ---------------------------------------------------------------- helpers

    private static MinjaValue Eval(string expression, IReadOnlyDictionary<string, MinjaValue>? context = null) =>
        MinjaEvaluator.Evaluate(expression, context);

    private static string Render(string template, IReadOnlyDictionary<string, MinjaValue>? context = null) =>
        MinjaEvaluator.Render(template, context);

    private static IReadOnlyDictionary<string, MinjaValue> CollectionContext() => Context(
        ("flag", MinjaValue.True),
        ("one", MinjaValue.Of(1L)),
        ("frac", MinjaValue.Of(0.5)),
        ("text", MinjaValue.Of("a")),
        ("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L), MinjaValue.Of(3L))),
        ("record", MapOf(("k", MinjaValue.Of(1L)), ("j", MinjaValue.Of(2L)))));

    private static MinjaValue ListOf(params MinjaValue[] items) => MinjaValue.List(items);

    private static MinjaValue MapOf(params (string Key, MinjaValue Value)[] entries) =>
        MinjaValue.Map(entries.Select(entry => new KeyValuePair<string, MinjaValue>(entry.Key, entry.Value)));

    private static IReadOnlyDictionary<string, MinjaValue> Context(params (string Name, MinjaValue Value)[] entries) =>
        entries.ToDictionary(entry => entry.Name, entry => entry.Value);
}

using Lair.Templating.Minja;

namespace Lair.Tests;

/// <summary>
/// Expression and value goldens for the managed minja evaluator: Python-like truthiness, none/undefined
/// equivalence, int/float and deep list/map equality, membership, indexing and slicing, arithmetic and
/// concatenation, operand-returning and/or, ternary expressions, Jinja-style rendering with comment and
/// whitespace elision, and minja-compatible JSON dump formatting.
/// </summary>
[Trait("Category", "MinjaEval")]
public sealed class MinjaEvalTests
{
    // ---------------------------------------------------------------- truthiness

    [Theory]
    [InlineData("None", false)]
    [InlineData("missing", false)]
    [InlineData("False", false)]
    [InlineData("0", false)]
    [InlineData("0.0", false)]
    [InlineData("''", false)]
    [InlineData("True", true)]
    [InlineData("1", true)]
    [InlineData("0.5", true)]
    [InlineData("'a'", true)]
    public void Expression_TruthinessFollowsPython(string expression, bool expected)
    {
        Assert.Equal(expected, Eval(expression).IsTruthy);
        Assert.Equal(!expected, Eval($"not ({expression})").IsTruthy);
    }

    [Fact]
    public void ContextValues_EmptyListAndMapAreFalsey_NonEmptyAreTruthy()
    {
        Assert.False(MinjaValue.EmptyList.IsTruthy);
        Assert.False(MinjaValue.EmptyMap.IsTruthy);
        Assert.False(MinjaValue.Of(0L).IsTruthy);
        Assert.False(MinjaValue.Of(0.0).IsTruthy);
        Assert.False(MinjaValue.EmptyString.IsTruthy);
        Assert.False(MinjaValue.None.IsTruthy);

        Assert.True(ListOf(MinjaValue.Of(0L)).IsTruthy);
        Assert.True(MapOf(("k", MinjaValue.None)).IsTruthy);
        Assert.True(MinjaValue.Of("0").IsTruthy);
    }

    [Fact]
    public void ContextValues_UndefinedNameIsNoneAndNotDistinctFromNone()
    {
        Assert.True(Eval("missing == None").IsTruthy);
        Assert.True(MinjaValue.None.Equals(Eval("missing")));
        Assert.Equal(MinjaValue.None, Eval("missing"));
    }

    // ---------------------------------------------------------------- equality and comparison

    [Theory]
    [InlineData("1 == 1", true)]
    [InlineData("1 == 1.0", true)]
    [InlineData("2.5 == 2.5", true)]
    [InlineData("1 == 2", false)]
    [InlineData("'a' == 'a'", true)]
    [InlineData("'a' == 'b'", false)]
    [InlineData("None == None", true)]
    [InlineData("None == 0", false)]
    [InlineData("True == 1", false)]
    [InlineData("1 != 2", true)]
    [InlineData("1 != 1.0", false)]
    public void Equality_Goldens(string expression, bool expected)
    {
        Assert.Equal(expected, Eval(expression).IsTruthy);
    }

    [Fact]
    public void Equality_ListsAndMapsCompareDeeply_MapsIgnoreInsertionOrder()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("left", ListOf(MinjaValue.Of(1L), MapOf(("k", MinjaValue.Of("v"))))),
            ("right", ListOf(MinjaValue.Of(1L), MapOf(("k", MinjaValue.Of("v"))))),
            ("other", ListOf(MinjaValue.Of(1L), MapOf(("k", MinjaValue.Of("w"))))),
            ("sameMapOtherOrder", MapOf(("k", MinjaValue.Of("v")), ("j", MinjaValue.Of(1L)))),
            ("sameMapOneOrder", MapOf(("j", MinjaValue.Of(1L)), ("k", MinjaValue.Of("v")))),
            ("shortList", ListOf(MinjaValue.Of(1L))),
            ("longList", ListOf(MinjaValue.Of(1L), MinjaValue.None)));

        Assert.True(Eval("left == right", context).IsTruthy);
        Assert.True(Eval("left != other", context).IsTruthy);
        Assert.True(Eval("sameMapOtherOrder == sameMapOneOrder", context).IsTruthy);
        Assert.True(Eval("shortList != longList", context).IsTruthy);
    }

    [Theory]
    [InlineData("1 < 2", true)]
    [InlineData("2 < 1", false)]
    [InlineData("1.5 > 1", true)]
    [InlineData("2 >= 2", true)]
    [InlineData("'abc' < 'abd'", true)]
    [InlineData("'b' >= 'a'", true)]
    public void Ordering_Goldens(string expression, bool expected)
    {
        Assert.Equal(expected, Eval(expression).IsTruthy);
    }

    [Fact]
    public void Ordering_ListsCompareLexicographically()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("a", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))),
            ("b", ListOf(MinjaValue.Of(1L), MinjaValue.Of(3L))),
            ("prefix", ListOf(MinjaValue.Of(1L))));

        Assert.True(Eval("a < b", context).IsTruthy);
        Assert.True(Eval("prefix < a", context).IsTruthy);
    }

    [Theory]
    [InlineData("None < 1")]
    [InlineData("'a' < 1")]
    [InlineData("m < 2")]
    public void Ordering_TypeMismatchThrowsRenderError(string expression)
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("m", MapOf(("x", MinjaValue.Of(1L)))));

        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Eval(expression, context));
        Assert.Equal("comparison", error.Construct);
    }

    // ---------------------------------------------------------------- membership

    [Theory]
    [InlineData("'ell' in 'hello'", true)]
    [InlineData("'z' in 'hello'", false)]
    [InlineData("'' in 'hello'", true)]
    [InlineData("'z' not in 'hello'", true)]
    public void Membership_StringIsSubstringContainment(string expression, bool expected)
    {
        Assert.Equal(expected, Eval(expression).IsTruthy);
    }

    [Fact]
    public void Membership_ListIsDeepElementEquality()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L), MinjaValue.Of(3L))),
            ("pair", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))),
            ("nested", ListOf(ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L)))));

        Assert.True(Eval("2 in nums", context).IsTruthy);
        Assert.True(Eval("4 not in nums", context).IsTruthy);
        Assert.True(Eval("pair in nested", context).IsTruthy);
    }

    [Fact]
    public void Membership_MapIsKeyPresence()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("message", MapOf(("role", MinjaValue.Of("user")))));

        Assert.True(Eval("'role' in message", context).IsTruthy);
        Assert.True(Eval("'content' not in message", context).IsTruthy);
        Assert.False(Eval("missing in message", context).IsTruthy);
    }

    [Fact]
    public void Membership_UndefinedContainerFindsNothing()
    {
        Assert.False(Eval("'a' in missing").IsTruthy);
        Assert.True(Eval("'a' not in missing").IsTruthy);
    }

    // ---------------------------------------------------------------- indexing

    [Fact]
    public void Indexing_ListsAcceptNegativeOffsets()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(10L), MinjaValue.Of(20L), MinjaValue.Of(30L))));

        Assert.Equal(10L, Eval("nums[0]", context).IntValue);
        Assert.Equal(30L, Eval("nums[-1]", context).IntValue);
        Assert.Equal(20L, Eval("nums[-2]", context).IntValue);
    }

    [Fact]
    public void Indexing_StringsReturnSingleCharacterStrings()
    {
        Assert.Equal("e", Eval("'hello'[1]").StringValue);
        Assert.Equal("o", Eval("'hello'[-1]").StringValue);
    }

    [Fact]
    public void Indexing_MapsReadKeys_AttributeAccessReadsKeysToo()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("message", MapOf(("role", MinjaValue.Of("user")))));

        Assert.Equal("user", Eval("message['role']", context).StringValue);
        Assert.Equal("user", Eval("message.role", context).StringValue);
    }

    [Fact]
    public void Indexing_MissingKeyOutOfRangeIndexAndUndefinedAttributeAllEvaluateAsNone()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("nums", ListOf(MinjaValue.Of(1L))),
            ("message", MapOf(("role", MinjaValue.Of("user")))));

        Assert.True(Eval("message['missing']", context).IsNone);
        Assert.True(Eval("nums[9]", context).IsNone);
        Assert.True(Eval("nums[-9]", context).IsNone);
        Assert.True(Eval("missing.role", context).IsNone);
        Assert.True(Eval("missing[0]", context).IsNone);
        Assert.True(Eval("'text'.attr", context).IsNone);
    }

    // ---------------------------------------------------------------- slicing

    [Theory]
    [InlineData("'hello'[1:3]", "el")]
    [InlineData("'hello'[:2]", "he")]
    [InlineData("'hello'[3:]", "lo")]
    [InlineData("'hello'[:]", "hello")]
    [InlineData("'hello'[::2]", "hlo")]
    [InlineData("'hello'[::-1]", "olleh")]
    [InlineData("'hello'[-3:]", "llo")]
    [InlineData("'hello'[1:-1]", "ell")]
    [InlineData("'hello'[10:20]", "")]
    public void Slicing_StringsMatchPythonSemantics(string expression, string expected)
    {
        Assert.Equal(expected, Eval(expression).StringValue);
    }

    [Fact]
    public void Slicing_ListsMatchPythonSemanticsIncludingClampedAndReversedForms()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L), MinjaValue.Of(3L), MinjaValue.Of(4L))));

        Assert.True(Eval("nums[1:3]", context).Equals(ListOf(MinjaValue.Of(2L), MinjaValue.Of(3L))));
        Assert.True(Eval("nums[::2]", context).Equals(ListOf(MinjaValue.Of(1L), MinjaValue.Of(3L))));
        Assert.True(Eval("nums[::-1]", context).Equals(ListOf(MinjaValue.Of(4L), MinjaValue.Of(3L), MinjaValue.Of(2L), MinjaValue.Of(1L))));
        Assert.True(Eval("nums[-2:]", context).Equals(ListOf(MinjaValue.Of(3L), MinjaValue.Of(4L))));
        Assert.True(Eval("nums[10:20]", context).Equals(MinjaValue.EmptyList));
    }

    [Fact]
    public void Slicing_MapsAndZeroStepThrowRenderError()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("m", MapOf(("k", MinjaValue.Of(1L)))));

        Assert.Throws<MinjaRenderException>(() => Eval("m[:]", context));
        Assert.Throws<MinjaRenderException>(() => Eval("'ab'[::0]"));
    }

    // ---------------------------------------------------------------- arithmetic and concatenation

    [Theory]
    [InlineData("1 + 2 * 3", 7.0)]
    [InlineData("(1 + 2) * 3", 9.0)]
    [InlineData("10 - 4", 6.0)]
    [InlineData("-3", -3.0)]
    [InlineData("-3.5", -3.5)]
    [InlineData("7 / 2", 3.5)]
    [InlineData("4 / 2", 2.0)]
    [InlineData("7 // 2", 3.0)]
    [InlineData("-7 // 2", -4.0)]
    [InlineData("7 % 3", 1.0)]
    [InlineData("-7 % 3", 2.0)]
    [InlineData("7 % -3", -2.0)]
    [InlineData("2 ** 10", 1024.0)]
    [InlineData("2 ** 3 ** 2", 512.0)]
    [InlineData("-2 ** 2", -4.0)]
    [InlineData("1 + 1.5", 2.5)]
    public void Arithmetic_Goldens(string expression, double expected)
    {
        Assert.Equal(expected, Eval(expression).NumberValue);
    }

    [Theory]
    [InlineData("1 / 0")]
    [InlineData("1 // 0")]
    [InlineData("1 % 0")]
    public void Arithmetic_DivisionByZeroThrowsRenderError(string expression)
    {
        Assert.Throws<MinjaRenderException>(() => Eval(expression));
    }

    [Fact]
    public void Arithmetic_IntegerResultsStayInteger_FloatResultsStayFloat()
    {
        Assert.Equal(MinjaValueKind.Int, Eval("2 + 3").Kind);
        Assert.Equal(MinjaValueKind.Float, Eval("2 + 3.0").Kind);
        Assert.Equal(MinjaValueKind.Float, Eval("7 / 2").Kind);
        Assert.Equal(MinjaValueKind.Int, Eval("7 // 2").Kind);
    }

    [Fact]
    public void Concatenation_StringsAndListsUsePlus()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))),
            ("tail", ListOf(MinjaValue.Of(3L))));

        Assert.Equal("ab", Eval("'a' + 'b'").StringValue);
        Assert.Equal("a b", Eval("'a' + ' ' + 'b'").StringValue);
        Assert.True(Eval("nums + tail", context).Equals(ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L), MinjaValue.Of(3L))));
    }

    [Fact]
    public void Concatenation_MismatchedKindsThrowRenderError()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("nums", ListOf(MinjaValue.Of(1L))));

        Assert.Throws<MinjaRenderException>(() => Eval("'a' + 1"));
        Assert.Throws<MinjaRenderException>(() => Eval("nums + 'a'", context));
    }

    // ---------------------------------------------------------------- boolean operators

    [Fact]
    public void BooleanOperators_AndOrReturnOperandsNotBooleans()
    {
        Assert.Equal("b", Eval("'a' and 'b'").StringValue);
        Assert.Equal("", Eval("'' and 'x'").StringValue);
        Assert.Equal("fallback", Eval("0 or 'fallback'").StringValue);
        Assert.Equal("a", Eval("'a' or 'b'").StringValue);
        Assert.Equal(0L, Eval("None or 0").IntValue);
        Assert.True(Eval("'' or None").IsNone);
        Assert.Equal(2L, Eval("1 and 2").IntValue);
        Assert.Equal(0L, Eval("0 and 'x'").IntValue);
    }

    // ---------------------------------------------------------------- ternary

    [Theory]
    [InlineData("'yes' if 1 else 'no'", "yes")]
    [InlineData("'yes' if 0 else 'no'", "no")]
    [InlineData("'yes' if '' else 'no'", "no")]
    [InlineData("'yes' if 'a' else 'no'", "yes")]
    public void Ternary_ConditionalExpressions(string expression, string expected)
    {
        Assert.Equal(expected, Eval(expression).StringValue);
    }

    [Fact]
    public void Ternary_ElseBranchNestsRightAssociatively()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("falsy", MinjaValue.Of(0L)));

        Assert.Equal("c", Eval("'a' if 0 else 'b' if 0 else 'c'").StringValue);
        Assert.Equal("b", Eval("'a' if 0 else 'b' if 1 else 'c'").StringValue);
        Assert.Equal("b", Eval("'a' if falsy else 'b'", context).StringValue);
    }

    // ---------------------------------------------------------------- rendering

    [Fact]
    public void Render_TextAndExpressionsProduceJinjaStyleOutput()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("name", MinjaValue.Of("world")));

        Assert.Equal("hello world!", Render("hello {{ name }}!", context));
        Assert.Equal("True", Render("{{ True }}"));
        Assert.Equal("", Render("{{ missing }}"));
        Assert.Equal("[1, 2]", Render("{{ nums }}", Context(("nums", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))))));
        Assert.Equal("{'role': 'user'}", Render("{{ message }}", Context(("message", MapOf(("role", MinjaValue.Of("user")))))));
    }

    [Fact]
    public void Render_CommentsNeverEmitOutput()
    {
        Assert.Equal("ab", Render("a{# hidden note #}b"));
        Assert.Equal("ab", Render("a{#- hidden -#}b"));
    }

    [Fact]
    public void Render_WhitespaceElisionFollowsTrimMarkers()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(("name", MinjaValue.Of("world")));

        Assert.Equal("aworld b", Render("a {{- name }} b", context));
        Assert.Equal("a worldb", Render("a {{ name -}} b", context));
        Assert.Equal("aworldb", Render("a {{- name -}} b", context));
        Assert.Equal("a world b", Render("a {{ name }} b", context));
    }

    [Fact]
    public void Render_WhitespaceBetweenTagsIsPreserved()
    {
        Assert.Equal("a b", Render("{{ 'a' }} {{ 'b' }}"));
    }

    [Fact]
    public void Render_StatementsExecuteWithContextValues()
    {
        // Statements used to raise an unsupported-statement error; they now execute against the context.
        // Deeper statement and scoping goldens live under Category=MinjaStatements.
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("x", MinjaValue.Of(1L)),
            ("items", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))));

        Assert.Equal("a", Render("{% if x %}a{% endif %}", context));
        Assert.Equal("", Render("{% if missing %}a{% endif %}", context));
        Assert.Equal("12", Render("{% for i in items %}{{ i }}{% endfor %}", context));
        Assert.Equal("1", Render("{% set x = 1 %}{{ x }}", context));
    }

    [Fact]
    public void Render_KnownBuiltinsResolveAndUnknownOnesStayActionable()
    {
        // Built-in filters, tests and function calls resolve through the minja built-in namespace. Goldens for
        // each one live under Category=MinjaFilters; this test keeps honest coverage for names outside that set.
        Assert.Equal("3", Render("{{ 'abc' | length }}"));
        Assert.Equal("True", Render("{{ x is defined }}", Context(("x", MinjaValue.Of(1L)))));
        Assert.Equal("[0, 1, 2]", Render("{{ range(3) }}"));

        MinjaRenderException unknownFilter = Assert.Throws<MinjaRenderException>(() => Render("{{ 'a' | upper }}"));
        Assert.Equal("filter", unknownFilter.Construct);
        Assert.Contains("unknown filter 'upper'", unknownFilter.Message);
        Assert.True(unknownFilter.Position >= 0);

        MinjaRenderException unknownTest = Assert.Throws<MinjaRenderException>(() => Render("{{ x is not_a_test }}"));
        Assert.Equal("test", unknownTest.Construct);
        Assert.Contains("not_a_test", unknownTest.Message);
        Assert.True(unknownTest.Position >= 0);

        MinjaRenderException unknownFunction = Assert.Throws<MinjaRenderException>(() => Render("{{ not_a_function(1) }}"));
        Assert.Equal("call", unknownFunction.Construct);
        Assert.Contains("not_a_function", unknownFunction.Message);
        Assert.True(unknownFunction.Position >= 0);
    }

    [Fact]
    public void Engine_ParseEvaluateRenderConvenienceMethodsWork()
    {
        Assert.Equal(3L, MinjaEngine.Evaluate("1 + 2").IntValue);
        Assert.Equal("hi 3", MinjaEngine.Render("hi {{ 1 + 2 }}"));
        Assert.Single(MinjaEngine.Parse("plain").Nodes);
    }

    // ---------------------------------------------------------------- JSON dump

    [Fact]
    public void JsonDump_MapsSerializeInInsertionOrder()
    {
        MinjaValue map = MapOf(
            ("b", MinjaValue.Of(2L)),
            ("a", MinjaValue.Of(1L)),
            ("c", MinjaValue.Of(3L)));

        Assert.Equal("{\"b\": 2, \"a\": 1, \"c\": 3}", map.ToJson());
    }

    [Fact]
    public void JsonDump_NestedListsAndMaps()
    {
        MinjaValue value = MapOf(
            ("list", ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))),
            ("nested", MapOf(("k", MinjaValue.Of("v")))),
            ("empty", MinjaValue.EmptyList));

        Assert.Equal("{\"list\": [1, 2], \"nested\": {\"k\": \"v\"}, \"empty\": []}", value.ToJson());
    }

    [Fact]
    public void JsonDump_NoneBooleansRenderAsJsonNullAndLiterals()
    {
        Assert.Equal("null", MinjaValue.None.ToJson());
        Assert.Equal("true", MinjaValue.True.ToJson());
        Assert.Equal("false", MinjaValue.False.ToJson());
    }

    [Theory]
    [InlineData(1.0, "1.0")]
    [InlineData(3.5, "3.5")]
    [InlineData(1e-05, "1e-05")]
    [InlineData(1e+17, "1e+17")]
    [InlineData(0.30000000000000004, "0.30000000000000004")]
    public void JsonDump_FloatsUsePythonReprFormatting(double number, string expected)
    {
        Assert.Equal(expected, MinjaValue.Of(number).ToJson());
    }

    [Theory]
    [InlineData(42L, "42")]
    [InlineData(-7L, "-7")]
    public void JsonDump_IntegersRenderPlain(long number, string expected)
    {
        Assert.Equal(expected, MinjaValue.Of(number).ToJson());
    }

    [Fact]
    public void JsonDump_StringsUsePythonJsonEscaping()
    {
        Assert.Equal("\"a\\nb\"", MinjaValue.Of("a\nb").ToJson());
        Assert.Equal("\"caf\\u00e9\"", MinjaValue.Of("caf\u00e9").ToJson());
        Assert.Equal("\"quote \\\" here\"", MinjaValue.Of("quote \" here").ToJson());
        Assert.Equal("\"back \\\\ slash\"", MinjaValue.Of("back \\ slash").ToJson());
    }

    [Fact]
    public void JsonDump_OrderedMapUpdateKeepsOriginalPosition()
    {
        MinjaMap map = new();
        map.Set("first", MinjaValue.Of(1L));
        map.Set("second", MinjaValue.Of(2L));
        map.Set("first", MinjaValue.Of(9L));

        MinjaValue value = MinjaValue.Map(map);
        Assert.Equal("{\"first\": 9, \"second\": 2}", value.ToJson());
    }

    // ---------------------------------------------------------------- inline list and map literals

    [Theory]
    [InlineData("[]", "[]")]
    [InlineData("[1]", "[1]")]
    [InlineData("['user', 'assistant']", "['user', 'assistant']")]
    [InlineData("[[1, 2], [3]]", "[[1, 2], [3]]")]
    [InlineData("{'role': 'system'}", "{'role': 'system'}")]
    [InlineData("{}", "{}")]
    // The value model renders none as empty text everywhere (its documented str() choice), so a nested None in a
    // literal repr follows the same rule instead of Python's 'None'; JSON dump covers null separately.
    [InlineData("{'k': None}", "{'k': }")]
    [InlineData("{'xs': [1, {'k': 'v'}], 'n': 2}", "{'xs': [1, {'k': 'v'}], 'n': 2}")]
    public void Literals_EvaluateAndRenderWithPythonRepr(string expression, string expected)
    {
        Assert.Equal(expected, Eval(expression).ToString());
    }

    [Fact]
    public void LiteralList_PreservesSourceOrder_AndLiteralMapPreservesInsertionOrder()
    {
        MinjaValue list = Eval("['b', 'a', 'c']");
        Assert.True(list.IsTruthy);
        Assert.Equal(3, list.Items.Count);

        IReadOnlyList<string> items = [.. list.Items.Select(item => item.StringValue)];
        Assert.Equal(["b", "a", "c"], items.ToArray());

        MinjaValue map = Eval("{'b': 2, 'a': 1}");
        Assert.True(map.IsTruthy);
        Assert.Equal(2, map.MapValue.Count);
        Assert.Equal(["b", "a"], map.MapValue.Keys.ToArray());
    }

    [Theory]
    [InlineData("[]", false)]
    [InlineData("[0]", true)]
    [InlineData("{}", false)]
    [InlineData("{'k': None}", true)]
    public void Literals_EmptyContainersAreFalsey_NonEmptyAreTruthy(string expression, bool expected)
    {
        Assert.Equal(expected, Eval(expression).IsTruthy);
    }

    [Fact]
    public void LiteralMap_DuplicateKeyKeepsOriginalPositionWithLastValue()
    {
        // MinjaMap.Set semantics applied to literals: the later entry wins but the key stays at its first position.
        Assert.Equal("{\"a\": 9, \"b\": 2}", Eval("{'a': 1, 'b': 2, 'a': 9}").ToJson());

        MinjaValue map = Eval("{'a': 1, 'b': 2, 'a': 9}");
        Assert.Equal(2, map.MapValue.Count);
        Assert.Equal(["a", "b"], map.MapValue.Keys.ToArray());
    }

    [Theory]
    // This library's maps are string-keyed everywhere (attributes, key lookups and JSON dumps), so a literal with a
    // non-string key is an actionable render error instead of silent coercion. minja itself accepts primitive keys;
    // this divergence is pinned until oracle parity work revisits the value model.
    [InlineData("{1: 'x'}")]
    [InlineData("{'a': 1, True: 2}")]
    public void LiteralMap_NonStringKeysThrowRenderError(string expression)
    {
        MinjaRenderException error = Assert.Throws<MinjaRenderException>(() => Eval(expression));

        Assert.Equal("map literal", error.Construct);
        Assert.Contains("strings", error.Message);
        Assert.True(error.Position >= 0);
    }

    [Fact]
    public void Literals_MembershipIndexingSlicingConcatenationAndEqualityWork()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("message", MapOf(("role", MinjaValue.Of("user")))),
            ("xs", ListOf(MinjaValue.Of("a"), MinjaValue.Of("b"))));

        Assert.True(Eval("message['role'] in ['user', 'assistant']", context).IsTruthy);
        Assert.True(Eval("'tool' not in ['user', 'assistant']").IsTruthy);
        Assert.True(Eval("'k' in {'k': 1}").IsTruthy);

        Assert.Equal("b", Eval("['a', 'b'][1]").StringValue);
        Assert.True(Eval("{'xs': [1, 2]}['xs']").Equals(ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))));

        // Indexing a string-keyed map with an integer key follows the existing lookup coercion and finds nothing.
        Assert.True(Eval("m[0]", Context(("m", MapOf(("a", MinjaValue.Of(1L)))))).IsNone);

        Assert.Equal("ab", Eval("'a' + ['b'][0]").StringValue);
        Assert.True(Eval("[1] + [2]", context).Equals(ListOf(MinjaValue.Of(1L), MinjaValue.Of(2L))));
        Assert.True(Eval("xs[0:2] == ['a', 'b']", context).IsTruthy);

        // A filter chain applies to a literal source exactly as it does to a context value.
        Assert.Equal(2L, Eval("[1, 2] | length").IntValue);
        Assert.Equal("1-2", Render("{{ [1, 2] | join('-') }}"));

        // Slicing a literal map stays the existing maps-cannot-be-sliced error.
        Assert.Throws<MinjaRenderException>(() => Eval("{'a': 1}[:]"));
    }

    [Fact]
    public void Literals_JsonDumpKeepsInsertionOrderAndEmptyShapes()
    {
        IReadOnlyDictionary<string, MinjaValue> context = Context(
            ("b", MinjaValue.Of(2L)),
            ("a", MinjaValue.Of("x")));

        Assert.Equal("{\"b\": 2, \"xs\": [1, {\"k\": null}], \"empty\": []}", Eval("{'b': b, 'xs': [1, {'k': None}], 'empty': []}", context).ToJson());
        Assert.Equal("[[1], {}]", Eval("[[1], {}]").ToJson());

        // A set statement bound to a literal dumps exactly like its parsed form.
        Assert.Equal("{\"first\": 1, \"second\": [2]}", Render("{% set metadata = {'first': 1, 'second': [2]} %}{{ metadata | tojson }}"));
    }

    // ---------------------------------------------------------------- helpers

    private static MinjaValue Eval(string expression, IReadOnlyDictionary<string, MinjaValue>? context = null) =>
        MinjaEvaluator.Evaluate(expression, context);

    private static string Render(string template, IReadOnlyDictionary<string, MinjaValue>? context = null) =>
        MinjaEvaluator.Render(template, context);

    private static MinjaValue ListOf(params MinjaValue[] items) => MinjaValue.List(items);

    private static MinjaValue MapOf(params (string Key, MinjaValue Value)[] entries) =>
        MinjaValue.Map(entries.Select(entry => new KeyValuePair<string, MinjaValue>(entry.Key, entry.Value)));

    private static IReadOnlyDictionary<string, MinjaValue> Context(params (string Name, MinjaValue Value)[] entries) =>
        entries.ToDictionary(entry => entry.Name, entry => entry.Value);
}

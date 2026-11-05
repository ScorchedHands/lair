using Lair.Templating.Minja;

namespace Lair.Tests;

/// <summary>
/// Syntax goldens for the managed minja parser: supported constructs parse into inspectable ASTs, comments and
/// whitespace elision are represented without literal comment leakage, unsupported non-minja constructs fail with an
/// actionable error naming the construct and its 0-based offset, and the Llama-3.2 {% set %} pattern that previously
/// broke ChatTemplateRenderer at template offset 37 parses as a set statement.
/// </summary>
[Trait("Category", "MinjaSyntax")]
public sealed class MinjaSyntaxTests
{
    // ---------------------------------------------------------------- supported constructs

    [Fact]
    public void PlainText_ParsesAsSingleTextNode()
    {
        TemplateNode root = MinjaParser.Parse("hello world");

        TextNode text = Assert.IsType<TextNode>(Assert.Single(root.Nodes));
        Assert.Equal("hello world", text.Text);
        Assert.Equal(0, text.Offset);
    }

    [Fact]
    public void OutputTag_ParsesLiterals()
    {
        TemplateNode root = MinjaParser.Parse("{{ 'hi' }} {{ 42 }} {{ True }} {{ None }}");

        // Whitespace between output tags is real output, so the tree holds three gap text nodes as well.
        List<OutputNode> outputs = [.. Outputs(root.Nodes)];
        Assert.Equal(4, outputs.Count);
        Assert.IsType<StringLiteral>(outputs[0].Expression);
        IntegerLiteral number = Assert.IsType<IntegerLiteral>(outputs[1].Expression);
        Assert.Equal(42L, number.Value);
        BoolLiteral flag = Assert.IsType<BoolLiteral>(outputs[2].Expression);
        Assert.True(flag.Value);
        Assert.IsType<NoneLiteral>(outputs[3].Expression);
    }

    [Fact]
    public void OutputTag_ParsesAttributeAndIndexAccess()
    {
        TemplateNode root = MinjaParser.Parse("{{ message['role'] }} {{ message.content }}");

        List<OutputNode> outputs = [.. Outputs(root.Nodes)];
        Assert.Equal(2, outputs.Count);

        IndexExpression index = Assert.IsType<IndexExpression>(outputs[0].Expression);
        Assert.False(index.Slice);
        NameExpression target = Assert.IsType<NameExpression>(index.Target);
        StringLiteral key = Assert.IsType<StringLiteral>(index.Start!);
        Assert.Equal("message", target.Name);
        Assert.Equal("role", key.Value);

        AttributeExpression attribute = Assert.IsType<AttributeExpression>(outputs[1].Expression);
        Assert.Equal("content", attribute.Name);
    }

    [Fact]
    public void OutputTag_ParsesIndexesAndSlices()
    {
        TemplateNode plain = MinjaParser.Parse("{{ items[2] }}");
        IndexExpression index = Assert.IsType<IndexExpression>(Output(plain, 0).Expression);
        Assert.False(index.Slice);
        Assert.NotNull(index.Start);

        TemplateNode head = MinjaParser.Parse("{{ items[:3] }}");
        IndexExpression prefix = Assert.IsType<IndexExpression>(Output(head, 0).Expression);
        Assert.True(prefix.Slice);
        Assert.Null(prefix.Start);
        Assert.NotNull(prefix.Stop);

        TemplateNode stepped = MinjaParser.Parse("{{ items[::2] }}");
        IndexExpression stepOnly = Assert.IsType<IndexExpression>(Output(stepped, 0).Expression);
        Assert.True(stepOnly.Slice);
        Assert.Null(stepOnly.Start);
        Assert.NotNull(stepOnly.Step);

        TemplateNode full = MinjaParser.Parse("{{ items[1:4:2] }}");
        IndexExpression slice = Assert.IsType<IndexExpression>(Output(full, 0).Expression);
        Assert.True(slice.Slice);
        Assert.NotNull(slice.Start);
        Assert.NotNull(slice.Stop);
        Assert.NotNull(slice.Step);
    }

    [Fact]
    public void OutputTag_ParsesCallsWithPositionalAndNamedArguments()
    {
        TemplateNode root = MinjaParser.Parse("{{ range(0, 5, step=2) }}");

        CallExpression call = Assert.IsType<CallExpression>(Output(root, 0).Expression);
        NameExpression callee = Assert.IsType<NameExpression>(call.Callee);
        Assert.Equal("range", callee.Name);
        Assert.Equal(3, call.Arguments.Count);
        Assert.Null(call.Arguments[0].Name);
        Assert.Null(call.Arguments[1].Name);
        Assert.Equal("step", call.Arguments[2].Name);
    }

    [Fact]
    public void OutputTag_ParsesFilterChain()
    {
        TemplateNode root = MinjaParser.Parse("{{ value | trim | length }}");

        FilterChainExpression chain = Assert.IsType<FilterChainExpression>(Output(root, 0).Expression);
        NameExpression source = Assert.IsType<NameExpression>(chain.Source);
        Assert.Equal(2, chain.Filters.Count);
        Assert.Equal("trim", chain.Filters[0].Name);
        Assert.Equal("length", chain.Filters[1].Name);

        TemplateNode withArgs = MinjaParser.Parse("{{ items | join(', ') }}");
        FilterChainExpression filtered = Assert.IsType<FilterChainExpression>(Output(withArgs, 0).Expression);
        CallArgument argument = Assert.Single(filtered.Filters[0].Arguments);
        StringLiteral separator = Assert.IsType<StringLiteral>(argument.Value);
        Assert.Equal(", ", separator.Value);
    }

    [Fact]
    public void OutputTag_ParsesTestsIncludingNegated()
    {
        TemplateNode defined = MinjaParser.Parse("{{ x is defined }}");
        TestExpression test = Assert.IsType<TestExpression>(Output(defined, 0).Expression);
        Assert.Equal("defined", test.Name);
        Assert.False(test.Negated);

        TemplateNode negated = MinjaParser.Parse("{{ messages[0] is not none }}");
        TestExpression notTest = Assert.IsType<TestExpression>(Output(negated, 0).Expression);
        Assert.True(notTest.Negated);
    }

    [Fact]
    public void OutputTag_ParsesOperatorsAndMembership()
    {
        TemplateNode arithmetic = MinjaParser.Parse("{{ a + b * -2 }}");
        BinaryExpression sum = Assert.IsType<BinaryExpression>(Output(arithmetic, 0).Expression);
        Assert.Equal(BinaryOperator.Add, sum.Operator);

        TemplateNode boolean = MinjaParser.Parse("{{ not x and y }}");
        BinaryExpression and = Assert.IsType<BinaryExpression>(Output(boolean, 0).Expression);
        UnaryExpression notX = Assert.IsType<UnaryExpression>(and.Left);
        Assert.Equal(UnaryOperator.Not, notX.Operator);

        TemplateNode comparison = MinjaParser.Parse("{{ a >= b }}");
        ComparisonExpression greaterOrEqual = Assert.IsType<ComparisonExpression>(Output(comparison, 0).Expression);
        Assert.Equal(CompareOperator.Gte, greaterOrEqual.Operator);

        TemplateNode membership = MinjaParser.Parse("{{ 'user' in roles }}");
        MembershipExpression inside = Assert.IsType<MembershipExpression>(Output(membership, 0).Expression);
        Assert.False(inside.Negated);

        TemplateNode notInside = MinjaParser.Parse("{{ 'tool' not in names }}");
        MembershipExpression outside = Assert.IsType<MembershipExpression>(Output(notInside, 0).Expression);
        Assert.True(outside.Negated);
    }

    [Fact]
    public void OutputTag_ParsesTernaryRightAssociatively()
    {
        TemplateNode root = MinjaParser.Parse("{{ a if b else c if d else e }}");

        ConditionalExpression outer = Assert.IsType<ConditionalExpression>(Output(root, 0).Expression);
        NameExpression condition = Assert.IsType<NameExpression>(outer.Condition);
        Assert.Equal("b", condition.Name);

        ConditionalExpression inner = Assert.IsType<ConditionalExpression>(outer.ElseBranch);
        NameExpression trueBranch = Assert.IsType<NameExpression>(inner.TrueBranch);
        Assert.Equal("c", trueBranch.Name);
    }

    [Fact]
    public void IfStatement_ParsesElifAndElseBranches()
    {
        TemplateNode root = MinjaParser.Parse("{% if role == 'system' %}S{% elif role == 'user' %}U{% else %}O{% endif %}");

        IfStatement statement = Assert.IsType<IfStatement>(Assert.Single(root.Nodes));

        // if/elif conditions become branches; the else body is modelled separately.
        Assert.Equal(2, statement.Branches.Count);
        Assert.NotNull(statement.ElseBody);

        ComparisonExpression first = Assert.IsType<ComparisonExpression>(statement.Branches[0].Condition);
        Assert.Equal(CompareOperator.Eq, first.Operator);

        TextNode second = Assert.IsType<TextNode>(Assert.Single(statement.Branches[1].Body));
        Assert.Equal("U", second.Text);
    }

    [Fact]
    public void ForStatement_ParsesNamesTupleTargetsRecursiveAndLoopCondition()
    {
        TemplateNode plain = MinjaParser.Parse("{% for message in messages %}{{ message }}{% endfor %}");
        ForStatement loop = Assert.IsType<ForStatement>(Assert.Single(plain.Nodes));
        NameTarget target = Assert.IsType<NameTarget>(loop.Target);
        Assert.Equal("message", target.Name);

        TemplateNode tupleTemplate = MinjaParser.Parse("{% for (k, v) in pairs %}{{ k }}{% endfor %}");
        ForStatement destructuring = Assert.IsType<ForStatement>(Assert.Single(tupleTemplate.Nodes));
        TupleTarget pair = Assert.IsType<TupleTarget>(destructuring.Target);
        Assert.Equal(2, pair.Items.Count);

        TemplateNode recursiveTemplate = MinjaParser.Parse("{% for item in tree recursive %}{{ item }}{% endfor %}");
        ForStatement deepLoop = Assert.IsType<ForStatement>(Assert.Single(recursiveTemplate.Nodes));
        Assert.True(deepLoop.Recursive);

        TemplateNode filteredTemplate = MinjaParser.Parse("{% for tool in tools if tool.enabled %}{{ tool }}{% endfor %}");
        ForStatement conditional = Assert.IsType<ForStatement>(Assert.Single(filteredTemplate.Nodes));
        AttributeExpression condition = Assert.IsType<AttributeExpression>(conditional.LoopCondition!);
        Assert.Equal("enabled", condition.Name);
    }

    [Fact]
    public void SetStatement_ParsesPlainNamespacedAndDestructuringTargets()
    {
        TemplateNode plain = MinjaParser.Parse("{% set tools = custom_tools %}");
        SetStatement assignment = Assert.IsType<SetStatement>(Assert.Single(plain.Nodes));
        SetTarget target = Assert.Single(assignment.Targets);
        Assert.Equal("tools", target.Name);

        TemplateNode namespacedTemplate = MinjaParser.Parse("{% set ns.attr = 1 %}");
        SetStatement namespaceAssignment = Assert.IsType<SetStatement>(Assert.Single(namespacedTemplate.Nodes));
        SetTarget dotted = Assert.Single(namespaceAssignment.Targets);
        Assert.Equal(["ns", "attr"], dotted.Attributes.ToArray());

        TemplateNode destructured = MinjaParser.Parse("{% set a, b = pair %}");
        SetStatement multi = Assert.IsType<SetStatement>(Assert.Single(destructured.Nodes));
        Assert.Equal(2, multi.Targets.Count);
    }

    [Fact]
    public void MacroCallAndFilterBlocks_ParseWithBodies()
    {
        TemplateNode macroTemplate = MinjaParser.Parse("{% macro render(name='x') %}[{{ name }}]{% endmacro %}");
        MacroStatement macro = Assert.IsType<MacroStatement>(Assert.Single(macroTemplate.Nodes));
        Assert.Equal("render", macro.Name);
        MacroParameter parameter = Assert.Single(macro.Parameters);
        StringLiteral defaultValue = Assert.IsType<StringLiteral>(parameter.DefaultValue!);
        Assert.Equal("x", defaultValue.Value);

        TemplateNode callTemplate = MinjaParser.Parse("{% call render('a') %}body{% endcall %}");
        CallStatement call = Assert.IsType<CallStatement>(Assert.Single(callTemplate.Nodes));
        NameExpression callee = Assert.IsType<NameExpression>(call.Callable);
        Assert.Equal("render", callee.Name);

        TemplateNode filterTemplate = MinjaParser.Parse("{% filter upper %}shout{% else %}quiet{% endfilter %}");
        FilterStatement filter = Assert.IsType<FilterStatement>(Assert.Single(filterTemplate.Nodes));
        Assert.Equal("upper", filter.Name);
        Assert.NotNull(filter.ElseBody);
    }

    [Fact]
    public void BreakAndContinue_ParseInsideMacroBodies()
    {
        TemplateNode root = MinjaParser.Parse("{% macro loop(items) %}{% for item in items %}{% if skip %}{% continue %}{% endif %}{% break %}{% endfor %}{% endmacro %}");

        Assert.True(AnyNode(root.Nodes, static node => node is ContinueStatement));
        Assert.True(AnyNode(root.Nodes, static node => node is BreakStatement));
    }

    // ---------------------------------------------------------------- comments and whitespace elision

    [Fact]
    public void Comment_StaysInTreeAndNeverLeaksAsText()
    {
        TemplateNode root = MinjaParser.Parse("a{# hidden note #}b");

        Assert.Contains(root.Nodes, static node => node is CommentNode);
        Assert.Equal("ab", string.Concat(Texts(root.Nodes).Select(text => text.Text)));
    }

    [Fact]
    public void WhitespaceOnlyRun_BetweenTagsIsRealOutput()
    {
        TemplateNode root = MinjaParser.Parse("{{ a }} {{ b }}");

        TextNode gap = Assert.IsType<TextNode>(root.Nodes[1]);
        Assert.Equal(" ", gap.Text);
    }

    [Fact]
    public void StatementTrimLeft_ElidesPrecedingWhitespace()
    {
        TemplateNode root = MinjaParser.Parse("Hello {%- if name %} world {% endif %}");

        IfStatement statement = Assert.IsType<IfStatement>(root.Nodes[1]);
        Assert.True(statement.TrimLeft);

        TextNode greeting = Assert.IsType<TextNode>(root.Nodes[0]);
        Assert.Equal("Hello", greeting.Text);

        // The branch's own leading space is untouched: elision only spans text adjacent to the marker.
        TextNode inside = Assert.IsType<TextNode>(Assert.Single(statement.Branches[0].Body));
        Assert.Equal(" world ", inside.Text);
    }

    [Fact]
    public void StatementTrimRight_ElidesFollowingWhitespace()
    {
        TemplateNode root = MinjaParser.Parse("{% if x -%}   body{% endif %} tail");

        IfStatement statement = Assert.IsType<IfStatement>(root.Nodes[0]);
        Assert.True(statement.TrimRight);

        TextNode inside = Assert.IsType<TextNode>(Assert.Single(statement.Branches[0].Body));
        Assert.Equal("body", inside.Text);
    }

    [Fact]
    public void EndTagTrimLeft_ElidesTrailingWhitespaceOfPrecedingBody()
    {
        TemplateNode root = MinjaParser.Parse("{% if x %}body   {%- endif %}tail");

        IfStatement statement = Assert.IsType<IfStatement>(root.Nodes[0]);
        TextNode inside = Assert.IsType<TextNode>(Assert.Single(statement.Branches[0].Body));
        Assert.Equal("body", inside.Text);
    }

    [Fact]
    public void CommentTrimMarkers_ElideNeighbouringWhitespace()
    {
        TemplateNode root = MinjaParser.Parse("a   {#- hidden -#}   b");

        CommentNode comment = Assert.IsType<CommentNode>(root.Nodes[1]);
        Assert.True(comment.TrimLeft);
        Assert.True(comment.TrimRight);
        Assert.Equal("ab", string.Concat(Texts(root.Nodes).Select(text => text.Text)));
    }

    // ---------------------------------------------------------------- known Llama-3.2 pattern

    [Fact]
    public void LlamaStyle_SetTools_ParsesAsSetStatement()
    {
        TemplateNode root = MinjaParser.Parse("{%- set tools = custom_tools %}");

        SetStatement statement = Assert.IsType<SetStatement>(Assert.Single(root.Nodes));
        SetTarget target = Assert.Single(statement.Targets);
        Assert.Equal("tools", target.Name);
        NameExpression value = Assert.IsType<NameExpression>(statement.Value);
        Assert.Equal("custom_tools", value.Name);
    }

    [Fact]
    public void LlamaStyle_TemplateWithSetToolsAndNamespace_ParsesEndToEnd()
    {
        string template = "{{ bos_token }}{%- if messages[0]['role'] == 'system' %}{% set first_system_msg = true %}{{ raise_exception('Only system message is not supported.') }}{% endif %}{%- for message in messages %}{% set content = message['content'] %}{%- endfor %}";

        TemplateNode root = MinjaParser.Parse(template);

        Assert.True(AnyNode(root.Nodes, static node => node is SetStatement));
        Assert.True(AnyNode(root.Nodes, static node => node is ForStatement));
        Assert.Contains(Outputs(root.Nodes), output => output.Expression is CallExpression call && NameOf(call.Callee) == "raise_exception");
    }

    // ---------------------------------------------------------------- rejections

    [Theory]
    [InlineData("raw")]
    [InlineData("block")]
    [InlineData("include")]
    [InlineData("extends")]
    public void UnsupportedJinjaConstruct_NamesItWithOffset(string construct)
    {
        string template = "before {% " + construct + " page %} after";

        MinjaSyntaxException exception = Assert.Throws<MinjaSyntaxException>(() => MinjaParser.Parse(template));

        Assert.Equal(construct, exception.Construct);
        Assert.Equal("before ".Length, exception.Position);
        Assert.Contains(construct, exception.Message);
    }

    [Fact]
    public void BlockTypo_RejectsAsBlock()
    {
        string template = "{% blcok body %}";

        MinjaSyntaxException exception = Assert.Throws<MinjaSyntaxException>(() => MinjaParser.Parse(template));

        Assert.Equal("block", exception.Construct);
        Assert.Equal(0, exception.Position);
    }

    [Fact]
    public void TupleLiteral_OutsideCallArgumentsIsRejected()
    {
        MinjaSyntaxException inOutput = Assert.Throws<MinjaSyntaxException>(() => MinjaParser.Parse("{{ (1, 2) }}"));
        Assert.Contains("tuple", inOutput.Message);

        MinjaSyntaxException inSet = Assert.Throws<MinjaSyntaxException>(() => MinjaParser.Parse("{% set pair = (1, 2) %}"));
        Assert.Contains("tuple", inSet.Message);
    }

    [Theory]
    [InlineData("{{ 'yes' if enabled }}")]
    [InlineData("{{ a if b else c | default if flag }}")]
    public void TernaryLessIf_IsRejected(string template)
    {
        MinjaSyntaxException exception = Assert.Throws<MinjaSyntaxException>(() => MinjaParser.Parse(template));

        Assert.Contains("without 'else'", exception.Message);
        Assert.True(exception.Position >= 0);
    }

    [Theory]
    [InlineData("{{ x")]
    [InlineData("{# never closed")]
    [InlineData("{% endif %}")]
    [InlineData("{% endfor junk %}")]
    [InlineData("{% for message messages %}")]
    [InlineData("{% set target %}")]
    [InlineData("a {{ b + }} c")]
    public void MalformedTags_ThrowActionableErrors(string template)
    {
        MinjaSyntaxException exception = Assert.Throws<MinjaSyntaxException>(() => MinjaParser.Parse(template));

        Assert.True(exception.Position >= 0);
        Assert.False(string.IsNullOrWhiteSpace(exception.Message));
    }

    [Fact]
    public void BreakOutsideBlock_IsRejected()
    {
        string template = "{% break %}";

        MinjaSyntaxException exception = Assert.Throws<MinjaSyntaxException>(() => MinjaParser.Parse(template));

        Assert.Contains("macro/call/filter", exception.Message);
        Assert.Equal(0, exception.Position);
    }

    // ---------------------------------------------------------------- helpers

    private static OutputNode Output(TemplateNode root, int index) => Assert.IsType<OutputNode>(root.Nodes[index]);

    private static string NameOf(ExpressionNode expression) => expression is NameExpression name ? name.Name : string.Empty;

    private static IEnumerable<TextNode> Texts(IEnumerable<MinjaNode> nodes)
    {
        foreach (MinjaNode node in nodes)
        {
            switch (node)
            {
                case TextNode text:
                    yield return text;
                    break;

                case IfStatement statement:
                    foreach (IfBranch branch in statement.Branches)
                    {
                        foreach (TextNode bodyText in Texts(branch.Body))
                            yield return bodyText;
                    }

                    if (statement.ElseBody is not null)
                    {
                        foreach (TextNode elseText in Texts(statement.ElseBody))
                            yield return elseText;
                    }

                    break;

                case ForStatement loop:
                    foreach (TextNode bodyText in Texts(loop.Body))
                        yield return bodyText;
                    break;

                case MacroStatement macro:
                    foreach (TextNode bodyText in Texts(macro.Body))
                        yield return bodyText;
                    break;

                case CallStatement call:
                    foreach (TextNode bodyText in Texts(call.Body))
                        yield return bodyText;
                    break;

                case FilterStatement filter:
                    foreach (TextNode bodyText in Texts(filter.Body))
                        yield return bodyText;

                    if (filter.ElseBody is not null)
                    {
                        foreach (TextNode elseText in Texts(filter.ElseBody))
                            yield return elseText;
                    }

                    break;
            }
        }
    }

    private static IEnumerable<OutputNode> Outputs(IEnumerable<MinjaNode> nodes)
    {
        foreach (MinjaNode node in nodes)
        {
            switch (node)
            {
                case OutputNode output:
                    yield return output;
                    break;

                case IfStatement statement:
                    foreach (IfBranch branch in statement.Branches)
                    {
                        foreach (OutputNode bodyOutput in Outputs(branch.Body))
                            yield return bodyOutput;
                    }

                    if (statement.ElseBody is not null)
                    {
                        foreach (OutputNode elseOutput in Outputs(statement.ElseBody))
                            yield return elseOutput;
                    }

                    break;

                case ForStatement loop:
                    foreach (OutputNode bodyOutput in Outputs(loop.Body))
                        yield return bodyOutput;
                    break;

                case MacroStatement macro:
                    foreach (OutputNode bodyOutput in Outputs(macro.Body))
                        yield return bodyOutput;
                    break;

                case CallStatement call:
                    foreach (OutputNode bodyOutput in Outputs(call.Body))
                        yield return bodyOutput;
                    break;

                case FilterStatement filter:
                    foreach (OutputNode bodyOutput in Outputs(filter.Body))
                        yield return bodyOutput;

                    if (filter.ElseBody is not null)
                    {
                        foreach (OutputNode elseOutput in Outputs(filter.ElseBody))
                            yield return elseOutput;
                    }

                    break;
            }
        }
    }

    private static bool AnyNode(IEnumerable<MinjaNode> nodes, Func<MinjaNode, bool> predicate)
    {
        foreach (MinjaNode node in nodes)
        {
            if (predicate(node))
                return true;

            switch (node)
            {
                case IfStatement statement:
                    foreach (IfBranch branch in statement.Branches)
                    {
                        if (AnyNode(branch.Body, predicate))
                            return true;
                    }

                    if (statement.ElseBody is not null && AnyNode(statement.ElseBody, predicate))
                        return true;
                    break;

                case ForStatement loop when AnyNode(loop.Body, predicate):
                    return true;

                case MacroStatement macro when AnyNode(macro.Body, predicate):
                    return true;

                case CallStatement call when AnyNode(call.Body, predicate):
                    return true;

                case FilterStatement filter:
                    if (AnyNode(filter.Body, predicate))
                        return true;

                    if (filter.ElseBody is not null && AnyNode(filter.ElseBody, predicate))
                        return true;
                    break;
            }
        }

        return false;
    }
}

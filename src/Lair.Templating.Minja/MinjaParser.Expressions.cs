namespace Lair.Templating.Minja;

public sealed partial class MinjaParser
{
    // ---------------------------------------------------------------- expression phase
    // Precedence (loosest first): ternary conditional, or, and, not, comparisons/tests/membership, + -, * / // %, unary -, **, postfix call/index/attribute with filter chains.
    // Filters bind tighter than arithmetic: {{ a | f + b }} parses as {{ (a | f) + b }}, matching how chat templates apply filters to operands.
    //
    // Recursion guard: every entry point that can nest (expression entry, primary, literal body, index body,
    // tuple target, argument list) claims a parser frame, so adversarial nesting fails as a bounded
    // "parser depth" syntax error instead of exhausting the .NET stack.

    private void EnterExpressionFrame() => _limits.EnterParserFrame(Math.Min(_pos, Math.Max(_source.Length - 1, 0)));

    private ExpressionNode ParseExpression() => ParseConditional();

    /// <summary>Ternary conditional, right-associative. Ternary-less 'a if b' is rejected as non-minja.</summary>
    private ExpressionNode ParseConditional()
    {
        EnterExpressionFrame();

        try
        {
            int start = _pos;
            ExpressionNode value = ParseOrLevel();

            SkipWhitespace();
            if (PeekKeyword() != "if")
                return value;

            MatchKeyword("if");
            ExpressionNode condition = ParseOrLevel();

            SkipWhitespace();
            if (!MatchKeyword("else"))
                throw new MinjaSyntaxException("'a if b' without 'else'", start);

            // Right recursion: a if b else c if d else e nests as a if b else (c if d else e).
            ExpressionNode elseBranch = ParseConditional();

            return new ConditionalExpression(value, condition, elseBranch) { Offset = start };
        }
        finally
        {
            _limits.ExitParserFrame();
        }
    }

    private ExpressionNode ParseOrLevel()
    {
        int start = _pos;
        ExpressionNode left = ParseAndLevel();

        while (MatchKeyword("or"))
            left = new BinaryExpression(BinaryOperator.Or, left, ParseAndLevel()) { Offset = start };

        return left;
    }

    private ExpressionNode ParseAndLevel()
    {
        int start = _pos;
        ExpressionNode left = ParseNotLevel();

        while (MatchKeyword("and"))
            left = new BinaryExpression(BinaryOperator.And, left, ParseNotLevel()) { Offset = start };

        return left;
    }

    private ExpressionNode ParseNotLevel()
    {
        int start = _pos;
        if (!MatchKeyword("not"))
            return ParseComparisonLevel();

        return new UnaryExpression(UnaryOperator.Not, ParseNotLevel()) { Offset = start };
    }

    /// <summary>Non-chaining comparisons plus membership and test operators: == != < <= > >= in 'not in' is 'is not'.</summary>
    private ExpressionNode ParseComparisonLevel()
    {
        int start = _pos;
        ExpressionNode left = ParseConcatLevel();

        while (true)
        {
            SkipWhitespace();

            char c = PeekAt(_pos);

            if (c == '=' && PeekAt(_pos + 1) == '=')
            {
                _pos += 2;
                left = new ComparisonExpression(left, CompareOperator.Eq, ParseConcatLevel()) { Offset = start };
                continue;
            }

            if (c == '!' && PeekAt(_pos + 1) == '=')
            {
                _pos += 2;
                left = new ComparisonExpression(left, CompareOperator.Ne, ParseConcatLevel()) { Offset = start };
                continue;
            }

            if (c == '>' && PeekAt(_pos + 1) == '=')
            {
                _pos += 2;
                left = new ComparisonExpression(left, CompareOperator.Gte, ParseConcatLevel()) { Offset = start };
                continue;
            }

            if (c == '<' && PeekAt(_pos + 1) == '=')
            {
                _pos += 2;
                left = new ComparisonExpression(left, CompareOperator.Lte, ParseConcatLevel()) { Offset = start };
                continue;
            }

            if (c == '>')
            {
                _pos++;
                left = new ComparisonExpression(left, CompareOperator.Gt, ParseConcatLevel()) { Offset = start };
                continue;
            }

            if (c == '<')
            {
                _pos++;
                left = new ComparisonExpression(left, CompareOperator.Lt, ParseConcatLevel()) { Offset = start };
                continue;
            }

            string keyword = PeekKeyword();

            if (keyword == "in")
            {
                MatchKeyword("in");
                left = new MembershipExpression(left, ParseConcatLevel(), Negated: false) { Offset = start };
                continue;
            }

            if (keyword == "not" && KeywordAtEquals(WordEnd(_pos), "in"))
            {
                // This 'not' belongs to the membership test ('not in'), not to a right-hand unary operand.
                MatchKeyword("not");
                MatchKeyword("in");
                left = new MembershipExpression(left, ParseConcatLevel(), Negated: true) { Offset = start };
                continue;
            }

            if (keyword == "is")
            {
                int testStart = _pos;
                MatchKeyword("is");
                SkipWhitespace();
                bool negatedTest = MatchKeyword("not");
                string testName = ReadName(testStart);
                List<CallArgument> arguments = MatchOpenParen() ? ParseArguments(testStart) : [];

                left = new TestExpression(testName, arguments, negatedTest, left) { Offset = start };
                continue;
            }

            return left;
        }
    }

    private ExpressionNode ParseConcatLevel()
    {
        int start = _pos;
        ExpressionNode left = ParseMulLevel();

        while (true)
        {
            SkipWhitespace();

            if (PeekAt(_pos) == '+')
            {
                _pos++;
                left = new BinaryExpression(BinaryOperator.Add, left, ParseMulLevel()) { Offset = start };
                continue;
            }

            // A dash directly before %} or }} is the tag's own trim marker, never a minus operator.
            if (PeekAt(_pos) == '-' && PeekAt(_pos + 1) != '%' && PeekAt(_pos + 1) != '}')
            {
                _pos++;
                left = new BinaryExpression(BinaryOperator.Sub, left, ParseMulLevel()) { Offset = start };
                continue;
            }

            return left;
        }
    }

    private ExpressionNode ParseMulLevel()
    {
        int start = _pos;
        ExpressionNode left = ParseUnaryLevel();

        while (true)
        {
            SkipWhitespace();

            if (PeekAt(_pos) == '*' && PeekAt(_pos + 1) != '*')
            {
                _pos++;
                left = new BinaryExpression(BinaryOperator.Mul, left, ParseUnaryLevel()) { Offset = start };
                continue;
            }

            if (PeekAt(_pos) == '/' && PeekAt(_pos + 1) == '/')
            {
                _pos += 2;
                left = new BinaryExpression(BinaryOperator.FloorDiv, left, ParseUnaryLevel()) { Offset = start };
                continue;
            }

            // '%' is modulo only when it is not the start of a %} tag close.
            if (PeekAt(_pos) == '/' || (PeekAt(_pos) == '%' && PeekAt(_pos + 1) != '}'))
            {
                BinaryOperator binaryOperator = _source[_pos] == '/' ? BinaryOperator.Div : BinaryOperator.Mod;
                _pos++;
                left = new BinaryExpression(binaryOperator, left, ParseUnaryLevel()) { Offset = start };
                continue;
            }

            return left;
        }
    }

    private ExpressionNode ParseUnaryLevel()
    {
        int start = _pos;
        SkipWhitespace();

        if (PeekAt(_pos) == '-')
        {
            _pos++;
            return new UnaryExpression(UnaryOperator.Negate, ParseUnaryLevel()) { Offset = start };
        }

        return ParsePowerLevel();
    }

    private ExpressionNode ParsePostfixChain(ExpressionNode target)
    {
        int start = _pos;

        while (true)
        {
            // Attribute access must stay adjacent; calls, indexes and filters may be spaced.
            if (PeekAt(_pos) == '.')
            {
                _pos++;
                string attribute = ReadName(start);
                target = new AttributeExpression(target, attribute) { Offset = start };
                continue;
            }

            if (MatchOpenParen())
            {
                List<CallArgument> arguments = ParseArguments(start);
                target = new CallExpression(target, arguments) { Offset = start };
                continue;
            }

            if (PeekAfterWhitespace('['))
            {
                SkipWhitespace();
                _pos++;
                target = ParseIndexOrSliceBody(target, start);
                continue;
            }

            if (PeekAfterWhitespace('|'))
            {
                SkipWhitespace();

                List<FilterApplication> filters = [];

                while (PeekAfterWhitespace('|'))
                {
                    // Whitespace before a chained pipe is allowed, so step over it before claiming the pipe itself.
                    SkipWhitespace();

                    int filterStart = _pos;
                    _pos++;

                    SkipWhitespace();
                    string name = ReadName(filterStart);
                    List<CallArgument> arguments = MatchOpenParen() ? ParseArguments(filterStart) : [];

                    filters.Add(new FilterApplication(name, arguments) { Offset = filterStart });
                }

                target = new FilterChainExpression(target, filters) { Offset = start };
                continue;
            }

            return target;
        }
    }

    private bool PeekAfterWhitespace(char candidate)
    {
        int pos = _pos;

        while (char.IsWhiteSpace(CharAt(pos)))
            pos++;

        return CharAt(pos) == candidate;
    }

    /// <summary>Parses the body of [...] after the opening bracket: a plain index or one of the slice forms [a:b], [a:], [:b], [::s].</summary>
    private ExpressionNode ParseIndexOrSliceBody(ExpressionNode target, int start)
    {
        EnterExpressionFrame();

        try
        {
            SkipWhitespace();

            if (PeekAt(_pos) == ']')
                throw Malformed(_pos, "empty index");

            // A body that opens with ':' has no start bound; parse it as the colon-first slice form.
            if (PeekAt(_pos) == ':')
                return ParseColonSliceBody(target, start);

            ExpressionNode? first = ParseExpression();

            SkipWhitespace();

            if (PeekAt(_pos) == ']')
            {
                _pos++;
                return new IndexExpression(target, first, Stop: null, Step: null, Slice: false) { Offset = start };
            }

            Expect(':', start);

            ExpressionNode? stop = PeekAny(']', ':') ? null : ParseExpression();
            ExpressionNode? step = null;

            SkipWhitespace();
            if (PeekAt(_pos) == ':')
            {
                _pos++;
                SkipWhitespace();
                step = PeekAt(_pos) == ']' ? null : ParseExpression();
            }

            Expect(']', start);

            return new IndexExpression(target, first, stop, step, Slice: true) { Offset = start };
        }
        finally
        {
            _limits.ExitParserFrame();
        }
    }

    /// <summary>
    /// Parses the body of [...] after the opening bracket when it starts directly with a colon: the [:b] and
    /// [::s] forms have no start bound, so the body is parsed as an empty start followed by stop and step.
    /// </summary>
    private ExpressionNode ParseColonSliceBody(ExpressionNode target, int start)
    {
        EnterExpressionFrame();

        try
        {
            Expect(':', start);

            ExpressionNode? stop = PeekAny(']', ':') ? null : ParseExpression();
            ExpressionNode? step = null;

            SkipWhitespace();
            if (PeekAt(_pos) == ':')
            {
                _pos++;
                SkipWhitespace();
                step = PeekAt(_pos) == ']' ? null : ParseExpression();
            }

            Expect(']', start);

            return new IndexExpression(target, Start: null, stop, step, Slice: true) { Offset = start };
        }
        finally
        {
            _limits.ExitParserFrame();
        }
    }

    private bool PeekAny(params char[] candidates)
    {
        SkipWhitespace();

        foreach (char candidate in candidates)
        {
            if (PeekAt(_pos) == candidate)
                return true;
        }

        return false;
    }

    private void Expect(char expected, int contextOffset)
    {
        SkipWhitespace();

        if (PeekAt(_pos) != expected)
            throw Malformed(Math.Min(_pos, Math.Max(_source.Length - 1, 0)), $"expected '{expected}'");

        _pos++;
    }

    private ExpressionNode ParsePowerLevel()
    {
        int start = _pos;
        ExpressionNode left = ParsePostfixChain(ParsePrimary());

        SkipWhitespace();
        if (PeekAt(_pos) == '*' && PeekAt(_pos + 1) == '*')
        {
            _pos += 2;
            // ** is right-associative.
            return new BinaryExpression(BinaryOperator.Pow, left, ParseUnaryLevel()) { Offset = start };
        }

        return left;
    }

    private ExpressionNode ParsePrimary()
    {
        EnterExpressionFrame();

        try
        {
            SkipWhitespace();

            int start = _pos;
            char c = PeekAt(_pos);

            if (c is >= '0' and <= '9')
                return ParseNumberLiteral(start);

            if (c == '\'' || c == '"')
                return new StringLiteral(ParseQuotedString(c, start)) { Offset = start };

            if (MatchOpenParen())
            {
                ExpressionNode inner = ParseExpression();

                SkipWhitespace();
                if (PeekAt(_pos) == ',')
                    throw new MinjaSyntaxException("tuple literal", start);

                Expect(')', start);
                return inner;
            }

            // '[' and '{' open inline literals only in primary position: postfix indexing peeks at a bracket after an existing expression, so it never reaches ParsePrimary.
            if (c == '[')
            {
                _pos++;
                return ParsePostfixChain(ParseListLiteralBody(start));
            }

            if (c == '{')
            {
                _pos++;
                return ParsePostfixChain(ParseMapLiteralBody(start));
            }

            if (IsWordStart(c))
            {
                string word = ReadWord();

                return word switch
                {
                    "True" => new BoolLiteral(true) { Offset = start },
                    "False" => new BoolLiteral(false) { Offset = start },
                    // Jinja/minja accept the lowercase spellings as the same boolean literals; real chat
                    // templates (e.g. Llama-3.2's tools_in_user_message default) rely on them.
                    "true" => new BoolLiteral(true) { Offset = start },
                    "false" => new BoolLiteral(false) { Offset = start },
                    "None" => new NoneLiteral() { Offset = start },
                    _ => new NameExpression(word) { Offset = start },
                };
            }

            throw Malformed(start, $"unexpected character '{c}'");
        }
        finally
        {
            _limits.ExitParserFrame();
        }
    }

    private ExpressionNode ParseNumberLiteral(int start)
    {
        int begin = _pos;

        while (PeekAt(_pos) is >= '0' and <= '9')
            _pos++;

        bool floating = false;

        if (PeekAt(_pos) == '.' && PeekAt(_pos + 1) is >= '0' and <= '9')
        {
            floating = true;
            _pos++;

            while (PeekAt(_pos) is >= '0' and <= '9')
                _pos++;
        }

        if (PeekAny('e', 'E'))
        {
            int mark = _pos;
            _pos++;

            if (PeekAny('+', '-'))
                _pos++;

            if (!IsDigit(PeekAt(_pos)))
            {
                // Not an exponent after all: rewind and let the caller treat e/E as a name boundary error.
                _pos = mark;
            }
            else
            {
                floating = true;

                while (PeekAt(_pos) is >= '0' and <= '9')
                    _pos++;
            }
        }

        string text = _source[begin.._pos];

        if (floating)
            return new FloatLiteral(double.Parse(text, System.Globalization.CultureInfo.InvariantCulture)) { Offset = start };

        return new IntegerLiteral(long.Parse(text, System.Globalization.CultureInfo.InvariantCulture)) { Offset = start };
    }

    /// <summary>
    /// Parses the body of a list literal after the opening bracket, mirroring minja's array grammar: comma-separated
    /// expressions closed by ']', with an empty [] shortcut and no trailing comma.
    /// </summary>
    private ExpressionNode ParseListLiteralBody(int start)
    {
        EnterExpressionFrame();

        try
        {
            List<ExpressionNode> items = [];

            SkipWhitespace();
            if (PeekAt(_pos) == ']')
            {
                _pos++;
                return new ListLiteral(items) { Offset = start };
            }

            while (true)
            {
                _limits.CheckCancellation("list literal", Math.Min(_pos, Math.Max(_source.Length - 1, 0)));
                items.Add(ParseExpression());

                SkipWhitespace();

                char c = PeekAt(_pos);

                if (c == ',')
                {
                    _pos++;

                    SkipWhitespace();
                    if (PeekAt(_pos) == ']')
                        throw Malformed(Math.Min(_pos, Math.Max(_source.Length - 1, 0)), "trailing comma in list literal");

                    continue;
                }

                if (c == ']')
                {
                    _pos++;
                    break;
                }

                throw Malformed(Math.Min(_pos, Math.Max(_source.Length - 1, 0)), "expected ',' or ']' in list literal");
            }

            return new ListLiteral(items) { Offset = start };
        }
        finally
        {
            _limits.ExitParserFrame();
        }
    }

    /// <summary>
    /// Parses the body of a map literal after the opening brace, mirroring minja's dictionary grammar: key ':' value
    /// pairs separated by commas closed by '}', with an empty {} shortcut and no trailing comma. A closing '}' is
    /// never part of a '}}' output-tag close here because that closer only follows a complete expression or literal.
    /// </summary>
    private ExpressionNode ParseMapLiteralBody(int start)
    {
        EnterExpressionFrame();

        try
        {
            List<MapEntry> entries = [];

            SkipWhitespace();
            if (PeekAt(_pos) == '}')
            {
                _pos++;
                return new MapLiteral(entries) { Offset = start };
            }

            while (true)
            {
                _limits.CheckCancellation("map literal", Math.Min(_pos, Math.Max(_source.Length - 1, 0)));

                ExpressionNode key = ParseExpression();

                Expect(':', start);

                entries.Add(new MapEntry(key, ParseExpression()));

                SkipWhitespace();

                char c = PeekAt(_pos);

                if (c == ',')
                {
                    _pos++;

                    SkipWhitespace();
                    if (PeekAt(_pos) == '}')
                        throw Malformed(Math.Min(_pos, Math.Max(_source.Length - 1, 0)), "trailing comma in map literal");

                    continue;
                }

                if (c == '}')
                {
                    _pos++;
                    break;
                }

                throw Malformed(Math.Min(_pos, Math.Max(_source.Length - 1, 0)), "expected ',' or '}' in map literal");
            }

            return new MapLiteral(entries) { Offset = start };
        }
        finally
        {
            _limits.ExitParserFrame();
        }
    }

    private string ParseQuotedString(char quote, int start)
    {
        _pos++; // opening quote
        var value = new System.Text.StringBuilder();

        while (true)
        {
            char c = PeekAt(_pos);

            if (c == '\0')
                throw Malformed(start, "unterminated string literal");

            if (c == quote)
            {
                _pos++;
                return value.ToString();
            }

            if (c == '\\')
            {
                char next = PeekAt(_pos + 1);
                value.Append(next switch
                {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    _ => "\\" + next,
                });

                _pos += 2;
                continue;
            }

            value.Append(c);
            _pos++;
        }
    }

    // ---------------------------------------------------------------- names, targets and argument lists

    private string ReadName(int contextOffset)
    {
        SkipWhitespace();

        if (!IsWordStart(PeekAt(_pos)))
            throw Malformed(Math.Min(_pos, Math.Max(_source.Length - 1, 0)), "expected a name");

        return ReadWord();
    }

    private bool MatchOpenParen()
    {
        SkipWhitespace();

        if (PeekAt(_pos) != '(')
            return false;

        _pos++;
        return true;
    }

    /// <summary>Parses a call or filter argument list after the opening '('. Bare commas are rejected as tuple literals.</summary>
    private List<CallArgument> ParseArguments(int contextOffset)
    {
        EnterExpressionFrame();

        try
        {
            List<CallArgument> arguments = [];

            while (true)
            {
                SkipWhitespace();

                if (PeekAt(_pos) == ')')
                    break;

                _limits.CheckCancellation("call arguments", Math.Min(_pos, Math.Max(_source.Length - 1, 0)));

                string? name = TryNamedArgumentName();
                ExpressionNode value = ParseExpression();
                arguments.Add(new CallArgument(name, value));

                SkipWhitespace();

                if (PeekAt(_pos) != ',')
                    break;

                _pos++;
            }

            Expect(')', contextOffset);
            return arguments;
        }
        finally
        {
            _limits.ExitParserFrame();
        }
    }

    /// <summary>Detects name = value argument syntax by lookahead and consumes the name plus '=' when present.</summary>
    private string? TryNamedArgumentName()
    {
        if (!IsWordStart(PeekAt(_pos)))
            return null;

        int scan = _pos;
        while (scan < _source.Length && IsWordChar(_source[scan]))
            scan++;

        int after = scan;
        while (after < _source.Length && char.IsWhiteSpace(_source[after]))
            after++;

        if (after >= _source.Length || _source[after] != '=' || PeekAt(after + 1) == '=')
            return null;

        string name = _source[_pos..scan];
        _pos = after;
        Expect('=', _pos);

        return name;
    }

    private List<MacroParameter> ParseMacroParameters()
    {
        int listStart = _pos - 1;
        List<MacroParameter> parameters = [];

        while (true)
        {
            SkipWhitespace();

            if (PeekAt(_pos) == ')')
                break;

            int start = _pos;
            string name = ReadName(start);

            ExpressionNode? defaultValue = null;
            if (TakeAssignOperator())
                defaultValue = ParseConditional();

            parameters.Add(new MacroParameter(name, defaultValue));

            SkipWhitespace();

            if (PeekAt(_pos) != ',')
                break;

            _pos++;
        }

        Expect(')', listStart);
        return parameters;
    }

    private TargetPattern ParseForTarget()
    {
        SkipWhitespace();

        int start = _pos;

        if (PeekAt(_pos) == '(')
            return new TupleTarget(ParseTupleItems(start)) { Offset = start };

        return new NameTarget(ReadName(start));
    }

    private List<TargetPattern> ParseTupleItems(int contextOffset)
    {
        EnterExpressionFrame();

        try
        {
            Expect('(', contextOffset);

            List<TargetPattern> items = [];

            while (true)
            {
                SkipWhitespace();

                if (PeekAt(_pos) == '(')
                    items.Add(new TupleTarget(ParseTupleItems(contextOffset)));
                else
                    items.Add(new NameTarget(ReadName(_pos)));

                SkipWhitespace();

                if (PeekAt(_pos) != ',')
                    break;

                _pos++;
            }

            Expect(')', contextOffset);
            return items;
        }
        finally
        {
            _limits.ExitParserFrame();
        }
    }

    /// <summary>Parses set targets after the set keyword: one or more comma-separated names, optionally wrapped in parentheses.</summary>
    private List<SetTarget> ParseSetTargets(int tagStart)
    {
        SkipWhitespace();

        if (PeekAt(_pos) == '(')
            return ParseParenthesizedSetTargets(tagStart);

        List<SetTarget> targets = [ParseDottedTarget(tagStart)];

        while (true)
        {
            SkipWhitespace();

            if (PeekAt(_pos) != ',')
                break;

            _pos++;
            targets.Add(ParseDottedTarget(tagStart));
        }

        return targets;
    }

    private List<SetTarget> ParseParenthesizedSetTargets(int tagStart)
    {
        Expect('(', tagStart);

        List<SetTarget> targets = [ParseDottedTarget(tagStart)];

        while (true)
        {
            SkipWhitespace();

            if (PeekAt(_pos) != ',')
                break;

            _pos++;
            targets.Add(ParseDottedTarget(tagStart));
        }

        Expect(')', tagStart);
        return targets;
    }

    private SetTarget ParseDottedTarget(int contextOffset)
    {
        SkipWhitespace();

        string first = ReadName(contextOffset);
        List<string> parts = [first];

        while (PeekAt(_pos) == '.')
        {
            _pos++;
            parts.Add(ReadName(contextOffset));
        }

        return new SetTarget(first, parts);
    }
}


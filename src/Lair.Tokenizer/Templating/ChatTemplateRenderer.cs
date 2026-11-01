using System.Globalization;
using System.Text;

namespace Lair.Tokenizer.Templating;

/// <summary>
/// Interpreter for a Mini Jinja subset of GGUF chat templates: output tags, for loops, if/elif/else blocks, string and
/// integer literals, map/list construction and index access, concatenation, equality comparisons, not, boolean
/// literals, the tojson and trim filters, and {%- / -%} whitespace control. Anything outside this subset fails at
/// parse time with a <see cref="TemplateSyntaxException"/> naming the construct and its position.
/// </summary>
public sealed class ChatTemplateRenderer
{
    private static readonly char[] WhitespaceChars = [' ', '\t', '\n', '\r'];

    private readonly List<Node> _program;

    private ChatTemplateRenderer(List<Node> program)
    {
        _program = program;
    }

    /// <summary>
    /// Parses a chat template into an executable program. Throws <see cref="TemplateSyntaxException"/> for constructs
    /// outside the supported subset, including their character position in the source.
    /// </summary>
    public static ChatTemplateRenderer Parse(string template)
    {
        ArgumentNullException.ThrowIfNull(template);

        TemplateParser parser = new(template);
        return new ChatTemplateRenderer(parser.ParseProgram());
    }

    /// <summary>
    /// Renders the template for a conversation. The context exposes exactly the variables real chat templates use:
    /// messages (each message map-like via 'role' and 'content') and add_generation_prompt.
    /// </summary>
    public string Render(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt)
    {
        ArgumentNullException.ThrowIfNull(messages);

        List<object?> context = new(messages.Count);
        foreach (ChatMessage message in messages)
        {
            ArgumentNullException.ThrowIfNull(message);
            context.Add(new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["role"] = message.Role,
                ["content"] = message.Content,
            });
        }

        Dictionary<string, object?> variables = new(StringComparer.Ordinal)
        {
            ["messages"] = context,
            ["add_generation_prompt"] = addGenerationPrompt,
        };

        StringBuilder output = new();
        ExecuteNodes(_program, variables, output);
        return output.ToString();
    }

    // ---------------------------------------------------------------- evaluation

    private static void ExecuteNodes(List<Node> nodes, Dictionary<string, object?> scope, StringBuilder output)
    {
        foreach (Node node in nodes)
        {
            switch (node)
            {
                case TextNode text:
                    output.Append(text.Text);
                    break;
                case OutputNode outputNode:
                    output.Append(ToTemplateString(Evaluate(outputNode.Expression, scope), "output"));
                    break;
                case ForNode forNode:
                    ExecuteFor(forNode, scope, output);
                    break;
                case IfNode ifNode:
                    ExecuteIf(ifNode, scope, output);
                    break;
            }
        }
    }

    private static void ExecuteFor(ForNode forNode, Dictionary<string, object?> scope, StringBuilder output)
    {
        object? sequence = Evaluate(forNode.Sequence, scope);
        if (sequence is not List<object?> items)
        {
            throw new TemplateRenderException($"'{{% for %}}' needs a list to iterate over but got {Describe(sequence)}");
        }

        foreach (object? item in items)
        {
            using NestedScope frame = new(scope);
            frame.Variables[forNode.Variable] = item;
            ExecuteNodes(forNode.Body, frame.Variables, output);
        }
    }

    private static void ExecuteIf(IfNode ifNode, Dictionary<string, object?> scope, StringBuilder output)
    {
        foreach (IfBranch branch in ifNode.Branches)
        {
            if (IsTruthy(Evaluate(branch.Condition, scope)))
            {
                ExecuteNodes(branch.Body, scope, output);
                return;
            }
        }

        if (ifNode.ElseBody is not null)
        {
            ExecuteNodes(ifNode.ElseBody, scope, output);
        }
    }

    private static object? Evaluate(Expression expression, Dictionary<string, object?> scope) => expression switch
    {
        LiteralExpression literal => literal.Value,
        NameReference name => ResolveName(name, scope),
        IndexExpression index => EvalIndex(index, scope),
        FilterExpression filter => ApplyFilter(filter, scope),
        ConcatExpression concat => EvalConcat(concat, scope),
        CompareExpression compare => EvalCompare(compare, scope),
        NotExpression not => !IsTruthy(Evaluate(not.Inner, scope)),
        MapLiteral map => EvalMap(map, scope),
        ListLiteral list => EvalList(list, scope),
        _ => throw new TemplateRenderException($"cannot evaluate expression of type {expression.GetType().Name}"),
    };

    private static object? ResolveName(NameReference name, Dictionary<string, object?> scope)
    {
        if (scope.TryGetValue(name.Name, out object? value))
        {
            return value;
        }

        throw new TemplateRenderException($"unknown variable '{name.Name}'");
    }

    private static object? EvalIndex(IndexExpression index, Dictionary<string, object?> scope)
    {
        object? target = Evaluate(index.Target, scope);
        object? key = Evaluate(index.Key, scope);

        if (target is Dictionary<string, object?> map)
        {
            if (key is not string textKey)
            {
                throw new TemplateRenderException($"map index needs a string key but got {Describe(key)}");
            }

            if (!map.TryGetValue(textKey, out object? value))
            {
                throw new TemplateRenderException($"map has no entry '{textKey}'");
            }

            return value;
        }

        if (target is List<object?> list)
        {
            if (key is not long integerKey)
            {
                throw new TemplateRenderException($"list index needs an integer but got {Describe(key)}");
            }

            int offset = checked((int)integerKey);
            if (offset < 0 || offset >= list.Count)
            {
                throw new TemplateRenderException($"index {offset} is out of range for a list of length {list.Count}");
            }

            return list[offset];
        }

        throw new TemplateRenderException($"cannot index into {Describe(target)}");
    }

    private static object? ApplyFilter(FilterExpression filter, Dictionary<string, object?> scope)
    {
        object? value = Evaluate(filter.Target, scope);
        switch (filter.Name)
        {
            case "tojson":
                return ToJson(value);
            case "trim":
                return ToTemplateString(value, "trim").Trim(WhitespaceChars);
            default:
                // Unreachable at runtime; the parser rejects unknown filters.
                throw new TemplateRenderException($"unsupported filter '{filter.Name}'");
        }
    }

    private static string EvalConcat(ConcatExpression concat, Dictionary<string, object?> scope)
    {
        StringBuilder builder = new();
        foreach (Expression part in concat.Parts)
        {
            builder.Append(ToTemplateString(Evaluate(part, scope), "concatenation"));
        }

        return builder.ToString();
    }

    private static bool EvalCompare(CompareExpression compare, Dictionary<string, object?> scope)
    {
        bool equal = ValuesEqual(Evaluate(compare.Left, scope), Evaluate(compare.Right, scope));
        return compare.Operator == "==" ? equal : !equal;
    }

    private static object? EvalMap(MapLiteral literal, Dictionary<string, object?> scope)
    {
        Dictionary<string, object?> map = new(literal.Entries.Count, StringComparer.Ordinal);
        foreach ((Expression keyExpression, Expression valueExpression) in literal.Entries)
        {
            object? key = Evaluate(keyExpression, scope);
            if (key is not string textKey)
            {
                throw new TemplateRenderException($"map keys must be strings but got {Describe(key)}");
            }

            map[textKey] = Evaluate(valueExpression, scope);
        }

        return map;
    }

    private static object? EvalList(ListLiteral literal, Dictionary<string, object?> scope)
    {
        List<object?> items = new(literal.Items.Count);
        foreach (Expression item in literal.Items)
        {
            items.Add(Evaluate(item, scope));
        }

        return items;
    }

    private static bool ValuesEqual(object? left, object? right) => (left, right) switch
    {
        (string a, string b) => string.Equals(a, b, StringComparison.Ordinal),
        (bool a, bool b) => a == b,
        (long a, long b) => a == b,
        (List<object?> a, List<object?> b) => SequencesEqual(a, b),
        (Dictionary<string, object?> a, Dictionary<string, object?> b) => MapsEqual(a, b),
        _ => false,
    };

    private static bool SequencesEqual(List<object?> left, List<object?> right)
    {
        if (left.Count != right.Count)
        {
            return false;
        }

        for (int i = 0; i < left.Count; i++)
        {
            if (!ValuesEqual(left[i], right[i]))
            {
                return false;
            }
        }

        return true;
    }

    private static bool MapsEqual(Dictionary<string, object?> left, Dictionary<string, object?> right)
    {
        if (left.Count != right.Count)
        {
            return false;
        }

        foreach ((string key, object? value) in left)
        {
            if (!right.TryGetValue(key, out object? other) || !ValuesEqual(value, other))
            {
                return false;
            }
        }

        return true;
    }

    private static bool IsTruthy(object? value) => value switch
    {
        null => false,
        bool flag => flag,
        long integer => integer != 0,
        string text => text.Length > 0,
        List<object?> list => list.Count > 0,
        Dictionary<string, object?> map => map.Count > 0,
        _ => true,
    };

    /// <summary>
    /// Jinja stringifies values for output and concatenation: strings verbatim, integers invariant, booleans as
    /// Python's True/False; maps and lists have no textual form and are rejected.
    /// </summary>
    private static string ToTemplateString(object? value, string context) => value switch
    {
        string text => text,
        long integer => integer.ToString(CultureInfo.InvariantCulture),
        bool flag => flag ? "True" : "False",
        _ => throw new TemplateRenderException(
            $"{context} needs text but got {Describe(value)}; apply the tojson filter to render structured values"),
    };

    private static string Describe(object? value) => value switch
    {
        null => "nothing",
        string => "a string",
        long => "an integer",
        bool => "a boolean",
        List<object?> => "a list",
        Dictionary<string, object?> => "a map",
        _ => value.GetType().Name,
    };

    /// <summary>
    /// JSON encoding matching Python json.dumps with ensure_ascii=False: the short escapes (\b \f \n \r \t \\ \" ),
    /// remaining control characters below U+0020 as lowercase-hexadecimal \uXXXX, and every other character emitted
    /// verbatim — including '&lt;', '&gt;', '&amp;', non-ASCII text and supplementary-plane code points, which
    /// System.Text.Json would escape.
    /// </summary>
    private static string ToJson(object? value)
    {
        StringBuilder builder = new();
        WriteJson(builder, value);
        return builder.ToString();
    }

    private static void WriteJson(StringBuilder builder, object? value)
    {
        switch (value)
        {
            case null:
                builder.Append("null");
                break;
            case string text:
                WriteJsonString(builder, text);
                break;
            case long integer:
                builder.Append(integer.ToString(CultureInfo.InvariantCulture));
                break;
            case bool flag:
                builder.Append(flag ? "true" : "false");
                break;
            case List<object?> list:
                builder.Append('[');
                for (int i = 0; i < list.Count; i++)
                {
                    if (i > 0)
                    {
                        builder.Append(',');
                    }

                    WriteJson(builder, list[i]);
                }

                builder.Append(']');
                break;
            case Dictionary<string, object?> map:
                builder.Append('{');
                bool first = true;
                foreach ((string key, object? item) in map)
                {
                    if (!first)
                    {
                        builder.Append(',');
                    }

                    first = false;
                    WriteJsonString(builder, key);
                    builder.Append(':');
                    WriteJson(builder, item);
                }

                builder.Append('}');
                break;
            default:
                throw new TemplateRenderException($"cannot JSON-encode {Describe(value)}");
        }
    }

    private static void WriteJsonString(StringBuilder builder, string text)
    {
        builder.Append('"');
        foreach (char current in text)
        {
            switch (current)
            {
                case '"':
                    builder.Append("\\\"");
                    break;
                case '\\':
                    builder.Append("\\\\");
                    break;
                case '\b':
                    builder.Append("\\b");
                    break;
                case '\f':
                    builder.Append("\\f");
                    break;
                case '\n':
                    builder.Append("\\n");
                    break;
                case '\r':
                    builder.Append("\\r");
                    break;
                case '\t':
                    builder.Append("\\t");
                    break;
                default:
                    if (current < '\u0020')
                    {
                        builder.Append("\\u").Append(((int)current).ToString("x4", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        builder.Append(current);
                    }

                    break;
            }
        }

        builder.Append('"');
    }

    // ---------------------------------------------------------------- runtime state

    private sealed class NestedScope(Dictionary<string, object?> parent) : IDisposable
    {
        public Dictionary<string, object?> Variables { get; } = new(parent, StringComparer.Ordinal);

        public void Dispose()
        {
            Variables.Clear();
        }
    }

    // ---------------------------------------------------------------- AST

    private abstract record Node;

    private sealed record TextNode(string Text) : Node;

    private sealed record OutputNode(Expression Expression) : Node;

    private sealed record ForNode(string Variable, Expression Sequence, List<Node> Body) : Node;

    private sealed record IfNode(List<IfBranch> Branches, List<Node>? ElseBody) : Node;

    private sealed record IfBranch(Expression Condition, List<Node> Body);

    private abstract record Expression(int Position);

    private sealed record LiteralExpression(object? Value, int Position) : Expression(Position);

    private sealed record NameReference(string Name, int Position) : Expression(Position);

    private sealed record IndexExpression(Expression Target, Expression Key, int Position) : Expression(Position);

    private sealed record FilterExpression(Expression Target, string Name, int Position) : Expression(Position);

    private sealed record ConcatExpression(List<Expression> Parts, int Position) : Expression(Position);

    private sealed record CompareExpression(Expression Left, string Operator, Expression Right, int Position)
        : Expression(Position);

    private sealed record NotExpression(Expression Inner, int Position) : Expression(Position);

    private sealed record MapLiteral(List<(Expression Key, Expression Value)> Entries, int Position)
        : Expression(Position);

    private sealed record ListLiteral(List<Expression> Items, int Position) : Expression(Position);

    // ---------------------------------------------------------------- parser

    private sealed class TemplateParser
    {
        private enum TokenKind { Text, Tag, Variable, Name, Number, String, Punct, End }

        private readonly record struct Token(TokenKind Kind, string Text, int Position, bool TrimLeft, bool TrimRight);

        private readonly string _source;
        private readonly List<Token> _tokens = [];
        private int _cursor;
        private int _tagDepth;
        private bool _pendingTrimNext;
        private string? _terminatedBy;

        public TemplateParser(string source)
        {
            _source = source;
            Tokenize();
        }

        public List<Node> ParseProgram()
        {
            List<Node> nodes = ReadNodes([]);
            Token token = Peek();
            if (token.Kind != TokenKind.End)
            {
                Read();
                throw new TemplateSyntaxException($"unexpected token '{token.Text}'", token.Position);
            }

            return nodes;
        }

        // ---------------------------------------------------------------- lexing

        private void Tokenize()
        {
            int position = 0;
            int textStart = 0;

            void FlushText()
            {
                if (position > textStart)
                {
                    _tokens.Add(new Token(TokenKind.Text, _source[textStart..position], textStart, false, false));
                }
            }

            while (position < _source.Length)
            {
                int open = _source.IndexOf("{%", position);
                int variable = _source.IndexOf("{{", position);

                if (open < 0 && variable < 0)
                {
                    break;
                }

                if (open >= 0 && variable > open && variable - open < 2)
                {
                    throw new TemplateSyntaxException("unterminated '{{' output tag", variable);
                }

                bool isTag = open >= 0 && (variable < 0 || open < variable);
                int tagStart = isTag ? open : variable;

                position = tagStart;
                FlushText();

                bool trimLeft = tagStart + 2 < _source.Length && _source[tagStart + 2] == '-';
                int contentStart = trimLeft ? tagStart + 3 : tagStart + 2;

                string close = isTag ? "%}" : "}}";
                int closeIndex = FindTagClose(contentStart, close);
                if (closeIndex < 0)
                {
                    throw new TemplateSyntaxException(
                        isTag ? "unterminated '{%' tag" : "unterminated '{{' output tag", tagStart);
                }

                bool trimRight = closeIndex > contentStart && _source[closeIndex - 1] == '-';
                int contentEnd = trimRight ? closeIndex - 1 : closeIndex;

                TokenKind kind = isTag ? TokenKind.Tag : TokenKind.Variable;
                _tokens.Add(new Token(kind, string.Empty, tagStart, trimLeft, trimRight));
                TokenizeTagContent(contentStart, contentEnd);
                _tokens.Add(new Token(kind, string.Empty, closeIndex, trimLeft, trimRight));

                position = closeIndex + close.Length;
                textStart = position;
            }

            // Trailing text after the last tag: the loop only flushes up to a tag start, so extend to the source end.
            position = _source.Length;
            FlushText();
            _tokens.Add(new Token(TokenKind.End, string.Empty, _source.Length, false, false));
        }

        /// <summary>
        /// Finds the tag's closing delimiter, skipping quoted string literals so a '}}' or '%}' inside a string does
        /// not end the tag early. Returns -1 when the tag contains another tag opening or never closes.
        /// </summary>
        private int FindTagClose(int contentStart, string close)
        {
            char quote = '\0';
            for (int i = contentStart; i < _source.Length; i++)
            {
                char current = _source[i];
                if (quote != '\0')
                {
                    if (current == '\\')
                    {
                        i++;
                    }
                    else if (current == quote)
                    {
                        quote = '\0';
                    }

                    continue;
                }

                if (current is '\'' or '"')
                {
                    quote = current;
                    continue;
                }

                if (current == close[0] && i + close.Length <= _source.Length
                    && _source.AsSpan(i, close.Length).SequenceEqual(close.AsSpan()))
                {
                    return i;
                }

                if (current == '{' && i + 1 < _source.Length && (_source[i + 1] == '{' || _source[i + 1] == '%'))
                {
                    return -1;
                }
            }

            return -1;
        }

        private void TokenizeTagContent(int start, int end)
        {
            int position = start;
            while (position < end)
            {
                char current = _source[position];
                if (char.IsWhiteSpace(current))
                {
                    position++;
                    continue;
                }

                if (char.IsLetter(current) || current == '_')
                {
                    int wordStart = position;
                    while (position < end
                        && (char.IsLetterOrDigit(_source[position]) || _source[position] == '_'))
                    {
                        position++;
                    }

                    _tokens.Add(new Token(TokenKind.Name, _source[wordStart..position], wordStart, false, false));
                    continue;
                }

                if (char.IsAsciiDigit(current))
                {
                    int numberStart = position;
                    while (position < end && char.IsAsciiDigit(_source[position]))
                    {
                        position++;
                    }

                    _tokens.Add(new Token(TokenKind.Number, _source[numberStart..position], numberStart, false, false));
                    continue;
                }

                if (current is '\'' or '"')
                {
                    position = TokenizeString(current, position, end);
                    continue;
                }

                if (current is '=' or '!' && position + 1 < end && _source[position + 1] == '=')
                {
                    _tokens.Add(new Token(TokenKind.Punct, current + "=", position, false, false));
                    position += 2;
                    continue;
                }

                if (position + 1 < end && _source[position + 1] == current && current is ':' or '.')
                {
                    throw new TemplateSyntaxException($"unsupported construct '{current}{current}'", position);
                }

                _tokens.Add(new Token(TokenKind.Punct, current.ToString(), position, false, false));
                position++;
            }
        }

        private int TokenizeString(char quote, int start, int end)
        {
            StringBuilder builder = new();
            int position = start + 1;
            while (true)
            {
                if (position >= end)
                {
                    throw new TemplateSyntaxException("unterminated string literal", start);
                }

                char current = _source[position];
                if (current == quote)
                {
                    position++;
                    break;
                }

                if (current == '\\')
                {
                    if (position + 1 >= end)
                    {
                        throw new TemplateSyntaxException("unterminated string literal", start);
                    }

                    char escaped = _source[position + 1];
                    builder.Append(escaped switch
                    {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '\\' => '\\',
                        '\'' => '\'',
                        '"' => '"',
                        _ => throw new TemplateSyntaxException($"unsupported escape '\\{escaped}'", position),
                    });
                    position += 2;
                    continue;
                }

                builder.Append(current);
                position++;
            }

            _tokens.Add(new Token(TokenKind.String, builder.ToString(), start, false, false));
            return position;
        }

        // ---------------------------------------------------------------- token access

        private Token Peek() => _tokens[_cursor];

        private Token Read() => _tokens[_cursor++];

        private void ExpectTagClose(TokenKind expectedKind)
        {
            Token token = Peek();

            // A block body's own nested tags are still open until their closing tags are consumed; only a tag at this
            // block's own nesting depth may close it. Otherwise a nested block would swallow the outer close marker and
            // the real failure (an unclosed inner block) would surface as a confusing error far from its source.
            if (token.Kind != expectedKind || _tagDepth != 1)
            {
                throw new TemplateSyntaxException("expected a tag close", token.Position);
            }

            Read();
            _tagDepth--;
            _pendingTrimNext = token.TrimRight;
        }

        // ---------------------------------------------------------------- node parsing

        private List<Node> ReadNodes(string[] stopKeywords)
        {
            List<Node> nodes = [];

            while (true)
            {
                Token token = Peek();
                switch (token.Kind)
                {
                    case TokenKind.End:
                        _terminatedBy = null;
                        return nodes;

                    case TokenKind.Text:
                        Read();
                        string text = token.Text;
                        if (_pendingTrimNext)
                        {
                            _pendingTrimNext = false;
                            text = text.TrimStart(WhitespaceChars);
                        }

                        if (text.Length > 0)
                        {
                            nodes.Add(new TextNode(text));
                        }

                        break;

                    case TokenKind.Variable:
                        if (token.TrimLeft)
                        {
                            TrimTail(nodes);
                        }

                        Read();
                        _tagDepth++;
                        Expression expression = ParseExpression();
                        ExpectTagClose(TokenKind.Variable);
                        nodes.Add(new OutputNode(expression));
                        break;

                    case TokenKind.Tag:
                        if (token.TrimLeft)
                        {
                            TrimTail(nodes);
                        }

                        Read();
                        _tagDepth++;
                        _pendingTrimNext = token.TrimRight;
                        if (Peek().Kind != TokenKind.Name)
                        {
                            Token offender = Peek();
                            throw new TemplateSyntaxException(
                                $"unsupported statement tag contents '{offender.Text}'", offender.Position);
                        }

                        Token keyword = Read();
                        switch (keyword.Text)
                        {
                            case "for":
                                nodes.Add(ParseFor(keyword.Position));
                                break;
                            case "if":
                                nodes.Add(ParseIf(keyword.Position));
                                break;
                            default:
                                if (Array.IndexOf(stopKeywords, keyword.Text) >= 0)
                                {
                                    _terminatedBy = keyword.Text;
                                    return nodes;
                                }

                                if (keyword.Text is "elif" or "else" or "endfor" or "endif")
                                {
                                    throw new TemplateSyntaxException(
                                        $"unexpected '{{% {keyword.Text} %}}'", keyword.Position);
                                }

                                throw new TemplateSyntaxException(
                                    $"unsupported statement '{{% {keyword.Text} %}}'", keyword.Position);
                        }

                        break;

                    default:
                        throw new TemplateSyntaxException($"unexpected token '{token.Text}'", token.Position);
                }
            }
        }

        private ForNode ParseFor(int keywordPosition)
        {
            Token variable = Peek();
            if (variable.Kind != TokenKind.Name || variable.Text == "in")
            {
                throw new TemplateSyntaxException(
                    $"'{{% for %}}' needs a loop variable but found '{variable.Text}'", variable.Position);
            }

            Read();
            Token inToken = Peek();
            if (inToken.Kind != TokenKind.Name || inToken.Text != "in")
            {
                throw new TemplateSyntaxException(
                    $"'{{% for %}}' needs 'in' but found '{inToken.Text}'", inToken.Position);
            }

            Read();
            bool keywordTrimRight = _pendingTrimNext;
            Expression sequence = ParseExpression();
            ExpectTagClose(TokenKind.Tag);

            // The keyword's own '-%}' trim applies to the whitespace between the tag and the loop body, not to the
            // sequence expression; restore it after the expression consumed the tag close.
            _pendingTrimNext = keywordTrimRight;
            List<Node> body = ReadNodes(["endfor"]);
            if (_terminatedBy != "endfor")
            {
                throw new TemplateSyntaxException("unclosed '{% for %}'", keywordPosition);
            }

            ExpectTagClose(TokenKind.Tag);
            return new ForNode(variable.Text, sequence, body);
        }

        private IfNode ParseIf(int keywordPosition)
        {
            List<IfBranch> branches = [];
            bool keywordTrimRight = _pendingTrimNext;
            Expression condition = ParseExpression();
            ExpectTagClose(TokenKind.Tag);

            // The keyword's own '-%}' trim applies to the whitespace between the tag and the branch body, not to the
            // condition expression; restore it after the expression consumed the tag close.
            _pendingTrimNext = keywordTrimRight;
            List<Node> body = ReadNodes(["elif", "else", "endif"]);
            branches.Add(new IfBranch(condition, body));

            while (true)
            {
                switch (_terminatedBy)
                {
                    case "elif":
                        branches.Add(new IfBranch(ParseExpression(), ReadIfBody()));
                        continue;
                    case "else":
                        ExpectTagClose(TokenKind.Tag);
                        List<Node> elseBody = ReadNodes(["endif"]);
                        if (_terminatedBy != "endif")
                        {
                            throw new TemplateSyntaxException("unclosed '{% if %}'", keywordPosition);
                        }

                        ExpectTagClose(TokenKind.Tag);
                        return new IfNode(branches, elseBody);
                    case "endif":
                        ExpectTagClose(TokenKind.Tag);
                        return new IfNode(branches, null);
                    default:
                        throw new TemplateSyntaxException("unclosed '{% if %}'", keywordPosition);
                }
            }
        }

        private List<Node> ReadIfBody()
        {
            ExpectTagClose(TokenKind.Tag);
            return ReadNodes(["elif", "else", "endif"]);
        }

        private static void TrimTail(List<Node> nodes)
        {
            while (nodes.Count > 0 && nodes[^1] is TextNode text)
            {
                string trimmed = text.Text.TrimEnd(WhitespaceChars);
                if (trimmed.Length == 0)
                {
                    nodes.RemoveAt(nodes.Count - 1);
                    continue;
                }

                nodes[^1] = new TextNode(trimmed);
                break;
            }
        }

        // ---------------------------------------------------------------- expression parsing

        private Expression ParseExpression() => ParseComparison();

        private Expression ParseComparison()
        {
            Expression left = ParseConcat();
            while (true)
            {
                Token token = Peek();
                if (token.Kind == TokenKind.Name && token.Text is "and" or "or")
                {
                    throw new TemplateSyntaxException($"unsupported operator '{token.Text}'", token.Position);
                }

                if (token.Kind != TokenKind.Punct || token.Text is not ("==" or "!="))
                {
                    break;
                }

                Read();
                Expression right = ParseConcat();
                left = new CompareExpression(left, token.Text, right, token.Position);
            }

            return left;
        }

        private Expression ParseConcat()
        {
            List<Expression> parts = [ParseFilterChain()];
            while (Peek().Kind == TokenKind.Punct && Peek().Text == "+")
            {
                Read();
                parts.Add(ParseFilterChain());
            }

            return parts.Count == 1 ? parts[0] : new ConcatExpression(parts, parts[0].Position);
        }

        private Expression ParseFilterChain()
        {
            Expression expression = ParseUnary();
            while (Peek().Kind == TokenKind.Punct && Peek().Text == "|")
            {
                Token pipe = Read();
                Token name = Peek();
                if (name.Kind != TokenKind.Name)
                {
                    throw new TemplateSyntaxException("expected a filter name after '|'", name.Position);
                }

                Read();
                if (name.Text is not ("tojson" or "trim"))
                {
                    throw new TemplateSyntaxException($"unsupported filter '{name.Text}'", name.Position);
                }

                if (Peek().Kind == TokenKind.Punct && Peek().Text == "(")
                {
                    throw new TemplateSyntaxException($"filter '{name.Text}' does not support arguments", Peek().Position);
                }

                expression = new FilterExpression(expression, name.Text, pipe.Position);
            }

            return expression;
        }

        private Expression ParseUnary()
        {
            Token token = Peek();
            if (token.Kind == TokenKind.Name && token.Text == "not")
            {
                Read();
                return new NotExpression(ParseUnary(), token.Position);
            }

            return ParsePostfix();
        }

        private Expression ParsePostfix()
        {
            Expression expression = ParsePrimary();
            while (true)
            {
                Token token = Peek();
                if (token.Kind == TokenKind.Punct && token.Text == "[")
                {
                    Read();
                    Expression key = ParseExpression();
                    Token close = Peek();
                    if (close.Kind == TokenKind.Punct && close.Text == ":")
                    {
                        throw new TemplateSyntaxException("unsupported construct 'slice'", close.Position);
                    }

                    if (close.Kind != TokenKind.Punct || close.Text != "]")
                    {
                        throw new TemplateSyntaxException("expected ']'", close.Position);
                    }

                    Read();
                    expression = new IndexExpression(expression, key, token.Position);
                    continue;
                }

                if (token.Kind == TokenKind.Punct && token.Text == "(")
                {
                    throw new TemplateSyntaxException("unsupported construct 'function call'", token.Position);
                }

                break;
            }

            return expression;
        }

        private Expression ParsePrimary()
        {
            Token token = Peek();
            switch (token.Kind)
            {
                case TokenKind.String:
                    Read();
                    return new LiteralExpression(token.Text, token.Position);

                case TokenKind.Number:
                    Read();
                    return new LiteralExpression(long.Parse(token.Text, CultureInfo.InvariantCulture), token.Position);

                case TokenKind.Name:
                    switch (token.Text)
                    {
                        case "true" or "True":
                            Read();
                            return new LiteralExpression(true, token.Position);
                        case "false" or "False":
                            Read();
                            return new LiteralExpression(false, token.Position);
                        case "and" or "or":
                            throw new TemplateSyntaxException($"unsupported operator '{token.Text}'", token.Position);
                        default:
                            Read();
                            return new NameReference(token.Text, token.Position);
                    }

                case TokenKind.Punct when token.Text == "{":
                    return ParseMap();

                case TokenKind.Punct when token.Text == "[":
                    return ParseListLiteral();

                case TokenKind.Punct when token.Text == "(":
                    Read();
                    Expression inner = ParseExpression();
                    Token close = Peek();
                    if (close.Kind != TokenKind.Punct || close.Text != ")")
                    {
                        throw new TemplateSyntaxException("expected ')'", close.Position);
                    }

                    Read();
                    return inner;

                default:
                    throw new TemplateSyntaxException($"unexpected token '{token.Text}'", token.Position);
            }
        }

        private Expression ParseMap()
        {
            int position = Read().Position;
            List<(Expression Key, Expression Value)> entries = [];
            while (!(Peek().Kind == TokenKind.Punct && Peek().Text == "}"))
            {
                if (Peek().Kind == TokenKind.End)
                {
                    throw new TemplateSyntaxException("unclosed map literal", position);
                }

                Expression key = ParseFilterChain();
                Token colon = Peek();
                if (colon.Kind != TokenKind.Punct || colon.Text != ":")
                {
                    throw new TemplateSyntaxException("expected ':' in map literal", colon.Position);
                }

                Read();
                entries.Add((key, ParseExpression()));
                if (Peek().Kind == TokenKind.Punct && Peek().Text == ",")
                {
                    Read();
                }
            }

            Read();
            return new MapLiteral(entries, position);
        }

        private Expression ParseListLiteral()
        {
            int position = Read().Position;
            List<Expression> items = [];
            while (!(Peek().Kind == TokenKind.Punct && Peek().Text == "]"))
            {
                if (Peek().Kind == TokenKind.End)
                {
                    throw new TemplateSyntaxException("unclosed list literal", position);
                }

                items.Add(ParseExpression());
                if (Peek().Kind == TokenKind.Punct && Peek().Text == ",")
                {
                    Read();
                }
            }

            Read();
            return new ListLiteral(items, position);
        }
    }
}

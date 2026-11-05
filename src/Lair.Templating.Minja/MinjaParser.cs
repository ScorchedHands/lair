namespace Lair.Templating.Minja;

/// <summary>
/// Parses a minja-compatible GGUF chat template into an immutable AST in two phases: the structure phase splits the
/// template into text runs, comments, output tags and statement blocks; the expression phase parses tag contents with
/// minja's expression grammar. Whitespace elision from {%- -%} and {#- -#} is applied to TextNode content at parse time
/// so a later renderer can emit byte-exact output without re-interpreting trim markers: TrimLeft/TrimRight on tagged
/// nodes record which marker requested the elision, while text runs already hold the exact bytes to emit. A run of only
/// whitespace between tags is kept unless a neighbouring marker elides it, so ordinary spacing renders unchanged; a run
/// trimmed to nothing is dropped from its body. Trim-left of an end/elif/else keyword is applied by the block parser that
/// owns the preceding body because Jinja's whitespace control spans only directly adjacent literal text: an elision flag
/// never crosses a statement boundary into another body list, and no literal text can sit between two adjacent tags.
/// Comments never appear as literal output. Inline list ([a, b]) and map/object ({k: v}) literals parse in every
/// expression context. Unsupported constructs ({% raw %}, {% block %}, {% include %}, {% extends %} and their trivial
/// typo variants, parenthesized tuple literals outside call arguments or destructuring targets, ternary-less if
/// expressions) and malformed tags throw <see cref="MinjaSyntaxException"/> naming the construct with its 0-based
/// template offset. Parsing only: evaluation is a separate concern.
/// </summary>
public sealed partial class MinjaParser
{
    private readonly string _source;

    // Single cursor for both phases: structure parsing advances it between tags, expression parsing inside tag contents.
    private int _pos;

    /// <summary>Depth of open macro/call/filter blocks; break and continue are only valid inside one.</summary>
    private int _blockDepth;

    private MinjaParser(string source) => _source = source;

    /// <summary>Parses a chat template into its AST. Throws <see cref="MinjaSyntaxException"/> for unsupported or malformed syntax.</summary>
    public static TemplateNode Parse(string template)
    {
        ArgumentNullException.ThrowIfNull(template);
        return new MinjaParser(template).ParseTemplate();
    }

    // ---------------------------------------------------------------- structure phase

    private TemplateNode ParseTemplate() => new(ParseNodes(null)) { Offset = 0, Length = _source.Length };

    /// <summary>
    /// Parses nodes until the end of the template or, inside a block body, until one of the terminator tag keywords is
    /// seen. The terminator tag itself is left unconsumed for the statement parser that owns it.
    /// </summary>
    private List<MinjaNode> ParseNodes(IReadOnlyList<string>? terminators)
    {
        List<MinjaNode> nodes = [];

        while (_pos < _source.Length)
        {
            int tagStart = NextTagStart();
            if (tagStart < 0)
            {
                AddText(nodes, _pos, _source.Length);
                _pos = _source.Length;
                break;
            }

            char marker = _source[tagStart + 1];

            if (marker == '%' && PeekTagKeyword(tagStart) is { } keyword && terminators is not null && ListContains(terminators, keyword))
            {
                // Text before a terminator belongs to the body being closed.
                AddText(nodes, _pos, tagStart);
                _pos = tagStart;
                break;
            }

            AddText(nodes, _pos, tagStart);

            // Opening-marker elision ({%- / {{- ) applies to the text run just added in this same body.
            if (OpenTrimLeft(tagStart))
                TrimTrailing(nodes);

            if (marker == '%')
            {
                nodes.Add(ParseStatementTag(tagStart, PeekTagKeyword(tagStart)));
            }
            else if (marker == '{')
            {
                nodes.Add(ParseOutputTag(tagStart));
            }
            else
            {
                nodes.Add(ParseCommentTag(tagStart));
            }
        }

        ApplyTrimRight(nodes);
        return nodes;
    }

    /// <summary>Parses a block body with break/continue allowed at this nesting depth.</summary>
    private List<MinjaNode> ParseBlockBody(IReadOnlyList<string> terminators)
    {
        _blockDepth++;
        try
        {
            return ParseNodes(terminators);
        }
        finally
        {
            _blockDepth--;
        }
    }

    private int NextTagStart()
    {
        for (int i = _pos; i < _source.Length; i++)
        {
            if (_source[i] != '{')
                continue;

            char next = CharAt(i + 1);
            if (next == '%' || next == '{' || next == '#')
                return i;
        }

        return -1;
    }

    private static bool ListContains(IReadOnlyList<string> list, string value)
    {
        for (int i = 0; i < list.Count; i++)
        {
            if (list[i] == value)
                return true;
        }

        return false;
    }

    /// <summary>Reads the keyword immediately after {% without consuming it.</summary>
    private string PeekTagKeyword(int tagStart)
    {
        int pos = TagBodyStart(tagStart);

        while (pos < _source.Length && char.IsWhiteSpace(_source[pos]))
            pos++;

        int start = pos;
        while (pos < _source.Length && IsWordChar(_source[pos]))
            pos++;

        return pos > start ? _source[start..pos] : throw Malformed(tagStart, "statement tag has no keyword");
    }

    private OutputNode ParseOutputTag(int tagStart)
    {
        _pos = tagStart + 2;
        bool trimLeft = TakeDash();

        SkipWhitespace();
        ExpressionNode expression = ParseExpression();
        CloseTag(tagStart, "}}", out bool trimRight);

        return new OutputNode(expression)
        {
            Offset = tagStart,
            Length = _pos - tagStart,
            TrimLeft = trimLeft,
            TrimRight = trimRight,
        };
    }

    private CommentNode ParseCommentTag(int tagStart)
    {
        bool trimLeft = CharAt(tagStart + 2) == '-';
        int closeHash = FindCommentClose(trimLeft ? tagStart + 3 : tagStart + 2);
        bool trimRight = _source[closeHash - 1] == '-';

        return new CommentNode
        {
            Offset = tagStart,
            Length = closeHash + 2 - tagStart,
            TrimLeft = trimLeft,
            TrimRight = trimRight,
        };
    }

    /// <summary>Finds the index of the closing #} of a comment; comment bodies are opaque text.</summary>
    private int FindCommentClose(int searchFrom)
    {
        for (int i = Math.Max(searchFrom, 0); i + 1 < _source.Length; i++)
        {
            if (_source[i] == '#' && CharAt(i + 1) == '}')
            {
                _pos = i + 2;
                return i;
            }
        }

        throw Malformed(Math.Min(searchFrom, Math.Max(_source.Length - 1, 0)), "unclosed comment: expected '#}'");
    }

    private MinjaNode ParseStatementTag(int tagStart, string name)
    {
        switch (name)
        {
            case "if":
                return ParseIf(tagStart);

            case "for":
                return ParseFor(tagStart);

            case "set":
                return ParseSet(tagStart);

            case "macro":
                return ParseMacro(tagStart);

            case "call":
                return ParseCall(tagStart);

            case "filter":
                return ParseFilter(tagStart);

            // break/continue are only valid inside an open macro/call/filter body.
            case "break" when _blockDepth > 0:
            {
                bool trimRight = ConsumeKeywordOnlyTag(tagStart, name);
                return new BreakStatement { Offset = tagStart, Length = _pos - tagStart, TrimLeft = OpenTrimLeft(tagStart), TrimRight = trimRight };
            }

            case "continue" when _blockDepth > 0:
            {
                bool trimRight = ConsumeKeywordOnlyTag(tagStart, name);
                return new ContinueStatement { Offset = tagStart, Length = _pos - tagStart, TrimLeft = OpenTrimLeft(tagStart), TrimRight = trimRight };
            }

            case "break":
            case "continue":
                throw Malformed(tagStart, $"'{name}' is only valid inside macro/call/filter");

            // minja does not support these Jinja constructs; the tag name and offset make the rejection actionable.
            case "raw":
            case "block":
            case "include":
            case "extends":
            case "import":
            case "from":
            case "with":
            case "autoescape":
            case "trans":
                throw new MinjaSyntaxException(name, tagStart);

            // trivial typo variants of rejected constructs get a corrected name in the error.
            case "rwa":
            case "blcok":
            case "incldue":
            case "extnds":
                throw new MinjaSyntaxException(TypoName(name), tagStart);

            default:
                if (name.StartsWith("end", StringComparison.Ordinal))
                    throw Malformed(tagStart, $"'{name}' has no matching block");

                throw new MinjaSyntaxException(name, tagStart);
        }
    }

    /// <summary>Consumes a statement tag whose only content is its keyword and returns its -%} flag.</summary>
    private bool ConsumeKeywordOnlyTag(int tagStart, string keyword)
    {
        ValidateBlockKeyword(tagStart, keyword, allowContent: false);

        _pos = TagBodyStart(tagStart);
        SkipWhitespace();
        ReadWord();

        CloseTag(tagStart, "%}", out bool trimRight);
        return trimRight;
    }

    private static string TypoName(string name) => name switch
    {
        "rwa" => "raw",
        "blcok" => "block",
        "incldue" => "include",
        "extnds" => "extends",
        _ => name,
    };

    private IfStatement ParseIf(int tagStart)
    {
        bool trimLeft = OpenTrimLeft(tagStart);
        _pos = TagBodyStart(tagStart);

        SkipWhitespace();
        ReadWord();

        ExpressionNode condition = ParseExpression();
        CloseTag(tagStart, "%}", out bool trimRight);

        List<MinjaNode> body = ParseNodes(["elif", "else", "endif"]);
        if (trimRight)
            TrimLeadingFirst(body);

        List<IfBranch> branches = [new IfBranch(condition, body)];
        List<MinjaNode>? elseBody = null;

        while (true)
        {
            RequireBlockEnd("if");

            int tag = _pos;
            string name = PeekTagKeyword(tag);

            if (name == "elif")
            {
                ExpressionNode elifCondition = ParseBranchTag(tag, body, out bool branchTrimRight);

                body = ParseNodes(["elif", "else", "endif"]);
                if (branchTrimRight)
                    TrimLeadingFirst(body);

                branches.Add(new IfBranch(elifCondition, body));
                continue;
            }

            if (name == "else")
            {
                elseBody = ConsumeMidTag(tag, "else", ["endif"], body, out bool midTrimRight);

                // -%} on a mid tag elides the body it introduces, not the statement's following sibling.
                if (midTrimRight)
                    TrimLeadingFirst(elseBody);

                break;
            }

            break;
        }

        int endTag = _pos;
        trimRight |= ConsumeBlockTag(endTag, "endif", elseBody ?? body);

        return new IfStatement(branches, elseBody)
        {
            Offset = tagStart,
            Length = _pos - tagStart,
            TrimLeft = trimLeft,
            TrimRight = trimRight,
        };
    }

    /// <summary>Parses the condition of an elif branch and consumes its opening tag.</summary>
    private ExpressionNode ParseBranchTag(int tagStart, List<MinjaNode> precedingBody, out bool trimRight)
    {
        ValidateBlockKeyword(tagStart, "elif", allowContent: true);
        TrimBeforeTag(tagStart, precedingBody);

        _pos = TagBodyStart(tagStart);
        SkipWhitespace();
        ReadWord();

        ExpressionNode condition = ParseExpression();
        CloseTag(tagStart, "%}", out trimRight);

        return condition;
    }

    private ForStatement ParseFor(int tagStart)
    {
        bool trimLeft = OpenTrimLeft(tagStart);
        _pos = TagBodyStart(tagStart);

        SkipWhitespace();
        ReadWord();

        TargetPattern target = ParseForTarget();

        if (!MatchKeyword("in"))
            throw new MinjaSyntaxException("missing 'in' in for loop", tagStart);

        // The iterable stops before trailing 'if'/'recursive' keywords, so it parses at the or level.
        ExpressionNode iterable = ParseOrLevel();

        bool recursive = MatchKeyword("recursive");

        SkipWhitespace();
        ExpressionNode? loopCondition = MatchKeyword("if") ? ParseConditional() : null;

        CloseTag(tagStart, "%}", out bool trimRight);

        List<MinjaNode> body = ParseBlockBody(["else", "endfor"]);
        if (trimRight)
            TrimLeadingFirst(body);

        RequireBlockEnd("for");

        int endTag = _pos;
        if (PeekTagKeyword(endTag) == "else")
            throw new MinjaSyntaxException("loop-else is not supported by minja", endTag);

        trimRight |= ConsumeBlockTag(endTag, "endfor", body);

        return new ForStatement(target, iterable, body, recursive, loopCondition)
        {
            Offset = tagStart,
            Length = _pos - tagStart,
            TrimLeft = trimLeft,
            TrimRight = trimRight,
        };
    }

    private SetStatement ParseSet(int tagStart)
    {
        bool trimLeft = OpenTrimLeft(tagStart);
        _pos = TagBodyStart(tagStart);

        SkipWhitespace();
        ReadWord();

        List<SetTarget> targets = ParseSetTargets(tagStart);

        if (!TakeAssignOperator())
            throw new MinjaSyntaxException("missing '=' in set statement", tagStart);

        ExpressionNode value = ParseExpression();
        CloseTag(tagStart, "%}", out bool trimRight);

        return new SetStatement(targets, value)
        {
            Offset = tagStart,
            Length = _pos - tagStart,
            TrimLeft = trimLeft,
            TrimRight = trimRight,
        };
    }

    private MacroStatement ParseMacro(int tagStart)
    {
        bool trimLeft = OpenTrimLeft(tagStart);
        _pos = TagBodyStart(tagStart);

        SkipWhitespace();
        ReadWord();

        string name = ReadName(tagStart);
        List<MacroParameter> parameters = MatchOpenParen() ? ParseMacroParameters() : [];
        CloseTag(tagStart, "%}", out bool trimRight);

        List<MinjaNode> body = ParseBlockBody(["endmacro"]);
        if (trimRight)
            TrimLeadingFirst(body);

        RequireBlockEnd("macro");
        trimRight |= ConsumeBlockTag(_pos, "endmacro", body);

        return new MacroStatement(name, parameters, body)
        {
            Offset = tagStart,
            Length = _pos - tagStart,
            TrimLeft = trimLeft,
            TrimRight = trimRight,
        };
    }

    private CallStatement ParseCall(int tagStart)
    {
        bool trimLeft = OpenTrimLeft(tagStart);
        _pos = TagBodyStart(tagStart);

        SkipWhitespace();
        ReadWord();

        ExpressionNode target = ParsePostfixChain(ParsePrimary());

        if (target is not CallExpression call)
            throw new MinjaSyntaxException("call requires a called expression such as {% call macro(args) %}", tagStart);

        CloseTag(tagStart, "%}", out bool trimRight);

        List<MinjaNode> body = ParseBlockBody(["endcall"]);
        if (trimRight)
            TrimLeadingFirst(body);

        RequireBlockEnd("call");
        trimRight |= ConsumeBlockTag(_pos, "endcall", body);

        return new CallStatement(call.Callee, call.Arguments, body)
        {
            Offset = tagStart,
            Length = _pos - tagStart,
            TrimLeft = trimLeft,
            TrimRight = trimRight,
        };
    }

    private FilterStatement ParseFilter(int tagStart)
    {
        bool trimLeft = OpenTrimLeft(tagStart);
        _pos = TagBodyStart(tagStart);

        SkipWhitespace();
        ReadWord();

        string name = ReadName(tagStart);
        List<CallArgument> arguments = MatchOpenParen() ? ParseArguments(tagStart) : [];
        CloseTag(tagStart, "%}", out bool trimRight);

        List<MinjaNode> body = ParseBlockBody(["else", "endfilter"]);
        if (trimRight)
            TrimLeadingFirst(body);

        List<MinjaNode>? elseBody = null;

        RequireBlockEnd("filter");

        int tag = _pos;
        if (PeekTagKeyword(tag) == "else")
        {
            elseBody = ConsumeMidTag(tag, "else", ["endfilter"], body, out bool midTrimRight);

            // -%} on a mid tag elides the body it introduces, not the statement's following sibling.
            if (midTrimRight)
                TrimLeadingFirst(elseBody);

            RequireBlockEnd("filter");
            tag = _pos;
        }

        trimRight |= ConsumeBlockTag(tag, "endfilter", elseBody ?? body);

        return new FilterStatement(name, arguments, body, elseBody)
        {
            Offset = tagStart,
            Length = _pos - tagStart,
            TrimLeft = trimLeft,
            TrimRight = trimRight,
        };
    }
}

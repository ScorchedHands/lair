namespace Lair.Templating.Minja;

public sealed partial class MinjaParser
{
    // ---------------------------------------------------------------- cursor and tag helpers shared by both phases

    private static bool IsWordChar(char c) => char.IsLetterOrDigit(c) || c == '_';

    private static bool IsWordStart(char c) => char.IsLetter(c) || c == '_';

    private static bool IsDigit(char c) => char.IsBetween(c, '0', '9');

    /// <summary>Character at an absolute source position; '\0' signals end of template.</summary>
    private char CharAt(int pos) => (uint)pos < (uint)_source.Length ? _source[pos] : '\0';

    private char PeekAt(int pos) => CharAt(pos);

    /// <summary>The index just past the first whole word at or after pos, skipping leading whitespace.</summary>
    private int WordEnd(int pos)
    {
        while (pos < _source.Length && char.IsWhiteSpace(_source[pos]))
            pos++;

        while (IsWordChar(CharAt(pos)))
            pos++;

        return pos;
    }

    /// <summary>True when the whole word at or after pos equals keyword.</summary>
    private bool KeywordAtEquals(int pos, string keyword)
    {
        while (char.IsWhiteSpace(CharAt(pos)))
            pos++;

        for (int i = 0; i < keyword.Length; i++)
        {
            if (CharAt(pos + i) != keyword[i])
                return false;
        }

        // The match must end at a word boundary: 'if' inside 'ifdef' is not the if keyword.
        return !IsWordChar(CharAt(pos + keyword.Length));
    }

    private void SkipWhitespace()
    {
        while (_pos < _source.Length && char.IsWhiteSpace(_source[_pos]))
            _pos++;
    }

    /// <summary>Consumes a '-' at the cursor when present (trim markers).</summary>
    private bool TakeDash()
    {
        if (PeekAt(_pos) != '-')
            return false;

        _pos++;
        return true;
    }

    private string ReadWord()
    {
        int start = _pos;

        while (_pos < _source.Length && IsWordChar(_source[_pos]))
            _pos++;

        return _source[start.._pos];
    }

    /// <summary>The whole word at or after the cursor without consuming it.</summary>
    private string PeekKeyword()
    {
        int pos = _pos;

        while (char.IsWhiteSpace(CharAt(pos)))
            pos++;

        int start = pos;

        while (IsWordChar(CharAt(pos)))
            pos++;

        return _source[start..pos];
    }

    /// <summary>Consumes the keyword when it is the next whole word.</summary>
    private bool MatchKeyword(string keyword)
    {
        SkipWhitespace();

        if (!KeywordAtEquals(_pos, keyword))
            return false;

        // Consume exactly this word: WordEnd would skip whitespace into the following token.
        while (IsWordChar(CharAt(_pos)))
            _pos++;

        return true;
    }

    /// <summary>Consumes a single '=' that is not the start of '=='; used for assignments and macro defaults.</summary>
    private bool TakeAssignOperator()
    {
        SkipWhitespace();

        if (PeekAt(_pos) != '=' || PeekAt(_pos + 1) == '=')
            return false;

        _pos++;
        return true;
    }

    // ---------------------------------------------------------------- statement tag plumbing

    private bool OpenTrimLeft(int tagStart) => CharAt(tagStart + 2) == '-';

    /// <summary>Index just past {% and an optional {%- dash.</summary>
    private int TagBodyStart(int tagStart) => tagStart + (CharAt(tagStart + 2) == '-' ? 3 : 2);

    /// <summary>True when a statement tag holds anything besides whitespace after its keyword.</summary>
    private bool HasExtraContent(int tagStart)
    {
        int pos = TagBodyStart(tagStart);

        while (char.IsWhiteSpace(CharAt(pos)))
            pos++;

        while (IsWordChar(CharAt(pos)))
            pos++;

        while (char.IsWhiteSpace(CharAt(pos)))
            pos++;

        if (CharAt(pos) == '-')
            pos++;

        return !(CharAt(pos) == '%' && CharAt(pos + 1) == '}');
    }

    private void ValidateBlockKeyword(int tagStart, string expected, bool allowContent)
    {
        string actual = PeekTagKeyword(tagStart);

        if (actual != expected || (!allowContent && HasExtraContent(tagStart)))
            throw Malformed(tagStart, $"expected '{expected}'");
    }

    private void RequireBlockEnd(string blockKeyword)
    {
        if (_pos < _source.Length)
            return;

        string close = CloseKeyword(blockKeyword);
        throw Malformed(Math.Max(_source.Length - 1, 0), $"'{blockKeyword}' block is not closed (expected '{close}')");
    }

    private static string CloseKeyword(string blockKeyword) => blockKeyword switch
    {
        "if" => "endif",
        "for" => "endfor",
        "macro" => "endmacro",
        "call" => "endcall",
        "filter" => "endfilter",
        _ => $"end{blockKeyword}",
    };

    /// <summary>Applies {%- of a mid/end tag to the text run that precedes it in the owning body.</summary>
    private void TrimBeforeTag(int tagStart, List<MinjaNode> precedingBody)
    {
        if (OpenTrimLeft(tagStart))
            TrimTrailing(precedingBody);
    }

    /// <summary>Parses a mid-block keyword such as else and returns the node list it introduces.</summary>
    private List<MinjaNode> ConsumeMidTag(int tagStart, string keyword, IReadOnlyList<string> terminators, List<MinjaNode> precedingBody, out bool trimRight)
    {
        ValidateBlockKeyword(tagStart, keyword, allowContent: false);
        TrimBeforeTag(tagStart, precedingBody);

        _pos = TagBodyStart(tagStart);
        SkipWhitespace();
        ReadWord();

        CloseTag(tagStart, "%}", out trimRight);

        return ParseNodes(terminators);
    }

    /// <summary>Consumes a block terminator such as endif and returns its -%} elision flag.</summary>
    private bool ConsumeBlockTag(int tagStart, string keyword, List<MinjaNode> precedingBody)
    {
        ValidateBlockKeyword(tagStart, keyword, allowContent: false);
        TrimBeforeTag(tagStart, precedingBody);

        _pos = TagBodyStart(tagStart);
        SkipWhitespace();
        ReadWord();

        CloseTag(tagStart, "%}", out bool trimRight);
        return trimRight;
    }

    /// <summary>Consumes the closing marker of a tag, honouring an attached -%} or -}} elision dash.</summary>
    private void CloseTag(int contextOffset, string closer, out bool trimRight)
    {
        SkipWhitespace();

        trimRight = PeekAt(_pos) == '-' && Matches(_pos + 1, closer);
        if (trimRight)
            _pos++;

        if (!Matches(_pos, closer))
            throw Malformed(Math.Min(_pos, Math.Max(_source.Length - 1, 0)), $"expected '{closer}'");

        _pos += closer.Length;
    }

    private bool Matches(int pos, string value) => (uint)pos <= (uint)_source.Length && _source.AsSpan(pos).StartsWith(value);

    // ---------------------------------------------------------------- text runs and whitespace elision

    /// <summary>Appends a literal run between tags. Whitespace-only runs are real output and stay.</summary>
    private void AddText(List<MinjaNode> nodes, int start, int exclusiveEnd)
    {
        if (exclusiveEnd <= start)
            return;

        string text = _source[start..exclusiveEnd];
        nodes.Add(new TextNode(text) { Offset = start, Length = text.Length });
    }

    /// <summary>Applies -%} and -#} elision to the text run that follows each tagged node in this body.</summary>
    private static void ApplyTrimRight(List<MinjaNode> nodes)
    {
        for (int i = 0; i + 1 < nodes.Count; i++)
        {
            if (nodes[i] is TaggedNode { TrimRight: true } && nodes[i + 1] is TextNode text)
                ReplaceText(nodes, i + 1, text.Text.TrimStart());
        }
    }

    /// <summary>Applies {%- elision to the trailing whitespace of this body's last text run.</summary>
    private static void TrimTrailing(List<MinjaNode> nodes)
    {
        if (nodes.Count > 0 && nodes[^1] is TextNode text)
            ReplaceText(nodes, nodes.Count - 1, text.Text.TrimEnd());
    }

    /// <summary>Applies -%} of a block's opening tag to the leading whitespace of its body's first text run.</summary>
    private static void TrimLeadingFirst(List<MinjaNode> nodes)
    {
        if (nodes.Count > 0 && nodes[0] is TextNode text)
            ReplaceText(nodes, 0, text.Text.TrimStart());
    }

    /// <summary>Replaces a text node with its trimmed form; a run trimmed to nothing is dropped.</summary>
    private static void ReplaceText(List<MinjaNode> nodes, int index, string trimmed)
    {
        if (trimmed.Length == 0)
        {
            nodes.RemoveAt(index);
            return;
        }

        if (nodes[index] is TextNode old)
            nodes[index] = new TextNode(trimmed) { Offset = old.Offset, Length = trimmed.Length };
    }

    private static MinjaSyntaxException Malformed(int position, string detail) =>
        MinjaSyntaxException.Malformed(position, detail);
}

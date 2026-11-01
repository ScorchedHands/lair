namespace Lair.Tokenizer;

/// <summary>
/// A vocabulary entry: the token piece exactly as stored in the GGUF
/// tokenizer metadata, plus its GGUF token type.
/// </summary>
public sealed record GgufTokenAttrs(string Piece, int Type);

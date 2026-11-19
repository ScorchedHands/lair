"""Extract quant code symbol streams from GGUF blocks.

Symbol granularity per format (documented, honest, round-trip consistent):
  Q8_0     : 32 code bytes/block  -> 256-symbol alphabet (byte stream)
  Q4_K     : 256 nibbles/block    -> 16-symbol alphabet (matches kernel element order)
  IQ4_XS   : 256 nibbles/block    -> 16-symbol alphabet (matches kernel element order)
  IQ3_XXS  : 96 code bytes/block  -> 256-symbol alphabet (storage-level coding of the
             bit-packed quant+aux region; 3-bit codes are not byte-aligned, so we code
             the actual stored bytes rather than unpack them)
  IQ3_S    : 96 code bytes/block  -> 256-symbol alphabet (storage-level, same rationale)

Scale/min metadata bytes are NOT coded; they are reported as always-present overhead.
Baseline size = full block bytes (code region + metadata).
"""

from gguf import GGML_TYPE, TYPE_SIZE, BLOCK_SIZE


def _nibbles_q4k(block):
    codes = []
    for j in range(8):
        base = 16 + 32 * (j // 2)
        for k in range(32):
            b = block[base + k]
            codes.append((b >> (4 * (j & 1))) & 0xF)
    return codes


def _nibbles_iq4xs(block):
    codes = []
    for ib in range(8):
        base = 8 + 16 * ib
        for k in range(32):
            b = block[base + (k & 15)]
            codes.append((b >> (4 * (k >> 4))) & 0xF)
    return codes


def _nibbles_iq4nl(block):
    codes = []
    for b in block[2:18]:
        codes.append(b & 0xF)
        codes.append((b >> 4) & 0xF)
    return codes


def _code_bytes(block, start, end):
    return list(block[start:end])


def extract_blocks(ggml_type, tensor_bytes):
    """Return a list of per-block dicts: {symbols, code_bytes, meta_bytes}.

    symbols is the fixed-width code stream for that block. code_bytes is the
    number of stored code bytes in the block; meta_bytes the scale/min bytes kept
    as always-present metadata.
    """
    ts = TYPE_SIZE[ggml_type]
    nb = len(tensor_bytes) // ts
    blocks = []
    for b in range(nb):
        blk = tensor_bytes[b * ts:(b + 1) * ts]
        if ggml_type == 8:
            blocks.append({"symbols": _code_bytes(blk, 2, 34), "code_bytes": 32, "meta_bytes": 2})
        elif ggml_type == 12:
            blocks.append({"symbols": _nibbles_q4k(blk), "code_bytes": 128, "meta_bytes": 16})
        elif ggml_type == 23:
            blocks.append({"symbols": _nibbles_iq4xs(blk), "code_bytes": 128, "meta_bytes": 8})
        elif ggml_type == 18:
            blocks.append({"symbols": _code_bytes(blk, 2, 98), "code_bytes": 96, "meta_bytes": 2})
        elif ggml_type == 21:
            blocks.append({"symbols": _code_bytes(blk, 2, 98), "code_bytes": 96, "meta_bytes": 14})
        elif ggml_type == 20:
            blocks.append({"symbols": _nibbles_iq4nl(blk), "code_bytes": 16, "meta_bytes": 2})
        else:
            raise ValueError(f"unsupported ggml type {ggml_type}")
    return blocks


def extract_stream(ggml_type, tensor_bytes):
    """Return (codes, n_blocks, code_bytes_per_block, meta_bytes_per_block).

    codes is a flat list of symbols in block order. Alphabet size is 16 for the
    4-bit nibble formats, 256 otherwise.
    """
    ts = TYPE_SIZE[ggml_type]
    nb = len(tensor_bytes) // ts

    if ggml_type == 8:          # Q8_0
        codes = _code_bytes(tensor_bytes, 2, 34) if nb else []
        # rebuild across all blocks: 32 code bytes each
        codes = []
        for b in range(nb):
            off = b * ts
            codes.extend(_code_bytes(tensor_bytes, off + 2, off + 34))
        return codes, nb, 32, 2

    if ggml_type == 12:        # Q4_K
        codes = []
        for b in range(nb):
            codes.extend(_nibbles_q4k(tensor_bytes[b * ts:(b + 1) * ts]))
        return codes, nb, 128, 16

    if ggml_type == 23:        # IQ4_XS
        codes = []
        for b in range(nb):
            codes.extend(_nibbles_iq4xs(tensor_bytes[b * ts:(b + 1) * ts]))
        return codes, nb, 128, 8

    if ggml_type == 18:        # IQ3_XXS  (code region = qs[2..98))
        codes = []
        for b in range(nb):
            codes.extend(_code_bytes(tensor_bytes, b * ts + 2, b * ts + 98))
        return codes, nb, 96, 2

    if ggml_type == 21:        # IQ3_S  (code region = qs[2..98); scales+signs kept as meta)
        codes = []
        for b in range(nb):
            codes.extend(_code_bytes(tensor_bytes, b * ts + 2, b * ts + 98))
        return codes, nb, 96, 14

    raise ValueError(f"unsupported ggml type {ggml_type}")

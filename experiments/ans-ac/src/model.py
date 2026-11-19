"""Shared frequency model used identically by AC and ANS.

Counts are built from a chunk's symbol stream. Every symbol present gets a
minimum count of 1 (Laplace floor for present symbols); absent symbols are not
in the model (both codecs then never emit/decode them). The same present-symbol
counts feed both codecs; each codec applies its own documented normalization.

Serialization (shared, codec-agnostic counts): u32 n_present, then per entry
u8 symbol + u16 count.
"""

import struct


def build_counts(symbols):
    counts = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    return counts


def present_model(counts):
    """Sorted list of (symbol, count) for symbols with count >= 1."""
    return sorted((s, c) for s, c in counts.items() if c >= 1)


def serialize_model(model):
    parts = [struct.pack("<I", len(model))]
    for sym, cnt in model:
        parts.append(struct.pack("<BH", sym, cnt))
    return b"".join(parts)


def deserialize_model(data, offset=0):
    n = struct.unpack_from("<I", data, offset)[0]
    off = offset + 4
    model = []
    for _ in range(n):
        sym, cnt = struct.unpack_from("<BH", data, off)
        model.append((sym, cnt))
        off += 3
    return model, off


def model_byte_len(model):
    return 4 + 3 * len(model)

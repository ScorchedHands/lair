"""Order-0 single-state rANS coder over the shared frequency model.

Counts scaled to a fixed table size M = 4096 (largest remainder, min frequency 1).
Encode processes symbols in reverse; decode forward. Final state flushed as 4
leading bytes. Model serialization is the shared raw-count form (model.py),
identical to AC's, so both codecs carry equal model bytes.
"""

from model import present_model, serialize_model, deserialize_model

RANS_L = 1 << 16
M = 1 << 12
M_MASK = M - 1


def _normalize(model):
    counts = [c for _, c in model]
    tot = sum(counts) if counts else 1
    freqs = [max(1, (c * M) // tot) for c in counts]
    diff = M - sum(freqs)
    if diff != 0 and counts:
        rems = sorted(range(len(counts)), key=lambda i: ((counts[i] * M) % tot), reverse=(diff > 0))
        step = 1 if diff > 0 else -1
        j = 0
        while diff != 0:
            i = rems[j % len(rems)]
            if freqs[i] + step >= 1:
                freqs[i] += step
                diff -= step
            j += 1
    cum = []
    acc = 0
    for f in freqs:
        cum.append(acc)
        acc += f
    table = [0] * M
    for i in range(len(model)):
        for slot in range(cum[i], cum[i] + freqs[i]):
            table[slot] = i
    return freqs, cum, table


def encode_chunk(symbols):
    model = present_model(_counts(symbols))
    freqs, cum, _ = _normalize(model)
    sym_of = [s for s, _ in model]
    x = RANS_L
    out = bytearray()
    for s in reversed(symbols):
        i = _index(sym_of, s)
        f = freqs[i]
        c = cum[i]
        x_max = ((RANS_L >> 12) << 8) * f
        while x >= x_max:
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // f) << 12) + (x % f) + c
    head = bytearray()
    for _ in range(4):
        head.append(x & 0xFF)
        x >>= 8
    return serialize_model(model), bytes(head) + bytes(reversed(out))


def decode_chunk(model_bytes, n_symbols, payload):
    model, _ = deserialize_model(model_bytes)
    freqs, cum, table = _normalize(model)
    sym_of = [s for s, _ in model]
    x = payload[0] | (payload[1] << 8) | (payload[2] << 16) | (payload[3] << 24)
    pos = 4
    out = []
    for _ in range(n_symbols):
        slot = x & M_MASK
        i = table[slot]
        out.append(sym_of[i])
        x = freqs[i] * (x >> 12) + slot - cum[i]
        while x < RANS_L:
            x = (x << 8) | (payload[pos] if pos < len(payload) else 0)
            pos += 1
    return out


def _counts(symbols):
    counts = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    return counts


def _index(sym_of, s):
    for i, x in enumerate(sym_of):
        if x == s:
            return i
    raise ValueError(f"symbol {s} not in model")

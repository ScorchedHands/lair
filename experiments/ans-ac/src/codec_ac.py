"""Carryless range arithmetic coder (Subbotin-style) over the shared frequency model.

Encoder and decoder mirror each other with modular 32-bit arithmetic so byte-level
carries are absorbed by the wrap; when the interval is too narrow to carry safely
the range is shrunk to a carryless boundary. Frequencies are scaled so the total
stays within BOT so per-symbol intervals never collapse. Model serialization is the
shared raw-count form in model.py, identical to ANS.
"""

from bisect import bisect_right
from model import present_model, serialize_model, deserialize_model

TOP = 1 << 24
BOT = 1 << 16
MASK = 0xFFFFFFFF


def _scale(freqs):
    tot = sum(freqs)
    if tot <= BOT:
        return freqs
    scale = BOT / tot
    scaled = [max(1, int(f * scale)) for f in freqs]
    while sum(scaled) > BOT:
        i = max(range(len(scaled)), key=lambda k: scaled[k])
        if scaled[i] <= 1:
            break
        scaled[i] -= 1
    return scaled


def _cum(freqs):
    cum = []
    acc = 0
    for f in freqs:
        cum.append(acc)
        acc += f
    return cum


def _renorm_state(low, rng):
    if (low ^ ((low + rng) & MASK)) < TOP:
        return True
    if rng < BOT:
        return "shrink"
    return False


def encode_chunk(symbols):
    model = present_model(_counts(symbols))
    freqs = _scale([c for _, c in model])
    cum = _cum(freqs)
    sym_of = [s for s, _ in model]
    tot = sum(freqs) if freqs else 1

    low = 0
    rng = MASK
    out = bytearray()
    for s in symbols:
        i = _index(sym_of, s)
        r = rng // tot
        low = (low + r * cum[i]) & MASK
        rng = r * freqs[i]
        while True:
            st = _renorm_state(low, rng)
            if st is True:
                pass
            elif st == "shrink":
                rng = (-low) & (BOT - 1)
            else:
                break
            out.append((low >> 24) & 0xFF)
            low = (low << 8) & MASK
            rng = (rng << 8) & MASK
    for _ in range(4):
        out.append((low >> 24) & 0xFF)
        low = (low << 8) & MASK
    return serialize_model(model), bytes(out)


def decode_chunk(model_bytes, n_symbols, payload):
    model, _ = deserialize_model(model_bytes)
    freqs = _scale([c for _, c in model])
    cum = _cum(freqs)
    sym_of = [s for s, _ in model]
    tot = sum(freqs) if freqs else 1

    low = 0
    rng = MASK
    pos = 0
    def nb():
        nonlocal pos
        b = payload[pos] if pos < len(payload) else 0
        pos += 1
        return b
    code = 0
    for _ in range(4):
        code = ((code << 8) | nb()) & MASK

    out = []
    for _ in range(n_symbols):
        r = rng // tot
        f = ((code - low) & MASK) // r
        i = min(bisect_right(cum, f) - 1, len(freqs) - 1)
        if i < 0:
            i = 0
        low = (low + r * cum[i]) & MASK
        rng = r * freqs[i]
        while True:
            st = _renorm_state(low, rng)
            if st is True:
                pass
            elif st == "shrink":
                rng = (-low) & (BOT - 1)
            else:
                break
            code = ((code << 8) | nb()) & MASK
            low = (low << 8) & MASK
            rng = (rng << 8) & MASK
        out.append(sym_of[i])
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

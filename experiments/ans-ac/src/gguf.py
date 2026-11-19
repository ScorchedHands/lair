"""GGUF reader: parse header + tensor table, expose per-tensor byte slices and block geometry.

Only the subset of ggml types LAIR registers is handled here (Q8_0, Q4_K, IQ4_XS,
IQ3_XXS, IQ3_S). IQ4_NL is intentionally skipped. Reads are little-endian.
"""

import struct


class Tensor:
    def __init__(self, name, dims, ggml_type, offset):
        self.name = name
        self.dims = dims          # ggml order: fastest-dim first
        self.ggml_type = ggml_type
        self.offset = offset

    def n_elements(self):
        n = 1
        for d in self.dims:
            n *= d
        return n

    def n_blocks(self):
        return (self.n_elements() + BLOCK_SIZE[self.ggml_type] - 1) // BLOCK_SIZE[self.ggml_type]


GGML_TYPE = {
    8: "Q8_0",
    12: "Q4_K",
    23: "IQ4_XS",
    18: "IQ3_XXS",
    21: "IQ3_S",
    20: "IQ4_NL",   # parsed but skipped downstream
}

# ggml blck_size (elements) and type_size (bytes) per supported type.
BLOCK_SIZE = {8: 32, 12: 256, 23: 256, 18: 256, 21: 256, 20: 32}
TYPE_SIZE = {8: 34, 12: 144, 23: 136, 18: 98, 21: 110, 20: 18}

SUPPORTED = {8, 12, 23, 18, 21}


def _read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8")


def _read_kv_value(f, vtype):
    if vtype == 0:      # uint8
        return struct.unpack("<B", f.read(1))[0]
    if vtype == 1:      # int8
        return struct.unpack("<b", f.read(1))[0]
    if vtype == 2:      # uint16
        return struct.unpack("<H", f.read(2))[0]
    if vtype == 3:      # int16
        return struct.unpack("<h", f.read(2))[0]
    if vtype == 4:      # uint32
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:      # int32
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:      # float32
        return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:      # bool
        return struct.unpack("<?", f.read(1))[0]
    if vtype == 8:      # string
        return _read_string(f)
    if vtype == 9:      # array: element type u32 + count u64
        etype = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        vals = []
        for _ in range(count):
            vals.append(_read_kv_value(f, etype))
        return vals
    if vtype == 10:     # uint64
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:     # int64
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:     # float64
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"unknown ggml kv type {vtype}")


class GgufFile:
    def __init__(self, path):
        self.path = path
        self.metadata = {}
        self.tensors = []
        self.data_start = 0
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError(f"not a GGUF file: {path}")
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            self.version = version
            for _ in range(n_kv):
                key = _read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                self.metadata[key] = _read_kv_value(f, vtype)
            alignment = self.metadata.get("general.alignment", 32)
            # tensor info table
            infos = []
            for _ in range(n_tensors):
                name = _read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                gtype = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                infos.append(Tensor(name, dims, gtype, offset))
            # pad to alignment, then data
            pos = f.tell()
            pad = (-pos) % alignment
            f.read(pad)
            self.data_start = pos + pad
            self.tensors = infos

    def tensor_bytes(self, tensor, data_bytes):
        bs = BLOCK_SIZE[tensor.ggml_type]
        ts = TYPE_SIZE[tensor.ggml_type]
        nb = tensor.n_blocks()
        return data_bytes[tensor.offset: tensor.offset + nb * ts]

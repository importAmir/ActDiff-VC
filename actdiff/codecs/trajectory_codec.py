"""Lossless coding of sparse point trajectories (TRJ1 format).

Layout: b"TRJ1" | T | N | init_bits | initial positions (x: 10 bits, y: 9 bits) | flags (u8)
        [| n_symbols | (symbol, code_length)* | payload_bits | payload]   (only if motion is non-zero)

Frame-to-frame displacements are zigzag-mapped and coded with one canonical Huffman code per chunk.
flags: bit0 = payload present, bit7 = sigma index present, bits1-3 = sigma index.
"""
import heapq
import struct
from io import BytesIO

import numpy as np

MAGIC = b"TRJ1"


class Node:
    _next_id = 0

    def __init__(self, symbol=None, freq=None, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
        self._ord = Node._next_id  # deterministic tie-break
        Node._next_id += 1

    def __lt__(self, other):
        if self.freq != other.freq:
            return self.freq < other.freq
        return self._ord < other._ord


def Write_Varint(f, value):
    value = int(value)
    while value >= 0x80:
        f.write(struct.pack('B', (value & 0x7F) | 0x80))
        value >>= 7
    f.write(struct.pack('B', value & 0x7F))


def Read_Varint(f):
    result, shift = 0, 0
    while True:
        byte = f.read(1)[0]
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result
        shift += 7
        if shift >= 64:
            raise ValueError("Varint too large")


def Zigzag_Encode(x):
    x32 = x.astype(np.int32)
    return ((x32 << 1) ^ (x32 >> 31)).astype(np.uint16)


def Zigzag_Decode(z):
    z32 = z.astype(np.uint32)
    return ((z32 >> 1) ^ (-(z32 & 1))).astype(np.int16)


def Huffman_Code_Lengths(samples):
    symbols, counts = np.unique(samples, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    heap = [Node(symbol=s, freq=p) for s, p in zip(symbols, probs)]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1, n2 = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, Node(freq=n1.freq + n2.freq, left=n1, right=n2))

    lengths = {}
    stack = [(heap[0], 0)]
    while stack:
        node, depth = stack.pop()
        if node.symbol is not None:
            lengths[int(node.symbol)] = max(1, depth)  # a single-symbol stream still needs 1 bit
        else:
            stack.append((node.left, depth + 1))
            stack.append((node.right, depth + 1))
    return lengths


def Canonical_Codes(lengths):
    """symbol -> (code, length) for a canonical Huffman code."""
    codes, code, prev_len = {}, 0, 0
    for sym, length in sorted(lengths.items(), key=lambda kv: (kv[1], kv[0])):
        code <<= length - prev_len
        codes[sym] = (code, length)
        code += 1
        prev_len = length
    return codes


def Pack_Symbols(symbols, codes):
    out, buf, nbits = bytearray(), 0, 0
    for s in symbols:
        code, length = codes[int(s)]
        buf = (buf << length) | code
        nbits += length
        while nbits >= 8:
            out.append((buf >> (nbits - 8)) & 0xFF)
            nbits -= 8
    if nbits:
        out.append((buf << (8 - nbits)) & 0xFF)
    total_bits = (len(out) - 1) * 8 + nbits if nbits else len(out) * 8
    return bytes(out), total_bits


def Iter_Bits(data, total_bits):
    used = 0
    for b in data:
        for k in range(8):
            if used == total_bits:
                return
            yield (b >> (7 - k)) & 1
            used += 1


def Decode_Symbols(bits, lengths, count):
    table, code, prev_len = {}, 0, 0
    for sym, length in sorted(lengths.items(), key=lambda kv: (kv[1], kv[0])):
        code <<= length - prev_len
        table[(length, code)] = sym
        code += 1
        prev_len = length

    def Symbols():
        length, code = 0, 0
        for b in bits:
            code = (code << 1) | b
            length += 1
            sym = table.get((length, code))
            if sym is not None:
                yield sym
                length, code = 0, 0

    decoded = np.fromiter(Symbols(), dtype=np.uint16, count=count)
    if len(decoded) != count:
        raise ValueError(f"Trajectory stream underflow: expected {count} symbols, got {len(decoded)}")
    return decoded


def Pack_Positions(positions):
    """(N, 2) int (x, y) -> bytes, using 10 bits for x and 9 bits for y."""
    out, buf, nbits = bytearray(), 0, 0
    for x, y in positions:
        buf = (buf << 10) | (int(x) & 0x3FF)
        buf = (buf << 9) | (int(y) & 0x1FF)
        nbits += 19
        while nbits >= 8:
            out.append((buf >> (nbits - 8)) & 0xFF)
            nbits -= 8
            buf &= (1 << nbits) - 1
    if nbits:
        out.append((buf << (8 - nbits)) & 0xFF)
    return bytes(out), len(positions) * 19


def Unpack_Positions(data, n, total_bits):
    bits = Iter_Bits(data, total_bits)
    positions = np.zeros((n, 2), dtype=np.int16)
    for i in range(n):
        x = y = 0
        for _ in range(10):
            x = (x << 1) | next(bits)
        for _ in range(9):
            y = (y << 1) | next(bits)
        positions[i] = (x, y)
    return positions


def Compress_Trajectories(tracks, save_path, sigma_index=None):
    """Write (T, N, 2) integer pixel tracks to `save_path`; returns the stream size in bytes."""
    T, N, _ = tracks.shape
    tracks = tracks.astype(np.int16)
    symbols = Zigzag_Encode(np.diff(tracks.astype(np.int32), axis=0).reshape(T - 1, N * 2)).flatten()
    has_motion = not np.all(symbols == 0)

    flags = 1 if has_motion else 0
    if sigma_index is not None:
        if not 0 <= sigma_index <= 7:
            raise ValueError(f"sigma_index must be in [0, 7], got {sigma_index}")
        flags |= 0x80 | ((sigma_index & 0x07) << 1)

    buf = BytesIO()
    buf.write(MAGIC)
    Write_Varint(buf, T)
    Write_Varint(buf, N)
    packed_init, init_bits = Pack_Positions(tracks[0])
    Write_Varint(buf, init_bits)
    buf.write(packed_init)
    buf.write(struct.pack('B', flags))

    if has_motion:
        lengths = Huffman_Code_Lengths(symbols)
        payload, total_bits = Pack_Symbols(symbols, Canonical_Codes(lengths))
        Write_Varint(buf, len(lengths))
        for sym in sorted(lengths):
            Write_Varint(buf, sym)
            buf.write(struct.pack('B', lengths[sym] & 0xFF))
        Write_Varint(buf, total_bits)
        buf.write(payload)

    data = buf.getvalue()
    with open(save_path, "wb") as f:
        f.write(data)
    return len(data)


def Read_Trajectory_Length(path):
    """Number of frames T stored in a trajectory stream."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"Not a TRJ1 trajectory stream: {path}")
        return Read_Varint(f)


def Decompress_Trajectories(path, target_T=None):
    """Read (T, N, 2) int16 tracks, optionally resampled in time to `target_T` frames."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"Not a TRJ1 trajectory stream: {path}")
        T = Read_Varint(f)
        N = Read_Varint(f)
        init_bits = Read_Varint(f)
        init = Unpack_Positions(f.read((init_bits + 7) // 8), N, init_bits)
        flags = f.read(1)[0]

        if flags & 0x01:
            lengths = {}
            for _ in range(Read_Varint(f)):
                sym = Read_Varint(f)
                lengths[sym] = f.read(1)[0]
            total_bits = Read_Varint(f)
            payload = f.read((total_bits + 7) // 8)
            symbols = Decode_Symbols(Iter_Bits(payload, total_bits), lengths, (T - 1) * N * 2)
            deltas = Zigzag_Decode(symbols.reshape(T - 1, N, 2)).astype(np.int32)
            tracks = np.empty((T, N, 2), dtype=np.int32)
            tracks[0] = init
            np.cumsum(deltas, axis=0, out=tracks[1:])
            tracks[1:] += tracks[0]
            tracks = tracks.astype(np.int16)
        else:
            tracks = np.tile(init[None], (T, 1, 1))

    if target_T is not None and target_T != T:
        tracks = Resample_Time(tracks, target_T)
    return tracks


def Resample_Time(tracks, target_T):
    """Linear interpolation of (T, N, 2) tracks onto `target_T` evenly spaced frames (keeps dtype)."""
    T, N, D = tracks.shape
    src_t = np.arange(T, dtype=np.float32)
    tgt_t = np.linspace(0.0, float(T - 1), num=target_T, dtype=np.float32)
    out = np.empty((target_T, N, D), dtype=tracks.dtype)
    for d in range(D):
        for n in range(N):
            out[:, n, d] = np.interp(tgt_t, src_t, tracks[:, n, d])
    return out

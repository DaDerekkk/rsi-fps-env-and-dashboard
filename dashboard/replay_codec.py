"""AGX replay chunk decoder, ported from the saved official viewer q2/L2.

Source: work/sources/agx-index.js. Verify packed, encoded and raw digests;
this is a transport codec only, not an inference or simulation.
"""
import hashlib
import struct
import zlib


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inflate(data, size):
    decoder = zlib.decompressobj()
    result = decoder.decompress(data, size + 1)
    if len(result) != size or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
        raise ValueError('Invalid or oversized zlib stream')
    return result


def delta_decode(data, raw_size):
    if data[:6] != b'AGXD1\0':
        raise ValueError('Wrong delta magic')
    pos = 6
    output = bytearray()
    previous = b''
    chain = 24
    while pos < len(data):
        if pos + 5 > len(data):
            raise ValueError('Truncated delta record')
        kind, size = struct.unpack_from('<BI', data, pos)
        pos += 5
        prefix = suffix = 0
        if not 1 <= size <= raw_size - len(output):
            raise ValueError('Invalid delta row size')
        if kind == 0:
            chain = 1
        elif kind == 1:
            if chain >= 24 or pos + 8 > len(data):
                raise ValueError('Invalid delta chain')
            prefix, suffix = struct.unpack_from('<II', data, pos)
            pos += 8
            chain += 1
            if prefix + suffix > min(len(previous), size):
                raise ValueError('Invalid delta spans')
        else:
            raise ValueError('Unknown delta record')
        literal_size = size - prefix - suffix
        if pos + literal_size > len(data):
            raise ValueError('Truncated delta literal')
        row = previous[:prefix] + data[pos:pos + literal_size] + (previous[-suffix:] if suffix else b'')
        pos += literal_size
        output.extend(row)
        previous = row
    if len(output) != raw_size:
        raise ValueError('Incomplete decoded chunk')
    return bytes(output)


def decode_chunk(data, headers):
    h = {k.lower(): v for k, v in headers.items()}
    codec = h.get('x-agx-chunk-codec')
    if codec is None:
        return data
    packed_size = int(h['x-agx-packed-bytes'])
    raw_size = int(h['x-agx-raw-bytes'])
    if not 1 <= raw_size <= 1048576 or packed_size != len(data) or packed_size > 1049600:
        raise ValueError('Chunk size header mismatch')
    if digest(data) != h['x-agx-packed-sha256']:
        raise ValueError('Packed digest mismatch')
    if codec == 'zlib':
        raw = inflate(data, raw_size)
    elif codec == 'frame-delta+zlib':
        encoded_size = int(h['x-agx-encoded-bytes'])
        if not 1 <= encoded_size <= 1048576:
            raise ValueError('Invalid encoded size')
        encoded = inflate(data, encoded_size)
        if digest(encoded) != h['x-agx-encoded-sha256']:
            raise ValueError('Encoded digest mismatch')
        raw = delta_decode(encoded, raw_size)
    else:
        raise ValueError('Unknown chunk codec')
    if digest(raw) != h['x-agx-raw-sha256']:
        raise ValueError('Raw digest mismatch')
    return raw

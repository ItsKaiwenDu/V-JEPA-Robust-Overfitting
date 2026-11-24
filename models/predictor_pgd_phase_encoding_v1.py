"""Unbound stdlib phase-byte prototype; no runtime, timer or admission API.

Only locally produced enc(row) bytes from a private string-keyed row are in
scope. This is not a general canonicalizer or an external-input admission gate.
"""
import json
import math


def _phase_json_value_end(text, at, decoder):
    """Walk compact JSON without recursive container decoding or key searches."""
    stack = []
    need_value = True
    while True:
        if at >= len(text):
            raise ValueError('Truncated canonical phase value.')
        if need_value:
            first = text[at]
            if first == '{':
                stack.append('object-first')
                at += 1
            elif first == '[':
                stack.append('array-first')
                at += 1
            else:
                value, at = decoder.raw_decode(text, at)
                if (type(value) not in (str, int, float, bool, type(None))
                        or type(value) is float and not math.isfinite(value)):
                    raise ValueError('Finite canonical phase scalar required.')
            need_value = False
        if not stack:
            return at
        if at >= len(text):
            raise ValueError('Truncated canonical phase container.')
        state = stack[-1]
        char = text[at]
        if state in ('object-first', 'object-key'):
            if state == 'object-first' and char == '}':
                stack.pop()
                at += 1
            else:
                if char != '"':
                    raise ValueError('Canonical phase object key required.')
                key, at = decoder.raw_decode(text, at)
                if type(key) is not str or at >= len(text) or text[at] != ':':
                    raise ValueError('Canonical phase object separator required.')
                at += 1
                stack[-1] = 'object-comma'
                need_value = True
        elif state == 'object-comma':
            if char == '}':
                stack.pop()
                at += 1
            elif char == ',':
                stack[-1] = 'object-key'
                at += 1
            else:
                raise ValueError('Canonical phase object end required.')
        elif state in ('array-first', 'array-value'):
            if state == 'array-first' and char == ']':
                stack.pop()
                at += 1
            else:
                stack[-1] = 'array-comma'
                need_value = True
        elif state == 'array-comma':
            if char == ']':
                stack.pop()
                at += 1
            elif char == ',':
                stack[-1] = 'array-value'
                at += 1
            else:
                raise ValueError('Canonical phase array end required.')


def insert_row_sha256(encoded, digest):
    """Insert/replace a top-level member using structural byte offsets.

The immutable input must come from the unchanged local enc(row), whose root
has unique string keys. Values are retained verbatim; no value is reserialized.
Producer provenance, rather than this parser, establishes canonical spelling.
"""
    if (type(encoded) is not bytes or not encoded.startswith(b'{')
            or not encoded.endswith(b'}\n') or len(encoded) < 3
            or type(digest) is not str or len(digest) != 64
            or any(char not in '0123456789abcdef' for char in digest)):
        raise ValueError('Immutable canonical phase object and lowercase SHA256 required.')
    text = encoded.decode('ascii')  # Unchanged enc uses ensure_ascii=True.
    decoder = json.JSONDecoder()
    end = len(text)-2
    at = 1
    previous = None
    replacement = insertion = None
    members = 0
    while at < end:
        start = at
        if text[at] != '"':
            raise ValueError('String-keyed canonical phase root required.')
        key, key_end = decoder.raw_decode(text, at)
        if (type(key) is not str or previous is not None and key <= previous
                or key_end >= end or text[key_end] != ':'):
            raise ValueError('Strictly sorted unique phase root keys required.')
        value_end = _phase_json_value_end(text, key_end+1, decoder)
        if value_end > end:
            raise ValueError('Phase root extent changed.')
        if key == 'row_sha256':
            replacement = (start, value_end)
        elif insertion is None and key > 'row_sha256':
            insertion = start
        members += 1
        previous = key
        if value_end == end:
            at = end
        elif text[value_end] == ',' and value_end+1 < end:
            at = value_end+1
        else:
            raise ValueError('Compact phase root separator required.')
    if at != end:
        raise ValueError('Exact canonical phase root boundary required.')
    member = b'"row_sha256":"'+digest.encode('ascii')+b'"'
    if replacement is not None:
        start, stop = replacement
        return encoded[:start]+member+encoded[stop:]
    if insertion is not None:
        return encoded[:insertion]+member+b','+encoded[insertion:]
    if members:
        return encoded[:-2]+b','+member+b'}\n'
    return b'{'+member+b'}\n'

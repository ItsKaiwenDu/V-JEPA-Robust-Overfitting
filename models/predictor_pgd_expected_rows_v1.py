"""Unbound LOCAL metadata prototype; never an original Snapshot or tensor adapter.

Only the private outer list object is reused. Every tuple and identity-port call
is fresh. Eligibility inspection is real extra work. No speed/memory fit claim.
Input rows/tokens are synthetic; no model, tensor, clock, process or network port.
"""


def baseline(kind, records, observed, identity):
    """Fresh list/tuple construction with the original row shapes and order."""
    if kind == 'parameters':
        return observed != [(n, identity(p), r) for n, p, r in records]
    if kind == 'buffers':
        return observed != [(n, identity(b)) for n, b in records]
    if kind == 'modules':
        return observed != [(n, identity(m)) for n, m, _, _, _ in records]
    raise ValueError('Synthetic row kind required.')


def _canonical(kind, records, observed):
    """Inspect only exact builtins; never consume a foreign iterable or compare it."""
    if type(records) is not list or type(observed) is not list:
        return False
    record_arity = {'parameters': 3, 'buffers': 2, 'modules': 5}[kind]
    output_arity = 3 if kind == 'parameters' else 2
    for row in records:
        if type(row) is not tuple or len(row) != record_arity:
            return False
        if type(row[0]) is not str or type(row[1]) is not int:
            return False
        if kind == 'parameters' and type(row[2]) is not bool:
            return False
    for row in observed:
        if type(row) is not tuple or len(row) != output_arity:
            return False
        if type(row[0]) is not str or type(row[1]) is not int:
            return False
        if kind == 'parameters' and type(row[2]) is not bool:
            return False
    return True


class PrivateOuterList:
    """Synthetic owner-local storage. Python reflection is NOT a security boundary.

    compare returns only the comparison result; close returns None. diagnostic
    returns fresh primitive metadata. No ordinary method returns the backing list.
    Privileged test reflection deliberately injects corruption; this is no target
    ownership, thread safety or isolation proof. Caller must not share an owner.
    """

    def __init__(self, kind):
        if type(kind) is not str or kind not in ('parameters', 'buffers', 'modules'):
            raise ValueError('Synthetic row kind required.')
        self._kind = kind
        self._rows = []
        self._busy = False
        self._closed = False
        self._route = 'unused'

    def compare(self, records, observed, identity):
        if self._closed or self._busy or not _canonical(self._kind, records, observed):
            self._route = 'fallback'
            return baseline(self._kind, records, observed, identity)
        if type(self._rows) is not list:
            self._rows = []
        rows = self._rows
        rows.clear()
        self._busy = True
        self._route = 'reuse'
        try:
            if self._kind == 'parameters':
                for n, p, r in records:
                    rows.append((n, identity(p), r))
            elif self._kind == 'buffers':
                for n, b in records:
                    rows.append((n, identity(b)))
            else:
                for n, m, _, _, _ in records:
                    rows.append((n, identity(m)))
            return observed != rows
        finally:
            rows.clear()
            self._busy = False
            self._route = 'reuse'

    def close(self):
        if self._busy:
            raise ValueError('Synthetic owner is active.')
        if type(self._rows) is list:
            self._rows.clear()
        else:
            self._rows = []
        self._closed = True

    def diagnostic(self):
        return {'kind': self._kind, 'busy': self._busy, 'closed': self._closed,
                'route': self._route,
                'retained_rows': len(self._rows) if type(self._rows) is list else None}

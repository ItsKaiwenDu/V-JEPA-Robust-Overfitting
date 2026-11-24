"""Local symbolic requirements model; never a tensor adapter or execution gate.

Inputs are inert JSON labels and small opaque bytes. No target behavior, physical
ownership, operator equivalence, elapsed time or resource fit is authenticated.
This module is excluded from the frozen worker/upload graph.
"""
import hashlib


class Refusal(ValueError):
    """A synthetic model refusal, not an original runtime exception."""


class SymbolicLifetime:
    def __init__(self, supplied):
        self.supplied = dict(supplied)
        self.state = 'unqualified'
        self.allocations = {}
        self.events = []
        self.primary = None
        self.notes = []
        self.capture_pending = self.consumer_pending = False
        self.captured_writer = None
        self.hasher = None
        self.consumer_complete = False
        self.used_generations = set()
        self.generation = None
        self.digests = []
        self.byte_streams = []
        self.bytes_read = self.peak_live_bytes = 0
        self.now = 0
        self.clock_domain = supplied['clock_domain']
        self.owner = supplied['owner']
        self.writer = supplied['writer']
        self.escape = False

    def _require(self, condition, code):
        if not condition:
            raise Refusal(code)

    def _owned(self):
        return sorted(name for name, row in self.allocations.items() if row['owned'])

    def _memory(self):
        self._require(len(self.allocations) <= 8, 'ALLOCATION_CAP')
        live = sum(row['capacity'] for row in self.allocations.values())
        self.peak_live_bytes = max(self.peak_live_bytes, live)

    def _guard(self):
        self._require(self.owner == self.supplied['owner'], 'OWNER_CHANGED')
        self._require(self.clock_domain == self.supplied['clock_domain'], 'CLOCK_DOMAIN')
        self._require(type(self.now) is int and self.now < self.supplied['deadline'], 'DEADLINE')

    def _failure(self, code):
        if self.primary is None:
            self.primary = code
        else:
            self.notes.append(code)
        self.hasher = None
        self.state = 'failed_held' if self._owned() or self.capture_pending or self.consumer_pending else 'failed_released'

    def step(self, action):
        self._require(len(self.events) < 64, 'EVENT_CAP')
        op = action['op']
        before = self.state
        refused = None
        try:
            if self.primary is not None:
                self._require(op in ('quiescent', 'release', 'cleanup_error'), 'FAILED_NO_RETRY')
            if op not in ('set_clock', 'mutate', 'fault', 'cleanup_error', 'quiescent', 'release', 'require_effect'):
                self._guard()
            self._apply(op, action)
        except Refusal as error:
            refused = str(error)
            self._failure(refused)
        self._memory()
        self.events.append({'op': op, 'before': before, 'after': self.state,
                            'refusal': refused, 'owned': self._owned(),
                            'capture_pending': self.capture_pending,
                            'consumer_pending': self.consumer_pending})

    def _apply(self, op, action):
        s = self.supplied
        if op == 'begin':
            self._require(self.state in ('unqualified', 'released'), 'OBSERVATION_ACTIVE')
            self._require(s['authority'] == 'fake-reviewed-only', 'AUTHORITY_UNPROVEN')
            self._require(s['domain'] == 'fake-known-only', 'DOMAIN_UNPROVEN')
            self._require(s['boundary'] == s['expected_boundary'] and not s['moved_boundary'], 'BOUNDARY_MOVED')
            self._require(s['recipe'] == s['expected_recipe'], 'RECIPE_CHANGED')
            generation = action['generation']
            self._require(type(generation) is str and generation not in self.used_generations, 'GENERATION_REUSED')
            self._require(not self._owned(), 'RESOURCES_HELD')
            self.used_generations.add(generation)
            self.generation = generation
            self.allocations = {'source': {'owned': False, 'capacity': s['source_capacity'], 'depends_on': None}}
            self.capture_pending = self.consumer_pending = False
            self.state = 'capture_pending'
        elif op == 'allocate':
            self._require(self.state in ('capture_pending', 'captured', 'export_held'), 'ALLOCATION_STATE')
            name, capacity = action['name'], action['capacity']
            self._require(name != 'source' and name not in self.allocations, 'ALLOCATION_REUSED')
            self._require(type(capacity) is int and 0 <= capacity <= 4096, 'CAPACITY')
            self._require(len(self.allocations) < 8, 'ALLOCATION_CAP')
            self.allocations[name] = {'owned': True, 'capacity': capacity, 'depends_on': None}
        elif op == 'capture':
            self._require(self.state == 'capture_pending' and not self.capture_pending and 'capture' in self.allocations, 'CAPTURE_STATE')
            self._require(s['writer_order'] and s['other_writers_ordered'], 'WRITER_ORDER_UNKNOWN')
            self._require(s['private'] and not s['aliases_source'], 'CAPTURE_ALIASES')
            self._require(s['alias_graph_known'], 'ALIAS_GRAPH_UNKNOWN')
            self.captured_writer = self.writer
            self.capture_pending = True
        elif op == 'confirm':
            self._require(self.capture_pending, 'NO_PENDING_CAPTURE')
            self._require(action['token'] == [self.owner, self.generation, s['stream'], self.clock_domain], 'COMPLETION_MISMATCH')
            self._require(self.writer == self.captured_writer, 'WRITER_CHANGED')
            self.capture_pending = False
            self.state = 'captured'
        elif op == 'export':
            self._require(self.state == 'captured' and not self.capture_pending, 'CAPTURE_PENDING')
            self._require(not self.escape and not s['exporter_escape'], 'EXPORTER_ESCAPED')
            self._require(not s['export_error'], s['export_error'] or 'EXPORT_ERROR')
            self._require('exporter' not in self.allocations and len(self.allocations) < 8, 'ALLOCATION_CAP')
            self.allocations['exporter'] = {'owned': True, 'capacity': 3, 'depends_on': 'capture'}
            self.state = 'export_held'
        elif op == 'consume':
            self._require(not self.capture_pending, 'CAPTURE_PENDING')
            self._require(self.state == 'export_held' and 'capture' in self.allocations and 'exporter' in self.allocations, 'CONSUME_STATE')
            self._require(not self.escape and not s['exporter_escape'] and not s['reentrant'], 'UNTRUSTED_CONSUMER')
            self._require(s['transform'] == 'reinterpret' and not s['padding'], 'BYTE_TRANSFORM')
            self._require(s['read_offset'] == s['logical_offset'], 'BYTE_EXTENT')
            if s['recipe'] == 'tensor_hash':
                chunks = [bytes.fromhex(s['payload_hex'])]
            else:
                self._require(s['state_names'] == s['expected_state_names'] and not s['added_separator'], 'STATE_STREAM_CHANGED')
                chunks = []
                for name, raw in zip(s['state_names'], s['state_payload_hex']):
                    chunks.extend((name.encode(), bytes.fromhex(raw)))
            stream = b''.join(chunks)
            self._require(len(stream) == s['logical_extent'], 'BYTE_EXTENT')
            self._require(sum(len(bytes.fromhex(raw)) for raw in s['state_payload_hex']) <= 4096 and len(stream) <= 4096, 'PAYLOAD_CAP')
            self.hasher = hashlib.sha256()
            limit = action.get('read_limit', len(stream))
            self._require(type(limit) is int and 0 <= limit <= len(stream), 'READ_EXTENT')
            self.consumer_complete = limit == len(stream)
            if self.consumer_complete:
                for chunk in chunks:
                    self.hasher.update(chunk)
            else:
                self.hasher.update(stream[:limit])
            self.bytes_read += limit
            self.byte_streams.append(stream[:limit].hex())
            self.consumer_pending = True
            self.state = 'consuming'
        elif op == 'consumer_done':
            self._require(self.state == 'consuming' and self.consumer_pending, 'CONSUMER_STATE')
            self._require(self.consumer_complete, 'CONSUMER_INCOMPLETE')
            self.consumer_pending = False
            self.state = 'consumed'
        elif op == 'release':
            self._require(not self.capture_pending and not self.consumer_pending, 'PREMATURE_RELEASE')
            name = action['name']
            self._require(name in self.allocations and self.allocations[name]['owned'], 'RELEASE_NOT_OWNED')
            self._require(not any(row['depends_on'] == name for row in self.allocations.values()), 'DEPENDENT_HELD')
            del self.allocations[name]
            if self.primary is not None and not self._owned():
                self.state = 'failed_released'
        elif op == 'receipt':
            self._require(self.state == 'consumed' and self.hasher is not None, 'RECEIPT_STATE')
            self._require(not self._owned() and not self.capture_pending and not self.consumer_pending, 'RESOURCES_HELD')
            self.digests.append({'generation': self.generation, 'digest': self.hasher.hexdigest()})
            self.hasher = None
            self.state = 'released'
        elif op == 'quiescent':
            self._require(self.primary is not None and action['confirmed'] is True, 'QUIESCENCE_UNPROVEN')
            self._require(action['token'] == [self.supplied['owner'], self.generation, s['stream'], s['clock_domain']], 'QUIESCENCE_MISMATCH')
            self.capture_pending = self.consumer_pending = False
            self.state = 'failed_held' if self._owned() else 'failed_released'
        elif op == 'fault':
            raise Refusal(action['code'])
        elif op == 'cleanup_error':
            raise Refusal(action['code'])
        elif op == 'set_clock':
            self._require(type(action['now']) is int and action['now'] >= self.now, 'CLOCK_RESET')
            self._require(action['domain'] == self.clock_domain == s['clock_domain'], 'CLOCK_DOMAIN')
            self.now, self.clock_domain = action['now'], action['domain']
        elif op == 'mutate':
            key = action['key']
            self._require(key in ('owner', 'writer', 'escape'), 'MUTATION_KEY')
            if key == 'owner': self.owner = action['value']
            elif key == 'writer': self.writer = action['value']
            else: self.escape = action['value']
        elif op == 'account':
            self._require(not action['overhead_subtracted'], 'OVERHEAD_SUBTRACTED')
            self._require(action['reported_peak'] == self.peak_live_bytes, 'LIVE_BYTES_MISMATCH')
        elif op == 'require_effect':
            # A label requesting an effect is refused before any real port lookup.
            raise Refusal('REAL_EFFECT_NEEDED')
        else:
            raise Refusal('UNKNOWN_EVENT')

    def result(self):
        return {'state': self.state, 'primary': self.primary, 'notes': self.notes,
                'owned': self._owned(), 'capture_pending': self.capture_pending,
                'consumer_pending': self.consumer_pending, 'digests': self.digests,
                'byte_streams': self.byte_streams, 'bytes_read': self.bytes_read,
                'peak_live_bytes': self.peak_live_bytes, 'events': self.events,
                'history': dict(self.supplied['history']), 'real_effect_attempts': 0,
                'physical_ownership_proven': False, 'runtime_refusal_equivalence_proven': False,
                'execution_binding': None, 'budget_gate_opened': False,
                'research_admitted': False, 'training_admitted': False}


def run_program(supplied, actions):
    """Consume only a bounded inert program; return synthetic observations."""
    if type(actions) is not list or len(actions) > 64:
        raise ValueError('At most64 supplied symbolic events required.')
    model = SymbolicLifetime(supplied)
    for action in actions:
        model.step(action)
    return model.result()

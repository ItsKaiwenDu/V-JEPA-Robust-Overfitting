"""UNEXECUTED local documentary validation draft; no real admission mechanism.

Only a new, separately approved local validation may import or call this module.
The raw-byte API and conservative local bounds below are draft refinements,
not an equivalent replacement for any original runtime input/guard domain.
Reference record formats and closed source namespaces remain unsupported.
No I/O, target/scientific port, grant writer, runtime binding, CLI or cache.
Every receipt fixes actual admission false, real attempts0 and binding null.
"""
import json
import math

MAX_PAYLOAD_BYTES = 65536
MAX_NESTING = 8
MAX_TREE_NODES = 2048
MAX_OBJECT_KEYS = 128
MAX_ARRAY_ITEMS = 64
MAX_STRING_CHARACTERS = 4096
MAX_INTEGER_TEXT = 20
MAX_FLOAT_TEXT = 64
MIN_INTEGER = -9223372036854775808
MAX_INTEGER = 9223372036854775807

EVIDENCE_STATUSES = ('UNKNOWN', 'UNAVAILABLE', 'SYNTHETIC', 'SOURCE_ASSERTION', 'REVIEWED_SAVED_RECORD')
OUTCOME_CLASSES = ('DRAFT_NOT_RUN', 'INHERITED_SYNTHETIC', 'SAVED_PARTIAL', 'SAVED_FAILED', 'SAVED_DOCUMENTARY_REVIEW')
UNRESOLVED_DOMAINS = ('closed-name-to-hash-map', 'bounded-evidence-reference-list',
                      'bounded-documentary-reference-or-null', 'bounded-documentary-reference')
PROFILE_FIELDS = (
    ('P01', (('record_kind', 'exact-string'), ('revision', 'exact-integer'),
             ('artifact_sha256', 'canonical-hash-or-null'), ('parent_specification_sha256', 'canonical-hash-or-null'),
             ('evidence_status', 'evidence-status-label'))),
    ('P02', (('record_kind', 'exact-string'), ('reviewed_source_hashes', 'closed-name-to-hash-map'),
             ('candidate_source_digest', 'canonical-hash-or-null'),
             ('inverse_domain_refusal_ownership_evidence', 'bounded-evidence-reference-list'),
             ('evidence_status', 'evidence-status-label'))),
    ('P03', (('record_kind', 'exact-string'), ('user_basis_reference', 'bounded-documentary-reference-or-null'),
             ('source_control_identity', 'closed-name-to-hash-map'), ('current_real_attempt_authority', 'exact-integer'),
             ('old_authority_reuse_or_retroactive_permission', 'exact-false'), ('evidence_status', 'evidence-status-label'))),
    ('P04', (('record_kind', 'exact-string'), ('retained_claim_references', 'bounded-evidence-reference-list'),
             ('history_status', 'evidence-status-label'), ('new_operational_claim', 'exact-null'),
             ('reset_or_retry_permission', 'exact-false'), ('evidence_status', 'evidence-status-label'))),
    ('P05', (('record_kind', 'exact-string'), ('clock_owner_domain_reference', 'bounded-documentary-reference-or-null'),
             ('origin_and_deadline_references', 'bounded-evidence-reference-list'),
             ('timestamp_domain', 'finite-exact-int-or-float-or-null'),
             ('timer_reset_or_extra_window_permission', 'exact-false'), ('evidence_status', 'evidence-status-label'))),
    ('P06', (('record_kind', 'exact-string'), ('identity_references', 'bounded-evidence-reference-list'),
             ('held_handle_death_status_closure_references', 'bounded-evidence-reference-list'),
             ('quiescence_status', 'evidence-status-label'), ('target_preflight_or_signal_permission', 'exact-false'),
             ('evidence_status', 'evidence-status-label'))),
    ('P07', (('record_kind', 'exact-string'), ('workload_source_reference', 'bounded-documentary-reference'),
             ('fixed_caps_reference', 'bounded-documentary-reference'),
             ('scientific_guard_hash_owner_durability_clock_change_permission', 'exact-false'),
             ('complete_unit_fit_status', 'evidence-status-label'), ('evidence_status', 'evidence-status-label'))),
    ('P08', (('record_kind', 'exact-string'), ('outcome_class', 'outcome-label'),
             ('completed_role_prefix_and_receipt_references', 'bounded-evidence-reference-list'),
             ('real_work_history_references', 'bounded-evidence-reference-list'),
             ('real_execution_research_training_admitted', 'three-exact-false-fields'),
             ('evidence_status', 'evidence-status-label'))),
)


class _DraftRefusal(Exception):
    """Expected documentary refusal with a fixed bounded source code only."""
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _receipt(disposition, code, *, profile=None, evidence_status=None, unresolved=(), outer_shape_checked=False):
    return {
        'kind': 'LOCAL-DOCUMENTARY-DRAFT-RECEIPT-v1',
        'documentary_disposition': disposition, 'refusal_code': code,
        'profile': profile, 'evidence_status': evidence_status,
        'unresolved_reasons': list(unresolved), 'outer_shape_checked': outer_shape_checked,
        'structure_or_labels_authenticate_evidence': False,
        'all_requirements_qualified': False, 'implementation_status': 'DRAFT_NOT_QUALIFIED',
        'real_execution_admitted': False, 'research_admitted': False, 'training_admitted': False,
        'maximum_new_real_attempts': 0, 'execution_binding': None, 'budget_gate_opened': False,
        'target_approval': 'HOLD_NOT_READY_NOT_REQUESTED',
    }


def _integer(text):
    if type(text) is not str or len(text) > MAX_INTEGER_TEXT:
        raise _DraftRefusal('D_INTEGER_BOUND')
    value = int(text)
    if not MIN_INTEGER <= value <= MAX_INTEGER:
        raise _DraftRefusal('D_INTEGER_BOUND')
    return value


def _floating(text):
    if type(text) is not str or len(text) > MAX_FLOAT_TEXT:
        raise _DraftRefusal('D_FLOAT_BOUND')
    value = float(text)
    if not math.isfinite(value):
        raise _DraftRefusal('D_NONFINITE')
    return value


def _constant(text):
    raise _DraftRefusal('D_NONFINITE')


def _pairs(rows):
    if len(rows) > MAX_OBJECT_KEYS:
        raise _DraftRefusal('D_OBJECT_BOUND')
    result = {}
    for key, value in rows:
        if type(key) is not str or key in result:
            raise _DraftRefusal('D_DUPLICATE_KEY')
        result[key] = value
    return result


def _scan_nesting(text):
    stack = []
    quoted = escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in ('{', '['):
            stack.append(character)
            if len(stack) > MAX_NESTING:
                raise _DraftRefusal('D_NESTING_BOUND')
        elif character in ('}', ']'):
            if not stack or (stack.pop(), character) not in (('{', '}'), ('[', ']')):
                raise _DraftRefusal('D_JSON_SYNTAX')
    if quoted or stack:
        raise _DraftRefusal('D_JSON_SYNTAX')


def _tree_shape(value):
    """Post-decode logical bounds; not peak parser/heap/RSS enforcement."""
    pending = [(value, 0)]
    seen = 0
    while pending:
        item, depth = pending.pop()
        seen += 1
        if seen > MAX_TREE_NODES or depth > MAX_NESTING:
            raise _DraftRefusal('D_TREE_BOUND')
        kind = type(item)
        if item is None or kind is bool:
            continue
        if kind is int:
            if not MIN_INTEGER <= item <= MAX_INTEGER:
                raise _DraftRefusal('D_INTEGER_BOUND')
        elif kind is float:
            if not math.isfinite(item):
                raise _DraftRefusal('D_NONFINITE')
        elif kind is str:
            if len(item) > MAX_STRING_CHARACTERS:
                raise _DraftRefusal('D_STRING_BOUND')
            if any(0xD800 <= ord(character) <= 0xDFFF for character in item):
                raise _DraftRefusal('D_UNICODE_SCALAR')
        elif kind is list:
            if len(item) > MAX_ARRAY_ITEMS:
                raise _DraftRefusal('D_ARRAY_BOUND')
            pending.extend((child, depth+1) for child in item)
        elif kind is dict:
            if len(item) > MAX_OBJECT_KEYS:
                raise _DraftRefusal('D_OBJECT_BOUND')
            for key, child in item.items():
                if type(key) is not str:
                    raise _DraftRefusal('D_KEY_TYPE')
                pending.append((key, depth+1))
                pending.append((child, depth+1))
        else:
            raise _DraftRefusal('D_PRIMITIVE_DOMAIN')


def _fields_for(record_kind):
    if type(record_kind) is not str:
        raise _DraftRefusal('D_RECORD_KIND')
    for profile, fields in PROFILE_FIELDS:
        if record_kind == 'REQUIREMENTS-ONLY/'+profile:
            return profile, fields
    raise _DraftRefusal('D_RECORD_KIND')


def _field_domain(name, domain, value):
    if domain in UNRESOLVED_DOMAINS:
        if domain == 'closed-name-to-hash-map' and type(value) is not dict:
            raise _DraftRefusal('D_FIELD_TYPE')
        if domain == 'bounded-evidence-reference-list' and type(value) is not list:
            raise _DraftRefusal('D_FIELD_TYPE')
        # Do not invent reference-record shapes, path bounds or source key sets.
        # Even empty reference lists/maps do not establish complete history.
        return ('UNSUPPORTED/'+domain,)
    if domain == 'exact-string':
        if type(value) is not str:
            raise _DraftRefusal('D_FIELD_TYPE')
    elif domain == 'exact-integer':
        if type(value) is not int:
            raise _DraftRefusal('D_FIELD_TYPE')
        expected = 1 if name == 'revision' else 0
        if value != expected:
            raise _DraftRefusal('D_REVISION_OR_AUTHORITY')
    elif domain == 'canonical-hash-or-null':
        if value is None:
            return ('UNAVAILABLE_DOCUMENTARY_DIGEST',)
        if type(value) is not str or len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
            raise _DraftRefusal('D_DIGEST_SPELLING')
    elif domain == 'exact-false':
        if value is not False:
            raise _DraftRefusal('D_PERMISSION_FIELD')
    elif domain == 'exact-null':
        if value is not None:
            raise _DraftRefusal('D_ACTIVE_MATERIAL_FIELD')
    elif domain == 'evidence-status-label':
        if type(value) is not str or value not in EVIDENCE_STATUSES:
            raise _DraftRefusal('D_EVIDENCE_LABEL')
        if value in ('UNKNOWN', 'UNAVAILABLE'):
            return ('UNKNOWN_OR_UNAVAILABLE_EVIDENCE',)
    elif domain == 'finite-exact-int-or-float-or-null':
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise _DraftRefusal('D_FIELD_TYPE')
        # A valid supplied number is still not a bound process-owned clock.
        return ('UNBOUND_CLOCK_EVIDENCE',)
    elif domain == 'outcome-label':
        if type(value) is not str or value not in OUTCOME_CLASSES:
            raise _DraftRefusal('D_OUTCOME_LABEL')
    elif domain == 'three-exact-false-fields':
        if (type(value) is not dict or set(value) != {'real_execution_admitted', 'research_admitted', 'training_admitted'}
                or any(flag is not False for flag in value.values())):
            raise _DraftRefusal('D_ADMISSION_FIELD')
    else:
        return ('UNIMPLEMENTED_FIELD_DESCRIPTOR',)
    return ()


def inspect_documentary(payload):
    """Draft inspection of one canonical UTF-8 JSON record; never admit work.

    The supported entry accepts exact bytes only, so foreign Python objects
    are refused before len/iteration/decoder hooks. JSON hooks receive stdlib
    primitive values. Unexpected errors/BaseException propagate unchanged.
    No physical privacy, traceback lifetime or secure-erasure claim is made.
    """
    if type(payload) is not bytes:
        return _receipt('DRAFT_REFUSED', 'D_INPUT_TYPE')
    if not 0 < len(payload) <= MAX_PAYLOAD_BYTES:
        return _receipt('DRAFT_REFUSED', 'D_PAYLOAD_BOUND')
    document = None
    text = None
    try:
        text = payload.decode('utf-8')
        _scan_nesting(text)
        document = json.loads(text, object_pairs_hook=_pairs, parse_int=_integer,
                              parse_float=_floating, parse_constant=_constant)
        _tree_shape(document)
        canonical = (json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                                allow_nan=False)+'\n').encode('utf-8')
        if canonical != payload:
            raise _DraftRefusal('D_NONCANONICAL')
        if type(document) is not dict:
            raise _DraftRefusal('D_RECORD_TYPE')
        profile, fields = _fields_for(document.get('record_kind'))
        if set(document) != {name for name, domain in fields}:
            raise _DraftRefusal('D_RECORD_KEYS')
        unresolved = []
        for name, domain in fields:
            unresolved.extend(_field_domain(name, domain, document[name]))
        evidence_status = document['evidence_status']
        if unresolved:
            return _receipt('DRAFT_UNSUPPORTED_DOMAIN', 'D_UNRESOLVED', profile=profile,
                            evidence_status=evidence_status, unresolved=sorted(set(unresolved)), outer_shape_checked=True)
        return _receipt('DRAFT_STRUCTURE_ONLY', 'D_NONE', profile=profile,
                        evidence_status=evidence_status, outer_shape_checked=True)
    except UnicodeError:
        return _receipt('DRAFT_REFUSED', 'D_UTF8')
    except json.JSONDecodeError:
        return _receipt('DRAFT_REFUSED', 'D_JSON_SYNTAX')
    except _DraftRefusal as refusal:
        return _receipt('DRAFT_REFUSED', refusal.code)
    finally:
        # Release these local references only; caller/traceback/parser lifetime
        # and allocator behavior are not established by these assignments.
        document = None
        text = None

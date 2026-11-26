"""UNEXECUTED FAKE-only admission design for local proposal review.

No imports, real port, executor, entry point, grant writer or gate-binding API.
Every result fixes actual admission FALSE and real attempts0. Synthetic labels
cannot attest actual authority, ownership, clocks, caps or target behavior.
This module is outside the unchanged worker/upload graph. Do not run until the
separate local-validation request is explicitly approved.
"""

FAKE_KIND = 'FAKE-DIAGNOSTIC-EVIDENCE-REQUEST-v1'
FAKE_STATE_KIND = 'FAKE-DIAGNOSTIC-STATE-v1'
FAKE_LIMITS = {
    'objective_seconds': 120, 'whole_unit_seconds': 720, 'setup_seconds': 180,
    'proof_seconds': 60, 'work_soft_seconds': 2280, 'work_hard_seconds': 2400,
    'final_seconds': 120, 'collection_seconds': 600, 'connection_seconds': 3000,
    'RSS_bytes': 4294967296, 'allocator_bytes': 7516192768,
}
FAKE_ROLES = ('proof', 'original-unit', 'candidate-unit', 'candidate-replay')
CONSTRAINTS = ('science_unchanged', 'guards_unchanged', 'ownership_unchanged',
               'hashes_unchanged', 'durability_unchanged', 'fresh_roles')
BOOLEANS = CONSTRAINTS + ('authority_explicit', 'source_reviewed', 'history_preserved',
    'quiescence_confirmed', 'bypass_requested', 'training_requested', 'real_effect_requested', 'fit_qualified')
ADMIT_KEYS = {'kind', 'op', 'grant', 'source', 'control', 'owner', 'clock_domain', 'now', 'deadline', 'limits'} | set(BOOLEANS)
FINISH_KEYS = {'kind', 'op', 'owner', 'outcome', 'roles', 'parent_complete', 'owned_exit_confirmed'}
STATE_KEYS = {'kind', 'consumed', 'active_grant', 'active_owner', 'status'}


def _reply(code, state):
    return {'kind': 'FAKE-DIAGNOSTIC-POLICY-RESULT-v1', 'code': code, 'state': state,
            'real_execution_admitted': False, 'research_admitted': False,
            'training_admitted': False, 'maximum_real_attempts': 0, 'execution_binding': None}


def _state_valid(state):
    if type(state) is not dict or any(type(key) is not str for key in state) or set(state) != STATE_KEYS:
        return False
    if state['kind'] != FAKE_STATE_KIND or type(state['consumed']) is not list or len(state['consumed']) > 1:
        return False
    if any(type(value) is not str or not value.startswith('FAKE-') for value in state['consumed']):
        return False
    if type(state['status']) is not str or state['status'] not in ('empty', 'active', 'complete', 'incomplete', 'held'):
        return False
    if state['active_grant'] is None:
        if state['active_owner'] is not None or state['status'] in ('active', 'held'):
            return False
        return (state['status'] == 'empty') == (not state['consumed'])
    return (type(state['active_grant']) is str and state['active_grant'] in state['consumed']
            and type(state['active_owner']) is str and state['active_owner'].startswith('FAKE-')
            and state['status'] in ('active', 'held'))


def evaluate_fake_step(state, event):
    if not _state_valid(state):
        return _reply('DENY_SCHEMA', None)
    out = dict(state); out['consumed'] = list(state['consumed'])
    if type(event) is not dict or any(type(key) is not str for key in event):
        return _reply('DENY_SCHEMA', out)
    if type(event.get('kind')) is not str:
        return _reply('DENY_SCHEMA', out)
    if event['kind'] != FAKE_KIND:
        return _reply('DENY_FAKE_ONLY', out)
    if event.get('op') == 'finish':
        if (set(event) != FINISH_KEYS or type(event['owner']) is not str or type(event['outcome']) is not str
                or event['outcome'] not in ('success', 'failure') or type(event['roles']) is not list
                or len(event['roles']) > 4 or any(type(role) is not str or role not in FAKE_ROLES for role in event['roles'])
                or len(set(event['roles'])) != len(event['roles'])
                or type(event['parent_complete']) is not bool or type(event['owned_exit_confirmed']) is not bool):
            return _reply('DENY_SCHEMA', out)
        if out['active_grant'] is None:
            return _reply('DENY_STATE', out)
        if event['owner'] != out['active_owner']:
            return _reply('DENY_OWNER', out)
        if not event['owned_exit_confirmed']:
            out['status'] = 'held'
            return _reply('FAKE_EVIDENCE_HELD', out)
        out['active_grant'] = out['active_owner'] = None
        if event['outcome'] == 'success' and event['roles'] == list(FAKE_ROLES) and event['parent_complete']:
            out['status'] = 'complete'
            return _reply('FAKE_EVIDENCE_RECORDED', out)
        out['status'] = 'incomplete'
        return _reply('FAKE_EVIDENCE_INCOMPLETE', out)
    if set(event) != ADMIT_KEYS or event.get('op') != 'admit':
        return _reply('DENY_SCHEMA', out)
    if (any(type(event[key]) is not bool for key in BOOLEANS)
            or any(type(event[key]) is not str for key in ('grant', 'source', 'control', 'owner', 'clock_domain'))
            or type(event['now']) is not int or type(event['deadline']) is not int
            or type(event['limits']) is not dict or any(type(key) is not str for key in event['limits'])
            or set(event['limits']) != set(FAKE_LIMITS) or any(type(value) is not int for value in event['limits'].values())):
        return _reply('DENY_SCHEMA', out)
    if event['real_effect_requested']:
        return _reply('DENY_REAL_EFFECT', out)
    if event['bypass_requested']:
        return _reply('DENY_BYPASS', out)
    if event['training_requested']:
        return _reply('DENY_TRAINING_PROMOTION', out)
    if not event['authority_explicit']:
        return _reply('DENY_AUTHORITY', out)
    if not event['history_preserved']:
        return _reply('DENY_HISTORY', out)
    if not event['quiescence_confirmed']:
        return _reply('DENY_QUIESCENCE', out)
    if not event['source_reviewed'] or event['source'] != 'FAKE-SOURCE-v1' or event['control'] != 'FAKE-CONTROL-v1':
        return _reply('DENY_SOURCE', out)
    if event['owner'] != 'FAKE-OWNER-1' or not event['grant'].startswith('FAKE-'):
        return _reply('DENY_OWNER', out)
    if event['clock_domain'] != 'FAKE-CLOCK-1':
        return _reply('DENY_CLOCK', out)
    if not 0 <= event['now'] < event['deadline'] <= FAKE_LIMITS['connection_seconds']:
        return _reply('DENY_DEADLINE', out)
    if event['limits'] != FAKE_LIMITS or not all(event[key] for key in CONSTRAINTS):
        return _reply('DENY_CONSTRAINT', out)
    if out['active_grant'] is not None:
        return _reply('DENY_OWNED_STATE', out)
    if event['grant'] in out['consumed']:
        return _reply('DENY_CONSUMED', out)
    if out['consumed']:
        return _reply('DENY_ATTEMPT_CEILING', out)
    out['consumed'].append(event['grant'])
    out['active_grant'] = event['grant']; out['active_owner'] = event['owner']; out['status'] = 'active'
    return _reply('FAKE_EVIDENCE_ADMITTED', out)

"""Strict stdlib local receipt/replay checks. Metadata never proves native execution."""
import hashlib
import json
import math
import re

IDENTITY = 'local-whole-pgd-input-gradient-workload-v1'
KINDS = ('analytic-local-control-v1', 'tiny-frozen-forward-preserving-eot8-v1')


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _seeds(video, audits):
    def seed(name, draw=0):
        raw = json.dumps(['evaluation-v1', 20261028, 'research_candidate_eot8_v1_'+name,
                          video, 0, draw], separators=(',', ':'), ensure_ascii=True).encode('ascii')
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % 2**63
    result = {'attack': [seed('attack', i) for i in range(8)],
              'score': [seed('main', i) for i in range(4)], 'start': seed('start')}
    result.update({f'audit_{j}': [seed(f'audit_{j}', i) for i in range(4)] for j in range(audits)})
    return result


def validate_receipt(row):
    _require(type(row) is dict and row.get('kind') == IDENTITY and row.get('status') == 'complete',
             'Wrong or incomplete local workload receipt.')
    _require(row.get('route') in KINDS and row.get('execution_binding') is None
             and row.get('research_admitted') is False and row.get('GPU_performance_qualified') is False
             and row.get('restored') is True and row.get('overhead_subtracted') is False,
             'Local control cannot become target/research/performance admission.')
    m = row['metrics']; settings = m['settings']; method = m['method']
    _require(method in ('primary', 'convergence') and settings['method'] == method,
             'Only complete declared primary/convergence units.')
    n = {'primary': 20, 'convergence': 50}[method]
    _require((settings['steps'], settings['restarts'], settings['attack_masks'], settings['scoring_masks'])
             == (n, 2, 8, 4) and settings['epsilon_255'] == 4 and settings['step_size_255'] in (.5, 1, 2)
             and settings['profile'] == 'research-primary-eot8-v1', 'Changed strength or profile.')
    audits = len(m['scoring_audits'])
    _require(audits in (0, 2) and m['video_id'].startswith('local-workload-control/'), 'Nonlocal control.')
    _require(m['seeds'] == _seeds(m['video_id'], audits), 'Changed starts/mask identities.')
    _require(_hash(row['adversarial']) and len(row['steps']) == n*2, 'Incomplete rollout.')
    for index, step in enumerate(row['steps']):
        _require(step['index'] == index and step['consider_before_backward'] is True
                 and _hash(step['pixels']) and _hash(step['gradient']) and len(step['losses']) == 1
                 and step['route']['kind'] == row['route'], 'Broken rollout token/route joins.')
        proof = step['route']
        if row['route'] == KINDS[0]:
            _require(proof.get('analytic_autograd_only') is True and proof.get('native_ownership_proven') is False,
                     'Analytic control cannot assert native ownership.')
        else:
            _require(proof.get('native_root_hook_removed') is True and proof.get('forward_calls') == 64
                     and proof.get('initial_predictor_VJPs') == 32 and proof.get('native_refusals') == 0,
                     'Missing frozen per-objective gate observations.')
    scores = row['scores']
    _require([s['bank'] for s in scores] == [1, 0, 0, 0, 0, 1] + [j for j in range(2, audits+2) for _ in range(2)],
             'Changed clean/baseline/final/independent scoring order.')
    for score in scores:
        _require(score['grad_enabled'] is False and score['route_active'] is False
                 and score['draws'] == (8 if score['bank'] == 0 else 4)
                 and _hash(score['pixels']) and len(score['losses']) == 1, 'Non-original scoring path.')
    _require(scores[0]['pixels'] == scores[1]['pixels'] == scores[2]['pixels']
             and scores[5]['pixels'] == row['adversarial'], 'Wrong clean or selected pixels.')
    _require(scores[0]['losses'][0] == m['clean_loss'] and scores[1]['losses'][0] == m['attack_clean_loss']
             and scores[5]['losses'][0] == m['adversarial_loss'], 'Scoring metrics do not join.')
    for j in range(audits):
        a, b = scores[6+2*j:8+2*j]
        _require(a['pixels'] == scores[0]['pixels'] and b['pixels'] == row['adversarial']
                 and m['scoring_audits'][j] == {'clean_loss': a['losses'][0], 'adversarial_loss': b['losses'][0]},
                 'Audit scoring changed the selected result.')
    values = [m[k] for k in ('clean_loss', 'adversarial_loss', 'attack_clean_loss', 'attack_loss', 'max_pixel_delta', 'seconds')]
    values += [s['losses'][0] for s in scores] + [s['losses'][0] for s in row['steps']]
    values += [s['context_seconds_including_observations'] for s in row['steps']]
    values += [row['full_unit_seconds_including_guards_journal_and_observations']]
    _require(all(type(v) in (float, int) and math.isfinite(v) for v in values), 'Nonfinite receipt values.')
    _require(0 <= m['max_pixel_delta'] <= 4/255+1e-7 and m['seconds'] >= 0
             and all(s['context_seconds_including_observations'] >= 0 for s in row['steps'])
             and row['full_unit_seconds_including_guards_journal_and_observations'] >= m['seconds'],
             'Projection/timing observations are invalid.')
    return {'kind': 'local-workload-metadata-validation-v1', 'complete_gradients': n*2,
            'original_no_grad_calls': len(scores), 'route': row['route'],
            'native_execution_authenticated_by_dictionary': False, 'research_admitted': False}


def compare_replay(first, second):
    validate_receipt(first); validate_receipt(second)
    def logical(row):
        m = {k: v for k, v in row['metrics'].items() if k != 'seconds'}
        steps = [{k: v for k, v in s.items() if k != 'context_seconds_including_observations'} for s in row['steps']]
        return {'route': row['route'], 'metrics': m, 'steps': steps,
                'scores': row['scores'], 'adversarial': row['adversarial']}
    _require(logical(first) == logical(second), 'Fresh local rollout replay changed logical values.')
    return {'kind': 'local-workload-replay-comparison-v1', 'logical_bytes_exact': True,
            'route': first['route'], 'fresh_process_authenticated_by_dictionary': False,
            'target_rollout_replay_qualified': False, 'research_admitted': False}

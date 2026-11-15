"""New execution records; historical references retain their consumed control."""
import copy
import re
from types import FunctionType
from models import predictor_forward_preserving_records_v1 as frozen
from models import predictor_replay_capsule_v1 as capsule

WORKER_KIND = 'forward-preserving-missing-candidate-replay-worker-v1'
MANIFEST_SHA = '8107d13d5c663102f6e5e85dc19a5f6dea52bb12054a57d5957a3d12720a67fd'
REFERENCES = {
    'same_replay_original': {'source_index': 2,
        'worker_sha256': '5a02933dc8a3df8854cecce1339d47e0258164c3ce1622dbb841ef2d176ae2e6',
        'completion_sha256': 'd09b715d89b325611016ced05bfc4ad0b48fba212578dbd400cb2c2da39e0821'},
    'fresh_process_candidate_reference': {'source_index': 1,
        'worker_sha256': '338f696f2fe232a7e1a77fa139be2984387ad39f7155e5aa82089e72d46a07de',
        'completion_sha256': 'cce83120a623817321a12371e68ec9cab7ac4358abace10a6d413954ffcf1177'},
}
atomic, LIMITS, read, sha_file = frozen.atomic, frozen.LIMITS, frozen.read, frozen.sha_file


def binding(control_sha256):
    if (not isinstance(control_sha256, str) or not re.fullmatch('[0-9a-f]{64}', control_sha256)
            or control_sha256 == capsule.SOURCE_CONTROL):
        raise ValueError('Distinct new execution control required; historical grant is consumed.')
    return {'kind': 'missing-candidate-replay-reference-binding-v1',
            'new_execution_control_sha256': control_sha256, 'capsule_manifest_sha256': MANIFEST_SHA,
            'historical_reference_control_sha256': capsule.SOURCE_CONTROL, 'references': copy.deepcopy(REFERENCES),
            'new_worker': {'index': 3, 'mode': 'candidate', 'replay': 1},
            'execution_permitted': False, 'authorization_required': True,
            'old_grant_or_partial_replay_reused': False}


_base_identity = FunctionType(frozen.identity.__code__, dict(frozen.identity.__globals__, WORKER_KIND=WORKER_KIND))


def identity(state):
    key = _base_identity(state)
    if (key['index'] != 3 or key['mode'] != 'candidate' or key['replay'] != 1 or key['local_tiny'] is not False
            or state.get('reference_binding') != binding(key['control_sha256'])):
        raise ValueError('Only the new target candidate replay with exact historical lineage is allowed.')
    return key


# Original numerical, native-ownership and VJP predicates; private identity only.
_native_scope = FunctionType(frozen._native_scope.__code__, dict(frozen._native_scope.__globals__, identity=identity))
check_scope = FunctionType(frozen.check_scope.__code__, dict(frozen.check_scope.__globals__, _native_scope=_native_scope))
write_final = FunctionType(frozen.write_final.__code__, dict(frozen.write_final.__globals__, identity=identity))
_load = FunctionType(frozen.load.__code__, dict(frozen.load.__globals__, identity=identity, check_scope=check_scope))


def load(folder, *, control_sha256, index=3, live=True):
    if index != 3 or live is not True:
        raise ValueError('Only one full-size CUDA candidate replay may be admitted.')
    binding(control_sha256)
    return _load(folder, control_sha256=control_sha256, index=3, live=True)

"""Separate association-v2 continuation; no university authorization at preparation."""
from models import predictor_sdpa_continuation_contract_v1 as prior
KIND='native-SDPA-component-only-collection-v2'
AUTH_KIND='native-SDPA-component-only-university-exception-v2'
PREFIX='vjepa-SDPA-components-v2-20261108-'
CONTRACT_SHA='71410f850db74e83fd46a7f623dccdf33f1fdebb500becaf8a5bb53cd51617c1'
CAPSULE_SHA=prior.CAPSULE_SHA
MANIFEST_SHA=prior.MANIFEST_SHA
PARENT_CONTROL_SHA=prior.PARENT_CONTROL_SHA
PRIOR_CLAIM_SHA='9933b5e9f50c5c3ee571b39bea8d2afd94724ea9adf7d772809a70aaaae7f1f0'
NATIVE_WORKER_SHA=prior.NATIVE_WORKER_SHA
BOUNDARY_SHA=prior.BOUNDARY_SHA
MiB=2**20
GiB=2**30
RESOURCE=dict(prior.RESOURCE)
ROLES=prior.ROLES
FROZEN_BOOT=(*prior.FROZEN_BOOT,'models/predictor_sdpa_continuation_contract_v1.py')
COMPONENT_SOURCES=('models/predictor_sdpa_component_v3.py', 'models/predictor_sdpa_component_records_v1.py', 'models/predictor_sdpa_component_events_v1.py', 'models/predictor_sdpa_capture_record_v1.py', 'models/attention_saved_evidence.py', 'models/execution.py', 'models/training.py', 'models/predictor_eot8_capture_v1.py', 'models/predictor_eot8_adapter_v1.py', 'scripts/check_predictor_sdpa_components_v3.py', 'tests/test_predictor_sdpa_component_integration_v3.py', 'models/predictor_sdpa_component_startup_v2.py', 'models/predictor_sdpa_component_target_v3.py', 'models/predictor_sdpa_component_events_v2.py')
REMOTE_NAMES=tuple(dict.fromkeys(('models/__init__.py',*FROZEN_BOOT,*COMPONENT_SOURCES,
 'models/predictor_sdpa_continuation_contract_v2.py','models/predictor_sdpa_continuation_storage_v2.py',
 'models/predictor_sdpa_continuation_operator_v2.py','scripts/capture_predictor_sdpa_continuation_v2.py',
 'configs/predictor_sdpa_continuation_v2.json')))
LOCAL_EXTRA=('models/predictor_sdpa_continuation_client_v2.py','scripts/prepare_predictor_sdpa_continuation_v2.py',
 'scripts/run_predictor_sdpa_continuation_v2.py','tests/test_predictor_sdpa_continuation_v2.py')

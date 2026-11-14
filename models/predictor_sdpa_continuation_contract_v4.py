"""Separate capture-before-validation handoff; initially unarmed."""
from models import predictor_sdpa_continuation_contract_v3 as prior
KIND='native-SDPA-component-only-collection-v4'
AUTH_KIND='native-SDPA-component-only-university-exception-v4'
PREFIX='vjepa-SDPA-components-v4-20261113-'
CONTRACT_SHA='60d72f5cf25b1119433e751b3621d78c46a27145dc720c22a37b453a512769b2'
CAPSULE_SHA=prior.CAPSULE_SHA
MANIFEST_SHA=prior.MANIFEST_SHA
PARENT_CONTROL_SHA=prior.PARENT_CONTROL_SHA
PRIOR_CLAIM_SHA='95a15a08e35721a7840e26915544b5599d3656635b2527162b3ec595236770b0'
NATIVE_WORKER_SHA=prior.NATIVE_WORKER_SHA
BOUNDARY_SHA=prior.BOUNDARY_SHA
MiB=2**20
GiB=2**30
RESOURCE=dict(prior.RESOURCE)
ROLES=prior.ROLES
FROZEN_BOOT=(*prior.FROZEN_BOOT,'models/predictor_sdpa_continuation_contract_v3.py')
COMPONENT_SOURCES=('models/predictor_sdpa_component_v5.py', 'models/predictor_sdpa_component_records_v1.py', 'models/predictor_sdpa_component_events_v1.py', 'models/predictor_sdpa_capture_record_v1.py', 'models/attention_saved_evidence.py', 'models/execution.py', 'models/training.py', 'models/predictor_eot8_capture_v1.py', 'models/predictor_eot8_adapter_v1.py', 'scripts/check_predictor_sdpa_components_v5.py', 'tests/test_predictor_sdpa_component_integration_v5.py', 'models/predictor_sdpa_component_startup_v2.py', 'models/predictor_sdpa_component_target_v5.py', 'models/predictor_sdpa_component_events_v2.py', 'models/predictor_sdpa_launch_evidence_v1.py', 'models/predictor_sdpa_raw_capture_v1.py')
REMOTE_NAMES=tuple(dict.fromkeys(('models/__init__.py',*FROZEN_BOOT,*COMPONENT_SOURCES,
 'models/predictor_sdpa_continuation_contract_v4.py','models/predictor_sdpa_continuation_storage_v4.py',
 'models/predictor_sdpa_continuation_operator_v4.py','models/predictor_sdpa_continuation_receipts_v2.py','scripts/capture_predictor_sdpa_continuation_v4.py',
 'configs/predictor_sdpa_continuation_v4.json')))
LOCAL_EXTRA=('models/predictor_sdpa_continuation_client_v4.py','scripts/prepare_predictor_sdpa_continuation_v4.py',
 'scripts/run_predictor_sdpa_continuation_v4.py','tests/test_predictor_sdpa_continuation_v4.py')

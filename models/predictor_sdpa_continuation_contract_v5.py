"""Separate capture-before-validation handoff; initially unarmed."""
from models import predictor_sdpa_continuation_contract_v4 as prior
KIND='native-SDPA-component-only-collection-v5'
AUTH_KIND='native-SDPA-component-only-university-exception-v5'
PREFIX='vjepa-SDPA-components-v5-20261113-'
CONTRACT_SHA='b00ff580bb2b7462de6d153681a9864b874703f1f1401637cc10942549a2adc8'
CAPSULE_SHA=prior.CAPSULE_SHA
MANIFEST_SHA=prior.MANIFEST_SHA
PARENT_CONTROL_SHA=prior.PARENT_CONTROL_SHA
PRIOR_CLAIM_SHA='3829d1b634b32053f73560610f2e78d5a835e7193eaea885b55211e6c2536c32'
NATIVE_WORKER_SHA=prior.NATIVE_WORKER_SHA
BOUNDARY_SHA=prior.BOUNDARY_SHA
MiB=2**20
GiB=2**30
RESOURCE=dict(prior.RESOURCE)
ROLES=prior.ROLES
FROZEN_BOOT=(*prior.FROZEN_BOOT,'models/predictor_sdpa_continuation_contract_v4.py')
COMPONENT_SOURCES=('models/predictor_sdpa_component_v6.py', 'models/predictor_sdpa_component_records_v1.py', 'models/predictor_sdpa_component_events_v1.py', 'models/predictor_sdpa_capture_record_v1.py', 'models/attention_saved_evidence.py', 'models/execution.py', 'models/training.py', 'models/predictor_eot8_capture_v1.py', 'models/predictor_eot8_adapter_v1.py', 'scripts/check_predictor_sdpa_components_v6.py', 'tests/test_predictor_sdpa_component_integration_v6.py', 'models/predictor_sdpa_component_startup_v2.py', 'models/predictor_sdpa_component_target_v6.py', 'models/predictor_sdpa_component_events_v2.py', 'models/predictor_sdpa_launch_evidence_v1.py', 'models/predictor_sdpa_raw_capture_v1.py', 'models/predictor_sdpa_launch_evidence_v2.py')
REMOTE_NAMES=tuple(dict.fromkeys(('models/__init__.py',*FROZEN_BOOT,*COMPONENT_SOURCES,
 'models/predictor_sdpa_continuation_contract_v5.py','models/predictor_sdpa_continuation_storage_v5.py',
 'models/predictor_sdpa_continuation_operator_v5.py','models/predictor_sdpa_continuation_receipts_v3.py','scripts/capture_predictor_sdpa_continuation_v5.py',
 'configs/predictor_sdpa_continuation_v5.json')))
LOCAL_EXTRA=('models/predictor_sdpa_continuation_client_v5.py','scripts/prepare_predictor_sdpa_continuation_v5.py',
 'scripts/run_predictor_sdpa_continuation_v5.py','tests/test_predictor_sdpa_continuation_v5.py')

METADATA_POLICY='fixed-SDPA-profiler-overhead-signature-v1'

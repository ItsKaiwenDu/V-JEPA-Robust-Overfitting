"""Separate numerical-policy-bound handoff; initially unarmed."""
from models import predictor_sdpa_continuation_contract_v5 as prior
KIND='native-SDPA-component-only-collection-v6'
AUTH_KIND='native-SDPA-component-only-university-exception-v6'
PREFIX='vjepa-SDPA-components-v6-20261113-'
CONTRACT_SHA='d63e59d1570648e53d97e7db977a575a0e4da4044504ebc9a6d0a689a0d497e9'
CAPSULE_SHA=prior.CAPSULE_SHA
MANIFEST_SHA=prior.MANIFEST_SHA
PARENT_CONTROL_SHA=prior.PARENT_CONTROL_SHA
PRIOR_CLAIM_SHA='8e5614d40fd19cf2fd35861e19a86eb90251cd113ef23b144db9a0682d070662'
NATIVE_WORKER_SHA=prior.NATIVE_WORKER_SHA
BOUNDARY_SHA=prior.BOUNDARY_SHA
MiB=2**20
GiB=2**30
RESOURCE=dict(prior.RESOURCE)
ROLES=prior.ROLES
FROZEN_BOOT=(*prior.FROZEN_BOOT,'models/predictor_sdpa_continuation_contract_v5.py')
COMPONENT_SOURCES=('models/predictor_sdpa_component_v7.py', 'models/predictor_sdpa_component_records_v2.py', 'models/predictor_sdpa_component_events_v1.py', 'models/predictor_sdpa_capture_record_v1.py', 'models/attention_saved_evidence.py', 'models/execution.py', 'models/training.py', 'models/predictor_eot8_capture_v1.py', 'models/predictor_eot8_adapter_v1.py', 'scripts/check_predictor_sdpa_components_v7.py', 'tests/test_predictor_sdpa_component_integration_v7.py', 'models/predictor_sdpa_component_startup_v2.py', 'models/predictor_sdpa_component_target_v7.py', 'models/predictor_sdpa_component_events_v2.py', 'models/predictor_sdpa_launch_evidence_v1.py', 'models/predictor_sdpa_raw_capture_v1.py', 'models/predictor_sdpa_launch_evidence_v2.py', 'models/predictor_sdpa_numerical_receipts_v1.py')
REMOTE_NAMES=tuple(dict.fromkeys(('models/__init__.py',*FROZEN_BOOT,*COMPONENT_SOURCES,
 'models/predictor_sdpa_continuation_contract_v6.py','models/predictor_sdpa_continuation_storage_v6.py',
 'models/predictor_sdpa_continuation_operator_v6.py','models/predictor_sdpa_continuation_receipts_v4.py','scripts/capture_predictor_sdpa_continuation_v6.py',
 'configs/predictor_sdpa_continuation_v6.json')))
LOCAL_EXTRA=('models/predictor_sdpa_continuation_client_v6.py','scripts/prepare_predictor_sdpa_continuation_v6.py',
 'scripts/run_predictor_sdpa_continuation_v6.py','tests/test_predictor_sdpa_continuation_v6.py')
METADATA_POLICY='fixed-SDPA-profiler-overhead-signature-v1'
NUMERICAL_POLICY='fixed-SDPA-numerical-receipt-bookkeeping-v1'
REFERENCE_KIND='native-SDPA-fixed-FP64-reference-v2'
WORKER_KIND='native-SDPA-fixed-input-component-v2'

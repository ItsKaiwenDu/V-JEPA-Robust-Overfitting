"""Separate raw-correlation component handoff; initially unarmed."""
from models import predictor_sdpa_continuation_contract_v2 as prior
KIND='native-SDPA-component-only-collection-v3'
AUTH_KIND='native-SDPA-component-only-university-exception-v3'
PREFIX='vjepa-SDPA-components-v3-20261113-'
CONTRACT_SHA='4d2c6ab42b0c1c6ded9be9e3cae69620ac2c060fef816e0933475e179ac239ba'
CAPSULE_SHA=prior.CAPSULE_SHA
MANIFEST_SHA=prior.MANIFEST_SHA
PARENT_CONTROL_SHA=prior.PARENT_CONTROL_SHA
PRIOR_CLAIM_SHA='6c46ce1fb7ab4a3f7dbe138866b9d45f5046f0c77b26bd8e689d7b8c3dcaa07a'
NATIVE_WORKER_SHA=prior.NATIVE_WORKER_SHA
BOUNDARY_SHA=prior.BOUNDARY_SHA
MiB=2**20
GiB=2**30
RESOURCE=dict(prior.RESOURCE)
ROLES=prior.ROLES
FROZEN_BOOT=(*prior.FROZEN_BOOT,'models/predictor_sdpa_continuation_contract_v2.py')
COMPONENT_SOURCES=('models/predictor_sdpa_component_v4.py', 'models/predictor_sdpa_component_records_v1.py', 'models/predictor_sdpa_component_events_v1.py', 'models/predictor_sdpa_capture_record_v1.py', 'models/attention_saved_evidence.py', 'models/execution.py', 'models/training.py', 'models/predictor_eot8_capture_v1.py', 'models/predictor_eot8_adapter_v1.py', 'scripts/check_predictor_sdpa_components_v4.py', 'tests/test_predictor_sdpa_component_integration_v4.py', 'models/predictor_sdpa_component_startup_v2.py', 'models/predictor_sdpa_component_target_v4.py', 'models/predictor_sdpa_component_events_v2.py', 'models/predictor_sdpa_launch_evidence_v1.py')
REMOTE_NAMES=tuple(dict.fromkeys(('models/__init__.py',*FROZEN_BOOT,*COMPONENT_SOURCES,
 'models/predictor_sdpa_continuation_contract_v3.py','models/predictor_sdpa_continuation_storage_v3.py',
 'models/predictor_sdpa_continuation_operator_v3.py','models/predictor_sdpa_continuation_receipts_v1.py','scripts/capture_predictor_sdpa_continuation_v3.py',
 'configs/predictor_sdpa_continuation_v3.json')))
LOCAL_EXTRA=('models/predictor_sdpa_continuation_client_v3.py','scripts/prepare_predictor_sdpa_continuation_v3.py',
 'scripts/run_predictor_sdpa_continuation_v3.py','tests/test_predictor_sdpa_continuation_v3.py')

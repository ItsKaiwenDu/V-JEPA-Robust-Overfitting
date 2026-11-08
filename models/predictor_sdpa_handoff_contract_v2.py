"""Separate mask-policy replacement scope; no target authority at preparation."""
from models import predictor_sdpa_handoff_contract_v1 as prior
KIND='native-fixed-SDPA-storage-collection-v2'
AUTH_KIND='native-fixed-SDPA-university-exception-v2'
PREFIX='vjepa-native-sdpa-v2-20261107-'
CONTRACT_SHA='386e88a4fa8d3d6fb08ac331ae50694debd182592e65b79785d021233f333466'
PACKET_SHA=prior.PACKET_SHA
PRIOR_CLAIM_SHA='c4afa75ef3c1ccb126f2c819d00b04ff9ce81ac671bbe153a30403edd17f8b5f'
PRIOR_METADATA_CLAIM_SHA='e924af4e7bd8222578ad86bb64bfcca6e3ad293ffb4ab2cc4607124a15934ced'
SCIENTIFIC_PRIOR_CLAIM_SHA=prior.PRIOR_CLAIM_SHA
MiB=2**20
GiB=2**30
RESOURCE={**prior.RESOURCE,'mask_indices_file_bytes':256*1024}
ROLES=prior.ROLES
FROZEN_BOOT=prior.FROZEN_BOOT
COMPONENT_SOURCES=prior.COMPONENT_SOURCES
REMOTE_NAMES=(*prior.REMOTE_NAMES,'models/predictor_sdpa_storage_v2.py',
 'models/predictor_sdpa_handoff_contract_v2.py','models/predictor_sdpa_handoff_operator_v2.py',
 'scripts/capture_predictor_sdpa_handoff_v2.py','configs/predictor_sdpa_handoff_v2.json')
LOCAL_EXTRA=('models/predictor_sdpa_handoff_client_v2.py','scripts/prepare_predictor_sdpa_handoff_v2.py',
 'scripts/run_predictor_sdpa_handoff_v2.py','tests/test_predictor_sdpa_handoff_operations_v2.py',
 'tests/test_predictor_sdpa_storage_v2.py')

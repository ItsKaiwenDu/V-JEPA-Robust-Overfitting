"""Explicit, source-bound opt-in contract; no research or spending authorization."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess

PROFILE = 'research-primary-eot8-v1'
CANDIDATE_SHA256 = 'dc0dba86c92f027296678918b3b8fe1ed6795bace094acde0f0a22c027bf9304'
BASE_SEED = 20261028
ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'models/candidate_contract.py', 'models/candidate_evaluation.py',
    'models/candidate_evaluation_store.py', 'models/candidate_calibration.py',
    'models/candidate_schedule.py', 'models/candidate_evaluation_checkpoint.py',
    'models/candidate_features.py', 'models/candidate_benchmark.py',
    'scripts/run_candidate_benchmark.py', 'models/workflow_diagnostic.py',
    'scripts/run_workflow_pilot.py',
    'models/workflow_pilot.py', 'datasets/prepared.py', 'models/fixture_state.py',
    'scripts/prepare_candidate_benchmark.py', 'models/pilot_bundle.py',
    'models/workflow_preflight.py', 'models/training.py', 'models/checkpointing.py',
    'models/evaluation.py', 'models/evaluation_store.py', 'models/calibration.py',
    'models/schedule.py', 'models/evaluation_checkpoint.py', 'models/features.py',
    'models/study_artifacts.py', 'models/vjepa2.py', 'models/pilot.py',
    'models/runtime.py', 'models/diagnostics.py', 'models/pilot_measurement.py',
    'attacks/pgd.py', 'datasets/staged_files.py', 'datasets/staged_subset.py',
    'datasets/staged_preparation.py', 'datasets/video.py', 'models/execution.py',
    'models/native_patch.py', 'models/patch_embed.py', 'models/precision_patch.py',
    'models/gradient_attribution.py', 'models/native_diagnostic.py',
    'models/precision_diagnostic.py',
    'models/native_probe.py', 'models/cuda_profile.py', 'models/cuda_convolution.py',
    'configs/study.yaml', 'configs/budget_study_plan.yaml',
    'report/checks/2026-10-20-staged-study-projection.json',
    'report/checks/2026-10-19-runpod-5090-pilot.json',
    'report/checks/2026-10-19-runpod-5090-attacks.json',
    'scripts/estimate_staged_study.py',
    'data/manifests/ucf101-split1-v1/train.csv',
    'data/manifests/ucf101-split1-v1/validation.csv',
    'data/manifests/ucf101-split1-v1/test.csv',
    'data/manifests/ucf101-staged-20x5-v1/metadata.json',
    'data/manifests/ucf101-staged-20x5-v1/train.csv',
    'data/manifests/ucf101-staged-20x5-v1/train_monitor.csv',
    'data/manifests/ucf101-staged-20x5-v1/exploration_validation.csv',
    'data/manifests/ucf101-staged-20x5-v1/confirmation_validation.csv',
    'data/manifests/ucf101-staged-20x5-v1/test.csv',
)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def binding():
    candidate = json.loads((ROOT / 'configs/research_attack_candidate_v1.json').read_text())
    if canonical_sha(candidate) != CANDIDATE_SHA256:
        raise ValueError('Candidate proposal changed; version the opt-in contract first.')
    return {'schema_version': 1, 'profile': PROFILE, 'candidate_sha256': CANDIDATE_SHA256,
            'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}}


def cuda_identity():
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if len(lines) != 1:
        raise ValueError('Candidate runtime requires exactly one CUDA device.')
    name, uuid, driver = [s.strip() for s in lines[0].split(',')]
    return {'gpu': name, 'uuid': uuid, 'driver_version': driver}


def bind_provenance(provenance, *, cuda=False):
    value = copy.deepcopy(provenance)
    if 'candidate_eot8' in value and value['candidate_eot8'] != binding():
        raise ValueError('Cannot relabel a different candidate implementation.')
    value['candidate_eot8'] = binding()
    if cuda:
        value['candidate_cuda_identity'] = cuda_identity()
    return value


def verify_binding(value):
    if value != binding():
        raise ValueError('Candidate settings/source contract differs; legacy artifacts cannot be relabeled.')


def verify_provenance(value):
    if not isinstance(value, dict):
        raise ValueError('Candidate provenance must be a mapping.')
    verify_binding(value.get('candidate_eot8'))


def verify_runtime(runtime, provenance):
    verify_provenance(provenance)
    if runtime['device'] == 'cuda' and provenance.get('candidate_cuda_identity') != cuda_identity():
        raise ValueError('Candidate CUDA GPU/UUID/driver differs; artifacts are not portable to this runtime.')
    if runtime['device'] != 'cuda' and 'candidate_cuda_identity' in provenance:
        raise ValueError('CUDA candidate provenance cannot be relabeled as a CPU fixture.')


def declared_calibration_rows():
    candidate = json.loads((ROOT / 'configs/research_attack_candidate_v1.json').read_text())
    binding()
    return [{'video_id': r['video_id'], 'source_group_id': r['source_group_id']}
            for r in candidate['calibration']['rows']]

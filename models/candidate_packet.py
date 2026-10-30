"""Prepare an unarmed lower-cost benchmark packet from the verified frozen source."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from models.pilot_bundle import digest, make_archive, safe_name

BASE_MANIFEST_SHA256 = '788574998d9741e5e0fd8d5d77279bf4ee7509b952322e30f601dad472550cf4'
BINDING_SHA256 = 'f418050bb82c2423b45403c2bf12b17fd888200473f2d0d0a02313bdd3f85d72'
EXTRA = 'scripts/check_workflow.py'
EXTRA_SHA256 = 'd8f16fe58200fba9a34f1cb9892ddb48eecc836473b594ba6d2464b8c08c36c7'
CPU_TESTS = ['tests/test_candidate_evaluation.py', 'tests/test_candidate_schedule.py',
             'tests/test_candidate_benchmark.py']


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def declared_sources(path):
    tree = ast.parse(path.read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in n.targets))


def source_files(root, manifest):
    """Use the prior explicit allowlist, never newly discovered worktree tests."""
    root = Path(root).resolve()
    binding = manifest['candidate']['candidate_binding']
    if canonical(binding) != BINDING_SHA256 or len(binding['source_sha256']) != 61:
        raise ValueError('Candidate scientific binding changed.')
    original = manifest['archives']['source']['files']
    if len(original) != 75 or EXTRA in original:
        raise ValueError('Use the original 75-file source snapshot, preserving its history.')
    expected = {**original, EXTRA: {'sha256': EXTRA_SHA256, 'bytes': 8187}}
    files = {}
    for name, item in expected.items():
        safe_name(name); path = root / name
        if (path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file()
                or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']):
            raise ValueError('Pinned source is missing or changed: ' + name)
        files[name] = path
    for name, expected_sha in binding['source_sha256'].items():
        if name not in files or digest(files[name]) != expected_sha:
            raise ValueError('Candidate source closure changed: ' + name)
    declared = declared_sources(root / 'models/workflow_diagnostic.py')
    if len(declared) != 24 or not set(declared) <= set(files) or not set(CPU_TESTS) <= set(files):
        raise ValueError('Provenance or CPU-fixture dependency missing from the explicit allowlist.')
    if set(declared_sources(root / 'models/candidate_contract.py')) != set(binding['source_sha256']):
        raise ValueError('The candidate implementation source list changed.')
    return files


def launch_template(candidate):
    return {'kind': 'candidate_eot8_benchmark_launch_v1',
        'candidate_binding': deepcopy(candidate['candidate_binding']), 'authorized': False,
        'diagnostic_only': True, 'execution_policy': 'shared-fp64-native',
        'required_phases': deepcopy(candidate['required_phases']), 'container_gb': 30,
        'persistent_gb': 80, 'volume_mount': '/workspace', 'minimum_system_ram_gib': 40,
        'total_hourly_usd': None, 'inclusive_session_budget_usd': None,
        'deployed_unix': None, 'deadline_unix': None, 'expected_gpu': None,
        'expected_gpu_uuid': None, 'ledger_sha256': None}


def setup_requirements(manifest_sha):
    """Preparation metadata only; no executable launcher or resource allowance."""
    return {'kind': 'candidate-eot8-lower-cost-setup-requirements-v1',
        'packet_manifest_sha256': manifest_sha, 'source_file_count': 76,
        'expected_CPU_test_count': 77, 'CPU_tests': CPU_TESTS,
        'CPU_pytest_options': ['-q', '-p', 'no:cacheprovider'],
        'CPU_basetemp': '<new-workspace-session>/cpu-fixtures-temp',
        'CPU_basetemp_must_be_on_workspace': True,
        'minimum_free_workspace_gib_after_inputs': 55, 'minimum_usable_system_ram_gib': 40,
        'minimum_visible_gpu_vram_gib': 22,
        'base_torch_requirement': '2.8.0+cu128', 'replace_base_torch_or_CUDA': False,
        'run_all_13_original_benchmark_phases': True, 'research_calibration_enabled': False,
        'minimum_prior_review': ['final billing and retained reservations',
            'fresh sufficient-memory all-in quote', 'concrete reviewed session allowance',
            'actual deployment-based clock and verified host/GPU binding',
            'tested operational launcher and explicit payload/destination approval'],
        'setup_upload_download_and_idle_count_from_actual_deployment': True,
        'old_session_deadline_or_grant_reused': False, 'old_financial_hold_released': False,
        'new_financial_hold_created': False, 'maximum_billed_hours_proposal': 2,
        'session_budget_usd': None, 'hourly_quote_usd': None,
        'SSH_endpoint': None, 'GPU_identity': None, 'launch_authorized': False,
        'automatic_billing_stop': False, 'remote_upload_performed': False}


def build(root, base, output):
    root, base, output = Path(root).resolve(), Path(base).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing packets: ' + str(output))
    if digest(base / 'manifest.json') != BASE_MANIFEST_SHA256:
        raise ValueError('Use the immutable verified v3 packet manifest.')
    old = json.loads((base / 'manifest.json').read_text())
    if (old['packaged_training_clips'] != 100 or old['held_out_pixels_included']
            or old['cuda_execution_performed'] or old['executed_training_clips'] != 0):
        raise ValueError('Require the original training-only, unexecuted input packet.')
    assets = {name + '.tar.gz': old['archives'][name]['sha256'] for name in ('source', 'training', 'wheels')}
    assets.update({'upstream.bundle': old['upstream_bundle_sha256'], 'restore.py': old['restore_sha256']})
    for name, expected in assets.items():
        if (base / name).is_symlink() or digest(base / name) != expected:
            raise ValueError('Frozen packet file changed: ' + name)
    files = source_files(root, old)
    output.mkdir(parents=True, exist_ok=False)
    archives = deepcopy(old['archives'])
    archives['source'] = make_archive(output / 'source.tar.gz', files)
    for name in ('training.tar.gz', 'wheels.tar.gz', 'upstream.bundle', 'restore.py'):
        shutil.copyfile(base / name, output / name)
        if digest(output / name) != assets[name]:
            raise ValueError('Copied input differs: ' + name)
    preserve = ('base_commit', 'checkpoint_bytes', 'checkpoint_sha256', 'checkpoint_url',
                'restore_sha256', 'upstream_bundle_sha256', 'upstream_commit', 'candidate')
    manifest = {name: deepcopy(old[name]) for name in preserve}
    manifest.update(schema_version=1, archives=archives, diagnostic_only=True, research_admitted=False,
        source_is_working_tree_snapshot=True, packaged_training_clips=100,
        executed_training_clips=0, held_out_pixels_included=False, cuda_execution_performed=False,
        upstream_bundle_uses_committed_objects_only=True,
        predecessor_manifest_sha256=BASE_MANIFEST_SHA256,
        source_preparation={'original_files_unchanged': 75, 'candidate_source_inputs_unchanged': 61,
            'original_provenance_script_added': EXTRA, 'addition_sha256': EXTRA_SHA256,
            'workflow_provenance_dependency_count': 24, 'allowlist_is_explicit_not_a_glob': True},
        new_quote_ledger_identity_and_deployment_deadline_required=True,
        launch_authorized=False, new_hold_created=False, prior_holds_released=False)
    def save(name, value):
        with (output / name).open('x') as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    save('manifest.json', manifest)
    save('launch.template.json', launch_template(manifest['candidate']))
    save('setup.requirements.json', setup_requirements(digest(output / 'manifest.json')))
    return manifest

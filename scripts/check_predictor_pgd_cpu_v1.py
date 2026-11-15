#!/usr/bin/env python3
"""One local paired PGD rollout or its conditional fresh-process candidate replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = ('attacks/pgd_context_v1.py', 'models/predictor_pgd_cpu_v1.py',
         'models/predictor_pgd_records_v1.py', 'scripts/check_predictor_pgd_cpu_v1.py',
         'tests/test_predictor_pgd_context_v1.py', 'configs/predictor_pgd_cpu_v1.json')


def sources(snapshot):
    values = json.loads(Path(snapshot).read_bytes())
    if not set(FILES) <= set(values):
        raise ValueError('Complete new source snapshot required before work.')
    for name, want in values.items():
        p = ROOT/name
        with p.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != want:
            raise ValueError('Source snapshot changed: '+name)
    return values


def run(stage, mode, snapshot):
    if platform.node().split('.')[0] in ('linux10603', 'linux10609'):
        raise ValueError('Local CPU only; university work refused before model access.')
    if mode not in ('paired', 'replay'):
        raise ValueError('One paired or conditional fresh-process rollout only.')
    pinned = sources(snapshot)
    stage = Path(stage).resolve(strict=True)
    if stage.is_symlink() or stage.stat().st_mode & 0o777 != 0o700:
        raise ValueError('Private existing stage required.')
    os.umask(0o077)
    sys.path.insert(0, str(ROOT))
    from models.predictor_pgd_records_v1 import review
    from models.predictor_pgd_cpu_v1 import (ObjectiveLedger, IteratedBinding, rollout,
                                            save_arrays, write_json, digest, IDENTITY)
    paired_sha = None
    if mode == 'replay':
        prior = json.loads((stage/'paired'/'receipt.json').read_bytes())
        if prior['pid'] == os.getpid() or prior['sources'] != pinned:
            raise ValueError('Replay must be a fresh process with the identical snapshot.')
        admitted = json.loads((stage/'paired-review.json').read_bytes())
        if not admitted.get('passed') or admitted != review(stage/'paired'):
            raise ValueError('All initial numerical/ownership/selection gates must pass before replay.')
        paired_sha = digest(stage/'paired'/'receipt.json')
    write_json(stage/('claim-'+mode+'.json'), {'kind': 'local-tiny-objective-claim-v1',
                'mode': mode, 'pid': os.getpid(), 'maximum_gradient_objectives': 8 if mode == 'paired' else 4,
                'sources': pinned, 'paired_receipt_sha256': paired_sha,
                'CPU_only': True, 'university_or_paid_execution_authorized': False})
    folder = stage/mode
    folder.mkdir(mode=0o700)
    ledger = ObjectiveLedger(folder, mode)
    result = dict(kind='tiny-pgd-context-rollout-v1', algorithm=IDENTITY, mode=mode,
                  pid=os.getpid(), sources=pinned, paired_receipt_sha256=paired_sha,
                  CPU_only=True, pretrained=False, CUDA_qualified=False,
                  training_or_research_qualified=False, complete=False)
    binding = None
    try:
        import torch
        from models.native_diagnostic import tiny_fixture
        from models.execution import configure_execution, apply_model_policy, precision_contract
        if torch.is_inference_mode_enabled() or torch.is_autocast_enabled('cpu'):
            raise ValueError('Only original FP32 first-order CPU execution.')
        torch.set_num_threads(2)
        configure_execution('shared-fp64-native', device='cpu')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261114)
            model, pixels, masks = tiny_fixture(ROOT/'external/vjepa2')
        apply_model_policy(model, 'shared-fp64-native')
        model.eval()
        targets = model.clean_targets(pixels)
        attack = [masks(19+i) for i in range(8)]
        score = [masks(119+i) for i in range(4)]
        binding = IteratedBinding(model, pixels, targets, attack, score, ledger)
        initial_precision, initial_rng = precision_contract(), torch.get_rng_state().clone()
        arrays = rollout(binding)
        if precision_contract() != initial_precision or not torch.equal(torch.get_rng_state(), initial_rng):
            raise ValueError('Global precision or RNG changed during PGD.')
        result.update(files=save_arrays(folder, arrays), complete=True,
                      model_state_sha256=binding.state, no_grad_calls=binding.no_grad_calls,
                      original_objectives=sum(r['role'] == 'original' for r in ledger.starts),
                      candidate_objectives=sum(r['role'] == 'candidate' for r in ledger.starts),
                      global_precision_restored=True, global_rng_unchanged=True,
                      torch_version=torch.__version__, threads=torch.get_num_threads(),
                      masks_seed=19, independent_scoring_seed=119, restart_seed=73,
                      fixed_test_step_255=1, step_is_research_calibration=False)
    except BaseException as error:
        result.update(error=repr(error), consumed_starts=ledger.starts,
                      no_grad_calls=[] if binding is None else binding.no_grad_calls)
        raise
    finally:
        write_json(folder/'receipt.json', result)
    print(json.dumps({'mode': mode, 'complete': True, 'gradient_objectives': len(ledger.starts),
                      'folder': str(folder), 'CPU_only': True}))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, type=Path)
    p.add_argument('--mode', required=True, choices=('paired', 'replay'))
    p.add_argument('--snapshot', required=True, type=Path)
    a = p.parse_args(argv)
    return run(a.stage, a.mode, a.snapshot)


if __name__ == '__main__':
    sys.exit(main())

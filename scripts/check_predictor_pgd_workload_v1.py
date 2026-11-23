#!/usr/bin/env python3
"""One bounded analytic CPU unit in a fresh interpreter. No target/GPU binding."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=('primary', 'convergence'))
    parser.add_argument('--journal', required=True, type=Path)
    parser.add_argument('--audits', type=int, choices=(0, 2), default=0)
    args = parser.parse_args()
    if not args.journal.is_absolute():
        parser.error('Use a fresh absolute local journal path.')
    # Explicit repository path under -I; entry exposes no worker/SSH/GPU option.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from models.candidate_evaluation import EvaluationSettings
    from models.predictor_pgd_workload_v1 import LocalUnit
    from models.predictor_pgd_workload_controls_v1 import AnalyticFactory, analytic_fixture
    from models.predictor_pgd_workload_records_v1 import validate_receipt
    torch.set_num_threads(1)
    model, clean, masks = analytic_fixture()
    unit = LocalUnit(factory=AnalyticFactory())
    row, _ = unit.run(model, clean, masks, EvaluationSettings.for_method(args.method),
                      video_id='local-workload-control/fresh-process-v1',
                      sensitivity_banks=args.audits, journal=args.journal)
    validate_receipt(row)
    print('complete analytic CPU '+args.method+'; pid='+str(os.getpid()))


if __name__ == '__main__':
    main()

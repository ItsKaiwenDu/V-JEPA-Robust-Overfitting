"""Assess the pinned, already downloaded v5 result; never contact a host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import predictor_sdpa_saved_reader_v1 as reader

REVIEW = 'report/checks/2026-11-13-predictor-sdpa-continuation-v5-target-review.json'
REVIEW_SHA = '704ec1836d6eb35b178085166d4fd537a8865a17c1caadc25f4d5fd7ca2649b5'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous saved-only assessments.')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    review_path = ROOT/REVIEW
    if sha(review_path) != REVIEW_SHA:
        raise ValueError('Frozen failed-attempt review changed.')
    review = json.loads(review_path.read_text())
    pins_path = ROOT/review['target_input_pins']['path']
    if sha(pins_path) != review['target_input_pins']['sha256']:
        raise ValueError('Frozen target descriptors changed.')
    pins = json.loads(pins_path.read_text())
    if len(pins) != 195 or review['overall_run_passed'] is not False:
        raise ValueError('The preserved failed cohort must stay failed.')
    for name, item in pins.items():
        p = ROOT/name
        s = p.lstat()
        if (s.st_size, s.st_mode & 0o777, s.st_mtime_ns, sha(p)) != (item['bytes'], item['mode'], item['mtime_ns'], item['sha256']):
            raise ValueError('Saved target artifact changed: ' + name)
    saved = ROOT/review['session']/'saved'
    result = reader.assess_saved_worker(saved/'results/components/0', saved/'results/components/component-plan.json',
                                       0, capture_root=saved/'results/capture')
    result.update(prior_review_sha256=REVIEW_SHA, prior_overall_attempt_stays_failed=True,
                  target_descriptors_verified=len(pins), tests_or_experiments_launched=False)
    os.umask(0o077)
    with args.output.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'saved_assessment_passed': True, 'prior_attempt_stays_failed': True,
                      'new_target_work': False, 'output': str(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()

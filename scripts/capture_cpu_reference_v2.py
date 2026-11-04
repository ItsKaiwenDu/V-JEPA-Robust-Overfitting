#!/usr/bin/env python3
"""Inner v2 CPU capture runner, invoked only by the authorized supervisor."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--launcher-started', type=float, required=True)
    p.add_argument('--control-sha256', required=True)
    a = p.parse_args()
    from models.cpu_oracle_capture_v2 import old
    if old.digest(a.scratch/'operator-control.json') != a.control_sha256:
        p.error('Changed new v2 control.')
    from models.cpu_oracle_capture_runtime_v2 import run_workers
    result = run_workers(a.packet, a.scratch, a.upstream, a.launcher_started)
    raise SystemExit(0 if result['complete'] and result['saved_value_checks_passed'] else 1)


if __name__ == '__main__': main()

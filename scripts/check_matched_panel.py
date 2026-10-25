#!/usr/bin/env python3
"""Execute or inspect the frozen training-only matched panel; never rent compute."""

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('first','remaining','retry','worker','review','aggregate'))
    p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--case-id')
    p.add_argument('--role',choices=('matrix','replay'))
    p.add_argument('--acceptance',type=Path)
    p.add_argument('--archive',type=Path)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--expected-sha256')
    p.add_argument('--trusted-handoff',type=Path)
    p.add_argument('--acceptance-output',type=Path)
    args=p.parse_args()
    if not args.diagnostic:p.error('Use explicit --diagnostic; this is not research execution.')
    if args.mode in ('first','remaining','retry','worker') and (args.checkpoint is None or not args.checkpoint.is_file()):
        p.error('Use the existing released checkpoint; no download or random fallback.')
    if args.mode in ('worker','retry') and not args.case_id:p.error('An explicit declared --case-id is required.')
    if args.mode=='worker' and (args.role is None or args.output.exists()):p.error('Worker requires role and fresh output.')
    if args.mode=='remaining' and args.acceptance is None:p.error('The independent first-case receipt is required.')
    if args.mode=='review':
        if (any(x is None for x in (args.archive,args.manifest,args.expected_sha256,args.trusted_handoff))
                or args.output.exists()):p.error('Review requires trusted handoff, archive/member manifest, external SHA and fresh output.')
        from models.matched_panel import first_acceptance
        from models.matched_panel_io import execution_identity,load_plan,review_export
        from models.study_artifacts import write_record
        plan=load_plan(args.trusted_handoff)
        execution=execution_identity(args.trusted_handoff,plan)
        from models.pilot_bundle import digest
        for name,expected in execution['source_sha256'].items():
            if digest(ROOT/name)!=expected:raise ValueError('Current verifier differs from the trusted handoff source: '+name)
        receipt=review_export(args.archive,json.loads(args.manifest.read_text()),args.expected_sha256,
                              args.output,plan=plan,execution=execution)
        if args.acceptance_output is not None:
            write_record(args.acceptance_output,first_acceptance(plan,execution,receipt,args.expected_sha256))
        print('Independent case export verification PASS:',receipt['case_id'])
        print('No new model forward, GPU job or research admission.')
    elif args.mode=='aggregate':
        from models.matched_panel import PanelJournal
        from models.matched_panel_io import execution_identity,load_plan
        from models.matched_panel_runner import export_index
        plan=load_plan(ROOT)
        summary=export_index(PanelJournal(args.output,plan=plan,execution=execution_identity(ROOT,plan)))
        print('Panel complete:',summary['complete'])
    elif args.mode=='worker':
        from models.matched_panel_runner import worker
        worker(ROOT,args)
    else:
        from models.matched_panel_runner import run_panel
        summary=run_panel(ROOT,args)
        print('Panel command finished. Complete collection:',summary['complete'])
        print('First mode pauses after one verified case; download and review its export before remaining.')


if __name__=='__main__':main()

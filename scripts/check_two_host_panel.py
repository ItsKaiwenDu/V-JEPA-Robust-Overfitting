#!/usr/bin/env python3
"""One explicitly revised panel case; no automatic queue or retry."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def checksum(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('preflight','run','worker','review','register-review'))
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--revision-sha256',required=True)
    parser.add_argument('--implementation-sha256',required=True)
    parser.add_argument('--request',type=Path)
    parser.add_argument('--request-sha256')
    parser.add_argument('--role',choices=('matrix','replay'))
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--expected-archive-sha256')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--marker',type=Path)
    parser.add_argument('--marker-sha256')
    args=parser.parse_args();packet=Path(__file__).resolve().parents[1]
    manifest=json.loads((packet/'IMPLEMENTATION.json').read_text())
    if checksum(manifest)!=args.implementation_sha256:parser.error('Implementation differs from external pin.')
    # Reject a changed implementation before importing or executing it.
    for name in ('models/two_host_panel.py','scripts/check_two_host_panel.py'):
        if hashlib.sha256((packet/name).read_bytes()).hexdigest()!=manifest['source_sha256'][name]:
            parser.error('Pinned entry point or operator source changed.')
    source=packet/'models/two_host_panel.py'
    spec=importlib.util.spec_from_file_location('two_host_panel_operator',source)
    ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
    shared=(args.package,packet,args.revision_sha256,args.implementation_sha256)
    numerical=(args.root,args.request,args.request_sha256)
    export=(args.archive,args.manifest,args.expected_archive_sha256,args.output)
    marker=(args.marker,args.marker_sha256)
    if args.mode in ('preflight','run','worker'):
        if any(v is None for v in numerical) or any(v is not None for v in (*export,*marker)):
            parser.error('Use only one case request, fixed root and external pins for execution.')
        if (args.mode=='worker')!=(args.role is not None):parser.error('Only a fresh worker accepts a role.')
        call=(args.package,packet,args.root,args.revision_sha256,args.implementation_sha256,args.request,args.request_sha256)
        if args.mode=='worker':ops.numeric_worker(*call,args.role);return
        result=(ops.run_one if args.mode=='run' else ops.preflight)(*call)
    elif args.mode=='review':
        if (any(v is None for v in export) or any(v is not None for v in (*numerical,*marker,args.role))
            or args.output.exists()):parser.error('Review needs a pinned export and fresh output, with no execution arguments.')
        result=ops.review_case(*shared,args.archive,json.loads(args.manifest.read_text()),args.expected_archive_sha256,args.output)
        ops.write_record(args.output/'independent-review.json',result)
    else:
        if (args.root is None or any(v is None for v in marker)
            or any(v is not None for v in (args.request,args.request_sha256,args.role,*export))):
            parser.error('Registration accepts only an externally pinned independent review and existing ledger.')
        result=ops.register_review(args.package,packet,args.root,args.revision_sha256,args.implementation_sha256,*marker)
    print(json.dumps(result,sort_keys=True,indent=2),flush=True)


if __name__=='__main__':main()

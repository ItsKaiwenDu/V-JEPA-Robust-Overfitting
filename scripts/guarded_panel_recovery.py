#!/usr/bin/env python3
"""Operator-only recovery envelope around the unchanged, qualified panel CLI.

No neural-model imports. Unknown old elapsed time is fully charged; extra
time needs a separate explicit authorization. The original session is never edited.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

MANIFEST_SHA='73f361fc072a053439bf71f35835bb91227932edd2db739adde2db5f96c6b0f4'
PLAN_SHA='f6b81dec841aeee703413dba23ea16afe3dbf14736edecadc0f7830f3ded34ae'
EXECUTION_SHA='1e59a735e2f0b6f5485cf0867851c638dcd027715d19270422f390f7745aa70e'
CASE='clip-01-bank-0'
LIMIT=4800


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read_record(path):
    value=json.loads(Path(path).read_text())
    if set(value)!={'payload','sha256'} or canonical(value['payload'])!=value['sha256']:
        raise ValueError('Record checksum mismatch: '+str(path))
    return value['payload']


def write_record(path,payload):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump({'payload':payload,'sha256':canonical(payload)},stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def assess_old_time(session,limit=LIMIT):
    elapsed=session.get('elapsed_seconds')
    terminal_failure=(session.get('complete') is False and session.get('passed') is False
        and isinstance(session.get('error'),str) and bool(session['error'])
        and type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0)
    if not terminal_failure:
        return {'old_elapsed_known':False,'old_allowance_charged_seconds':limit,
                'old_allowance_remaining_seconds':0,'additional_authorization_required':True}
    charged=min(limit,elapsed+180)
    return {'old_elapsed_known':True,'old_allowance_charged_seconds':charged,
            'old_allowance_remaining_seconds':limit-charged,
            'additional_authorization_required':False}


def validate_authorization(proposal,authorization):
    expected={'kind':'panel-recovery-extra-time-authorization-v1','approved':True,
        'proposal_sha256':canonical(proposal),'case_id':CASE,'extra_case_seconds':LIMIT,
        'fresh_attempt_from_scratch':True,'research_admitted':False}
    if any(k not in authorization or canonical(authorization[k])!=canonical(v) for k,v in expected.items()):
        raise ValueError('A matching explicit additional-time authorization is required.')
    grant=authorization.get('grant_id','')
    if not isinstance(grant,str) or re.fullmatch('[a-f0-9]{32}',grant) is None:
        raise ValueError('Invalid single-use grant identity.')
    return grant


def verify_inventory(directory,expected):
    directory=Path(directory).resolve()
    actual={str(p.relative_to(directory)):p for p in directory.rglob('*') if p.is_file() or p.is_symlink()}
    if set(actual)!=set(expected):raise ValueError('Partial attempt inventory changed.')
    for name,path in actual.items():
        if path.is_symlink() or not path.resolve().is_relative_to(directory) or digest(path)!=expected[name]:
            raise ValueError('Partial attempt artifact changed: '+name)


def preflight(package,panel,proposal,authorization):
    grant=validate_authorization(proposal,authorization)
    package,panel=Path(package).resolve(),Path(panel).resolve()
    if socket.gethostname()!='linux10609' or panel!=package/'runs/panel.qIXeDn':
        raise ValueError('Use the declared university host and original panel directory.')
    if digest(__file__)!=proposal['operator_source_sha256'] or digest(package/'PANEL-HANDOFF.json')!=MANIFEST_SHA:
        raise ValueError('Operator or qualified package identity changed.')
    manifest=json.loads((package/'PANEL-HANDOFF.json').read_text())
    for name,row in manifest['files'].items():
        relative=Path(name);path=package/relative
        if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(package) or not path.is_file()
                or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']):
            raise ValueError('Qualified handoff changed: '+name)
    if digest(panel/'journal.json')!=proposal['journal_file_sha256']:
        raise ValueError('Panel journal changed since recovery review.')
    journal=read_record(panel/'journal.json')
    if (journal['plan_sha256']!=PLAN_SHA or journal['execution_sha256']!=EXECUTION_SHA
            or set(journal['completed'])!={'clip-00-bank-0','clip-00-bank-1'}
            or journal['research_admitted'] is not False):
        raise ValueError('Unexpected panel recovery boundary.')
    for case_id,pointer in journal['completed'].items():
        def bound(relative):
            candidate=panel/relative
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(panel):
                raise ValueError('Escaped completed-case artifact.')
            return candidate
        receipt_path=bound(pointer['directory']+'/receipt.json')
        if digest(receipt_path)!=pointer['receipt_file_sha256']:
            raise ValueError('A completed receipt changed.')
        receipt=read_record(receipt_path)
        if receipt['case_id']!=case_id or receipt['execution_sha256']!=EXECUTION_SHA:
            raise ValueError('Completed receipt identity changed.')
        if digest(bound(pointer['directory']+'/session.json'))!=receipt['session_sha256']:
            raise ValueError('Completed session changed.')
        for relative,pin in receipt['worker_files_sha256'].items():
            if digest(bound(pointer['directory']+'/'+relative))!=pin:
                raise ValueError('Completed worker evidence changed.')
        for key in ('archive','archive_manifest'):
            if digest(bound(pointer[key+'_path']))!=pointer[key+'_sha256']:
                raise ValueError('Completed archive evidence changed.')
    attempt=panel/'cases'/CASE/'attempt-01'
    verify_inventory(attempt,proposal['partial_files_sha256'])
    if assess_old_time(read_record(attempt/'session.json'))!=proposal['old_time_assessment']:
        raise ValueError('Old runtime assessment differs.')
    if (panel/'cases'/CASE/'attempt-02').exists():raise ValueError('A second attempt already exists.')
    if digest(panel/'first-case-acceptance.json')!=proposal['first_acceptance_file_sha256']:
        raise ValueError('First-case acceptance changed.')
    return grant


def consume(panel,grant,proposal,authorization):
    directory=Path(panel)/'operator-recovery'/grant
    directory.mkdir(parents=True,exist_ok=False)
    write_record(directory/'proposal.json',proposal)
    write_record(directory/'authorization.json',authorization)
    return directory


def supervise(command,directory,seconds,*,heartbeat_seconds=15,kill_grace_seconds=12):
    """Bound one process tree and write fresh per-invocation status, never retry."""
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
           for v in (seconds,heartbeat_seconds,kill_grace_seconds)):
        raise ValueError('Finite positive operator deadlines are required.')
    directory=Path(directory);started=time.monotonic();process=None;stop_reason=None
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    def stop(number,_frame):
        nonlocal stop_reason
        stop_reason='signal-'+str(number)
    for number in handlers:signal.signal(number,stop)
    state={'kind':'panel-recovery-operator-status-v1','state':'starting','elapsed_seconds':0.,
           'research_admitted':False,'scientific_case_verified':False}
    def save():
        state['elapsed_seconds']=time.monotonic()-started
        state['updated_UTC']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        write_record(directory/'status.json',state)
    save()
    try:
        with (directory/'operator.log').open('xb') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
            state.update(state='running',pid=process.pid,case_seconds_limit=seconds)
            save();next_heartbeat=time.monotonic()+heartbeat_seconds
            while process.poll() is None:
                if stop_reason is not None:break
                if time.monotonic()-started>=seconds:
                    stop_reason='deadline';break
                if time.monotonic()>=next_heartbeat:
                    save();next_heartbeat=time.monotonic()+heartbeat_seconds
                time.sleep(min(.1,max(.001,seconds-(time.monotonic()-started))))
            if stop_reason is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=kill_grace_seconds)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=process.wait()
            state.update(state='process_finished' if code==0 and stop_reason is None else 'stopped',
                         exit_status=code,stop_reason=stop_reason)
            save()
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=kill_grace_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
        state.update(state='operator_error',stop_reason='operator-exception');save()
        raise
    finally:
        for number,handler in handlers.items():signal.signal(number,handler)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--proposal',type=Path,required=True)
    parser.add_argument('--authorization',type=Path,required=True)
    args=parser.parse_args()
    proposal,authorization=read_record(args.proposal),read_record(args.authorization)
    grant=preflight(args.package,args.panel,proposal,authorization)
    python=Path.home()/'.venvs/vjepa/bin/python'
    checkpoint=Path.home()/'Desktop/VJEPA-Robust-Overfitting/V-JEPA-Robust-Overfitting/checkpoints/vitl.pt'
    if not python.is_file() or not checkpoint.is_file():raise ValueError('Existing runtime/checkpoint required.')
    directory=consume(args.panel,grant,proposal,authorization)
    command=[str(python),'-u',str(args.package/'scripts/check_matched_panel.py'),'retry','--diagnostic',
        '--case-id',CASE,'--output',str(args.panel),'--checkpoint',str(checkpoint),
        '--acceptance',str(args.panel/'first-case-acceptance.json')]
    print('Attended single-case recovery. Keep this SSH session open.',flush=True)
    print('Fresh operator status:',directory/'status.json',flush=True)
    print('Fresh operator log:',directory/'operator.log',flush=True)
    state=supervise(command,directory,LIMIT)
    if state.get('exit_status')!=0 or state['state']!='process_finished':return 1
    verification='''import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from models.matched_panel_io import load_plan,execution_identity
from models.matched_panel import PanelJournal
r,p=Path(sys.argv[1]),Path(sys.argv[2]);plan=load_plan(r)
j=PanelJournal(p,plan=plan,execution=execution_identity(r,plan))
receipt=j.verify_completed('clip-01-bank-0')
assert j.read()['completed']['clip-01-bank-0']['directory']=='cases/clip-01-bank-0/attempt-02'
print('Recovered case independently reads back; remaining queue is not started.')
'''
    try:
        subprocess.run([str(python),'-c',verification,str(args.package),str(args.panel)],check=True,timeout=180)
    except BaseException:
        state.update(state='verification_failed');write_record(directory/'status.json',state);raise
    state.update(state='case_verified',scientific_case_verified=True)
    write_record(directory/'status.json',state)
    print('Single-case recovery finished; download its export for local review.',flush=True)
    return 0


if __name__=='__main__':sys.exit(main())

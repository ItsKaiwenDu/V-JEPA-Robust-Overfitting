"""Bounded host/GPU observation; no university file writes or model launch.

Definitions are inert on import. The approved SSH operator supplies a nonce and
control/source hashes; a separate GNU timeout bounds the remote query group.
"""
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
import math
import os
import platform
import selectors
import subprocess
import sys
import time

HOST={'hostname':'linux10603','system':'Linux','python':'3.11.17'}
GPU={'name':'NVIDIA GeForce RTX 3070','uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f','memory_total_MiB':8192.0,'driver':'580.178.04'}
COMMANDS=(
 ['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,driver_version','--format=csv,noheader,nounits'],
 ['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'])
QUERY_SECONDS=8;QUERY_STDOUT=8192;QUERY_STDERR=2048;RESPONSE_CAP=65536


def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(value).hexdigest()
def is_sha(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


class CaptureError(RuntimeError):
    def __init__(self,reason,stdout=b'',stderr=b''):
        super().__init__(reason);self.stdout=bytes(stdout);self.stderr=bytes(stderr)


def capture(argv,*,deadline,stdout_cap,stderr_cap,clock=None,popen=None,selector_factory=None,set_blocking=None,read=None):
    """Bounded pipes and original deadline; only the owned unreaped child is killed.

    No shell, process-group discovery, numeric-PID API or retry. The remote outer
    GNU timeout independently covers this program and its query children.
    """
    clock=time.monotonic if clock is None else clock;popen=subprocess.Popen if popen is None else popen
    selector_factory=selectors.DefaultSelector if selector_factory is None else selector_factory
    set_blocking=os.set_blocking if set_blocking is None else set_blocking;read=os.read if read is None else read
    if (type(argv) is not list or not argv or any(type(a) is not str or not a or '\0' in a for a in argv)
            or type(deadline) not in (int,float) or not math.isfinite(deadline)
            or not all(type(v) is int and 0<v<=65536 for v in (stdout_cap,stderr_cap))):raise ValueError('Exact bounded argv/deadline/output limits required.')
    proc=None;selector=None;streams=[];buffers={'stdout':bytearray(),'stderr':bytearray()};caps={'stdout':stdout_cap,'stderr':stderr_cap}
    try:
        if clock()>=deadline:raise CaptureError('deadline before spawn')
        proc=popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True,shell=False,bufsize=0)
        selector=selector_factory()
        for name,stream in (('stdout',proc.stdout),('stderr',proc.stderr)):
            streams.append(stream);fd=stream.fileno();set_blocking(fd,False);selector.register(fd,selectors.EVENT_READ,name)
        while selector.get_map():
            remaining=deadline-clock()
            if remaining<=0:raise CaptureError('deadline while reading')
            for key,mask in selector.select(min(.25,remaining)):
                name=key.data
                try:chunk=read(key.fd,min(4096,caps[name]-len(buffers[name])+1))
                except BlockingIOError:continue
                if not chunk:selector.unregister(key.fd);continue
                if type(chunk) is not bytes or len(buffers[name])+len(chunk)>caps[name]:raise CaptureError(name+' exceeded bound')
                buffers[name].extend(chunk)
        remaining=deadline-clock()
        if remaining<=0:raise CaptureError('deadline before wait')
        status=proc.wait(timeout=remaining)
        if clock()>=deadline:raise CaptureError('deadline after wait')
        return status,bytes(buffers['stdout']),bytes(buffers['stderr'])
    except BaseException as error:
        if proc is not None:
            try:
                if proc.poll() is None:proc.kill() # Only this still-owned unreaped Popen child.
            except BaseException as cleanup:error.add_note('Owned query cleanup also failed: '+type(cleanup).__name__)
            finally:
                try:proc.wait(timeout=max(0,min(.25,deadline-clock())))
                except BaseException:pass # No extension beyond original clock.
        if isinstance(error,(KeyboardInterrupt,SystemExit)):
            error.stdout=bytes(buffers['stdout']);error.stderr=bytes(buffers['stderr']);raise
        reason=str(error)[:256] or type(error).__name__
        raise CaptureError(reason,buffers['stdout'],buffers['stderr']) from error
    finally:
        if selector is not None:selector.close()
        for stream in streams:stream.close()


def rows(text,maximum):
    if type(text) is not str or len(text.encode())>QUERY_STDOUT:raise ValueError('Bounded query text required.')
    value=[row for row in csv.reader(io.StringIO(text),skipinitialspace=True) if row and any(s.strip() for s in row)]
    if len(value)>maximum:raise ValueError('Query row count exceeded.')
    return [[s.strip() for s in row] for row in value]


def number(value):
    if type(value) is not str or len(value)>32:raise ValueError('Bounded memory value required.')
    result=float(value)
    if not math.isfinite(result) or not 0<=result<=8192:raise ValueError('Finite expected memory range required.')
    return result


def evaluate(host,queries):
    """Observation only; idle is neither a reservation nor model authorization."""
    result={'availability':'unknown','gpu':None,'compute_processes':[],'errors':[]}
    if host!=HOST:
        result.update(availability='identity_refused',errors=['Exact university host/runtime differs.']);return result
    try:
        if type(queries) is not list or not 1<=len(queries)<=2:raise ValueError('Expected one or two query records.')
        for i,q in enumerate(queries):
            if type(q) is not dict or set(q)!={'argv','exit_status','stdout','stderr','capture_error'} or q['argv']!=COMMANDS[i]:raise ValueError('Unexpected query or fields.')
            if type(q['stdout']) is not str or type(q['stderr']) is not str or len(q['stdout'].encode())>QUERY_STDOUT or len(q['stderr'].encode())>QUERY_STDERR:raise ValueError('Bounded query result required.')
            if q['capture_error'] is not None or type(q['exit_status']) is not int or q['exit_status']!=0:raise ValueError('Query failed or timed out.')
        gpu_rows=rows(queries[0]['stdout'],1)
        if len(gpu_rows)!=1 or len(gpu_rows[0])!=5:raise ValueError('Exactly one complete GPU row required.')
        name,uuid,total,used,driver=gpu_rows[0];observed={'name':name,'uuid':uuid,'memory_total_MiB':number(total),'driver':driver}
        if observed!=GPU:
            result.update(availability='identity_refused',errors=['GPU/UUID/driver/total memory differs.']);return result
        result['gpu']=dict(observed,memory_used_MiB=number(used))
        if len(queries)!=2:raise ValueError('Compute-process query is missing.')
        processes=[];seen=set()
        for row in rows(queries[1]['stdout'],64):
            if (len(row)!=3 or not row[0].isdigit() or len(row[0])>10 or not 0<int(row[0])<2**31
                    or not row[1] or len(row[1])>512 or int(row[0]) in seen):raise ValueError('Malformed/duplicate compute process row.')
            seen.add(int(row[0]));processes.append({'pid':int(row[0]),'process_name':row[1],'used_memory_MiB':number(row[2])})
        result['compute_processes']=processes
        result['availability']='busy_observed' if processes or result['gpu']['memory_used_MiB']>512 else 'idle_observed'
    except (ValueError,TypeError,OverflowError,csv.Error) as error:result.update(availability='unknown',errors=[str(error)[:256]])
    return result


def observe(nonce,control_sha256,program_sha256,*,capture_fn=None,clock=None,identity=None,utc=None):
    if not all(is_sha(v) for v in (nonce,control_sha256,program_sha256)):raise ValueError('Bound nonce/control/source hashes required.')
    clock=time.monotonic if clock is None else clock;capture_fn=capture if capture_fn is None else capture_fn
    utc=(lambda:datetime.now(timezone.utc).isoformat()) if utc is None else utc
    identity=(lambda:{'hostname':platform.node().split('.')[0],'system':platform.system(),'python':platform.python_version()}) if identity is None else identity
    started=clock();host=identity();queries=[]
    if host==HOST:
        for argv in COMMANDS:
            try:
                status,out,err=capture_fn(list(argv),deadline=min(started+19,clock()+QUERY_SECONDS),stdout_cap=QUERY_STDOUT,stderr_cap=QUERY_STDERR)
                q={'argv':list(argv),'exit_status':status,'stdout':out.decode('utf-8','strict'),'stderr':err.decode('utf-8','strict'),'capture_error':None}
            except (CaptureError,UnicodeError) as error:
                q={'argv':list(argv),'exit_status':None,'stdout':getattr(error,'stdout',b'').decode('utf-8','replace')[:QUERY_STDOUT],
                   'stderr':getattr(error,'stderr',b'').decode('utf-8','replace')[:QUERY_STDERR],'capture_error':str(error)[:256]}
            queries.append(q)
            if q['exit_status']!=0 or q['capture_error'] is not None:break
    result={'kind':'university-gpu-availability-observation-v1','request_nonce':nonce,'control_sha256':control_sha256,'program_sha256':program_sha256,
        'host':host,'checked_utc':utc(),'elapsed_seconds':clock()-started,'queries':queries,**evaluate(host,queries),
        'read_only':True,'GPU_reserved':False,'model_work_authorized':False,'scientific_qualified':False,'training_admitted':False}
    if len(encoded(result))>RESPONSE_CAP:
        result.update(queries=[],availability='unknown',gpu=None,compute_processes=[],errors=['Response exceeded its bound; no readiness inference.'])
    if len(encoded(result))>RESPONSE_CAP:raise ValueError('Bounded response required.')
    return result


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if len(argv)!=3:raise SystemExit('Three bound public metadata values required.')
    result=observe(*argv)
    sys.stdout.buffer.write(encoded(result));sys.stdout.buffer.flush()
    return 0 if result['availability'] in ('idle_observed','busy_observed') else 1


if __name__=='__main__':sys.exit(main())

"""One-phase boundary evidence; never a five-phase or study qualification."""
import hashlib
import os
from pathlib import Path
from collections import Counter
from models import predictor_eot8_record_v2 as atomic
from models.predictor_boundary_contract_v1 import WORKER_KIND

FORMAT='predictor-boundary-record-v1'
IDENTITY=('kind','execution_host','local_tiny','mode','index','pid','control_sha256')
LIMITS={atomic.CALLS:atomic.MAX_CALL_BYTES,atomic.SCOPES:atomic.MAX_SCOPE_BYTES,
        atomic.HEADER:atomic.MAX_HEADER_BYTES,atomic.COMPLETE:atomic.MAX_MARKER_BYTES,atomic.FAILURE:atomic.MAX_MARKER_BYTES}


def identity(state):
    value={k:state.get(k) for k in IDENTITY}
    if (value['kind']!=WORKER_KIND or type(value['local_tiny']) is not bool
            or type(value['index']) is not int or value['index'] not in (0,1)
            or value['mode']!=('original' if value['index']==0 else 'predictor-math')
            or type(value['pid']) is not int or value['pid']<=0
            or not isinstance(value['execution_host'],str) or not 1<=len(value['execution_host'])<=255
            or not isinstance(value['control_sha256'],str) or len(value['control_sha256'])!=64):
        raise ValueError('Exact one-phase worker identity required.')
    return value


def verify_scope(state,rows,scopes,refusals):
    identity(state)
    if refusals or len(scopes)!=1 or not 0<len(rows)<=4096:raise ValueError('One complete native scope required.')
    s=scopes[0]
    if (s.get('phase')!='EOT8-before' or s.get('index')!=0 or s.get('state')!='complete'
            or s.get('hook_removed') is not True or s.get('pid')!=state['pid']
            or type(s.get('graph_id')) is not int or s['graph_id']<0
            or any(type(s.get(k)) is not int or s[k]<=0 for k in ('creator_thread','runner_thread','root_hook_thread','loss_id','input_id'))
            or not isinstance(s.get('call_threads'),list) or not 1<=len(set(s['call_threads']))<=8
            or len(s['call_threads'])!=len(set(s['call_threads']))):raise ValueError('Incomplete native graph/thread/hook receipt.')
    counts=(2,2) if state['local_tiny'] else (24,12)
    owners={r+'.'+str(i) for r,n in zip(('encoder','predictor'),counts) for i in range(n)}
    forward=Counter();back=Counter()
    for i,r in enumerate(rows):
        if (r.get('index')!=i or r.get('phase')!='EOT8-before' or r.get('pid')!=state['pid']
                or r.get('owner') not in owners or r['owner']!=str(r.get('role'))+'.'+str(r.get('block'))
                or r.get('creator_thread')!=s['creator_thread'] or r.get('entry_flags')!=r.get('exit_flags')
                or type(r.get('during_backward')) is not bool):raise ValueError('Native call identity/restoration changed.')
        if state['mode']=='predictor-math' and r['role']=='predictor':
            if r.get('effective_flags')!={'math':True,'flash':False,'cudnn':False,'mem_efficient':False}:raise ValueError('Predictor selection context changed.')
        elif r.get('effective_flags')!=r.get('entry_flags'):raise ValueError('Original/encoder call flags changed.')
        if r['during_backward']:
            if (r.get('graph_id')!=s['graph_id'] or r.get('backward_scope')!=0 or r.get('thread') not in s['call_threads']
                    or not (r.get('outcome')=='returned' or r.get('outcome')=='interrupted' and r.get('error_type')=='_StopRecomputationError')):
                raise ValueError('Unowned or incomplete native recomputation.')
            back[r['owner']]+=1
        else:
            if r.get('graph_id')!=-1 or r.get('backward_scope') is not None or r.get('thread')!=s['creator_thread'] or r.get('outcome')!='returned':raise ValueError('Original forward call changed.')
            forward[r['owner']]+=1
    if forward!=Counter({owner:16 for owner in owners}) or set(back)!=owners:raise ValueError('All eight draws/two families/owners required.')
    return {'kind':'predictor-boundary-one-phase-scope-v1','forward_calls':sum(forward.values()),
            'backward_calls':sum(back.values()),'scopes':1,'owners':len(owners),'hooks_removed':True,
            'backend_kernel_qualified':False,'full_five_phase_qualified':False}


class BoundaryRecord:
    def __init__(self,folder):
        self.folder,_=atomic._folder(folder)
        if any(os.path.lexists(self.folder/n) or os.path.lexists(self.folder/(n+'.partial')) for n in LIMITS):raise FileExistsError('Preserve existing boundary records.')
        self.previous={};self.row_hashes=[];self.closed=False;self.poisoned=False

    def write(self,state,route=None,*,final=False):
        if self.closed or self.poisoned:raise ValueError('Writer cannot retry or reuse a record.')
        try:return self._write(state,route,final)
        except BaseException:self.poisoned=True;raise

    def _write(self,state,route,final):
        key=identity(state)
        if any(n in state for n in atomic.INLINE):raise ValueError('No inline call/scope arrays.')
        rows=[] if route is None else route.rows;scopes=[] if route is None else route.scopes;refusals=[] if route is None else route.refusals
        if not 0<=len(rows)<=4096 or len(rows)<len(self.row_hashes) or len(scopes)>1 or len(refusals)>32:raise ValueError('Boundary record extent changed.')
        for n,item in self.previous.items():
            data=atomic._read(self.folder,n,item['bytes'])
            if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Prior record changed.')
        if state.get('passed'):
            if not final or not state.get('capture_complete') or not state.get('numerical_passed') or not state.get('reproduces_anchor') or not state.get('observer_restored') or not state.get('adapter_restored') or not state.get('model_unchanged'):raise ValueError('Passing one-phase capture requires complete original checks.')
        if state.get('capture_complete'):
            if verify_scope(state,rows,scopes,refusals)!=state.get('scope_check'):raise ValueError('Capture scope changed.')
        hashes=[]
        def lines():
            for i,row in enumerate(rows):
                if row.get('index')!=i:raise ValueError('Ordered actual calls required.')
                line=atomic._json(row)
                if len(line)>2048:raise ValueError('Call-row byte cap.')
                checksum=hashlib.sha256(line).hexdigest()
                if i<len(self.row_hashes) and checksum!=self.row_hashes[i]:raise ValueError('Earlier actual call changed.')
                hashes.append(checksum);yield line
        calls=atomic._atomic(self.folder,atomic.CALLS,lines(),LIMITS[atomic.CALLS]);calls['records']=len(rows)
        status='success' if state.get('passed') else 'failure' if final else 'running'
        scope=atomic._atomic(self.folder,atomic.SCOPES,[atomic._json({'kind':FORMAT,'identity':key,'status':status,'calls':calls,'scopes':scopes,'refusals':refusals})],LIMITS[atomic.SCOPES])
        header=atomic._atomic(self.folder,atomic.HEADER,[atomic._json({**state,'record_format':FORMAT,'evidence':{'calls':calls,'scopes':scope,'status':status}})],LIMITS[atomic.HEADER])
        if final:atomic._atomic(self.folder,atomic.COMPLETE if state.get('passed') else atomic.FAILURE,[atomic._json({'kind':FORMAT,'identity':key,'status':status,'header':header})],atomic.MAX_MARKER_BYTES)
        self.previous={x['file']:x for x in (calls,scope,header)};self.row_hashes=hashes;self.closed=final

    def failure(self,state,error):
        self.poisoned=True
        value={'kind':FORMAT,'identity':identity(state),'status':'writer-error','original_error_type':str(state.get('error_type',''))[:128],
               'original_error':str(state.get('error',''))[:2048],'error_type':type(error).__name__,'error':str(error)[:2048]}
        return atomic._atomic(self.folder,atomic.FAILURE,[atomic._json(value)],atomic.MAX_MARKER_BYTES)


def load(folder,*,control_sha256,require_passed=True):
    folder=Path(folder);raw=atomic._read(folder,atomic.HEADER,atomic.MAX_HEADER_BYTES);w=atomic._parse(raw)
    key=identity(w)
    if key['control_sha256']!=control_sha256 or w.get('record_format')!=FORMAT or any(n in w for n in atomic.INLINE):raise ValueError('Boundary domain/control changed.')
    marker_name=atomic.COMPLETE if w.get('passed') else atomic.FAILURE
    if w.get('passed') and os.path.lexists(folder/atomic.FAILURE):raise ValueError('Failed record is not passing.')
    marker=atomic._parse(atomic._read(folder,marker_name,atomic.MAX_MARKER_BYTES))
    status='success' if w.get('passed') else 'failure'
    if (marker.get('kind')!=FORMAT or marker.get('identity')!=key or marker.get('status')!=status
            or marker.get('header')!={'file':atomic.HEADER,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}):raise ValueError('Final boundary marker does not bind the header.')
    if require_passed and w.get('passed') is not True:raise ValueError('Passing original boundary capture required.')
    evidence=w.get('evidence',{})
    calls=atomic._verified_file(folder,evidence.get('calls'),atomic.CALLS,atomic.MAX_CALL_BYTES)
    scope=atomic._parse(atomic._verified_file(folder,evidence.get('scopes'),atomic.SCOPES,atomic.MAX_SCOPE_BYTES))
    if scope.get('identity')!=key or scope.get('kind')!=FORMAT or scope.get('status')!=status or scope.get('calls')!=evidence.get('calls') or evidence.get('status')!=status:raise ValueError('Boundary scope identity changed.')
    lines=calls.splitlines(keepends=True)
    if (calls and not calls.endswith(b'\n')) or len(lines)!=evidence['calls']['records'] or len(lines)>4096:raise ValueError('Call extent changed.')
    rows=[]
    for i,line in enumerate(lines):
        row=atomic._parse(line)
        if len(line)>2048 or row.get('index')!=i:raise ValueError('Call row/order changed.')
        rows.append(row)
    if len(scope.get('scopes',[]))>1 or len(scope.get('refusals',[]))>32:raise ValueError('Scope count changed.')
    result={**w,'call_rows':rows,'backward_scopes':scope['scopes'],'ownership_refusals':scope['refusals']}
    if w.get('capture_complete') and verify_scope(w,rows,scope['scopes'],scope['refusals'])!=w.get('scope_check'):raise ValueError('Recorded complete capture scope changed.')
    return result

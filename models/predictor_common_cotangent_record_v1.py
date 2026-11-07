"""Controlled VJP records: attribution completion never implies native acceptance."""
import hashlib
import os
import stat
from pathlib import Path
from models import predictor_eot8_record_v2 as atomic
from models.predictor_common_cotangent_contract_v1 import WORKER_KIND,CELLS

FORMAT='predictor-common-cotangent-record-v1'
COMPLETE='attribution-complete.json'
FAILURE='attribution-failure.json'
REJECTION='native-numerical-rejection.json'
IDENTITY=('kind','execution_host','local_tiny','mode','index','cell','path','cotangent','pid','control_sha256')
LIMITS={atomic.CALLS:atomic.MAX_CALL_BYTES,atomic.SCOPES:atomic.MAX_SCOPE_BYTES,atomic.HEADER:atomic.MAX_HEADER_BYTES,
        COMPLETE:atomic.MAX_MARKER_BYTES,FAILURE:atomic.MAX_MARKER_BYTES,REJECTION:atomic.MAX_MARKER_BYTES}


def _read(folder,name,limit):
    if name not in (COMPLETE,FAILURE,REJECTION):return atomic._read(folder,name,limit)
    if type(limit) is not int or not 0<=limit<=atomic.MAX_MARKER_BYTES:raise ValueError('Fixed bounded attribution marker required.')
    folder,parent=atomic._folder(folder);path=folder/name;before=path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_uid!=os.getuid()
            or before.st_dev!=parent.st_dev or not 0<=before.st_size<=limit):raise ValueError('Attribution marker ownership/type/size changed.')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        actual=os.fstat(f.fileno())
        if (actual.st_dev,actual.st_ino)!=(before.st_dev,before.st_ino):raise ValueError('Attribution marker changed while opening.')
        data=f.read(limit+1);after=os.fstat(f.fileno())
    if len(data)>limit or (actual.st_size,actual.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or len(data)!=after.st_size:
        raise ValueError('Attribution marker changed during bounded read.')
    return data


def _atomic(folder,name,chunks,limit):
    if name not in (COMPLETE,FAILURE,REJECTION):
        if name not in (atomic.CALLS,atomic.SCOPES,atomic.HEADER) or not 0<=limit<=LIMITS[name]:raise ValueError('Fixed bounded controlled record required.')
        return atomic._atomic(folder,name,chunks,limit)
    if type(limit) is not int or not 0<=limit<=atomic.MAX_MARKER_BYTES:raise ValueError('Fixed bounded attribution marker required.')
    folder,parent=atomic._folder(folder);path=folder/name;partial=folder/(name+'.partial')
    if os.path.lexists(path):_read(folder,name,limit)
    size=0;digest=hashlib.sha256()
    with os.fdopen(os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:
        for chunk in chunks:
            if not isinstance(chunk,bytes) or size+len(chunk)>limit:raise ValueError('Attribution marker byte ceiling.')
            f.write(chunk);size+=len(chunk);digest.update(chunk)
        f.flush();os.fsync(f.fileno())
    current=folder.lstat()
    if (current.st_dev,current.st_ino)!=(parent.st_dev,parent.st_ino):raise ValueError('Attribution directory changed during marker publication.')
    os.replace(partial,path);atomic._sync_directory(folder)
    return {'file':name,'bytes':size,'sha256':digest.hexdigest()}


def identity(state):
    value={k:state.get(k) for k in IDENTITY};i=value['index']
    if (value['kind']!=WORKER_KIND or type(value['local_tiny']) is not bool or type(i) is not int or i not in range(4)
            or any(type(value[k]) is not int for k in ('path','cotangent')) or tuple(value[k] for k in ('cell','path','cotangent'))!=CELLS[i]
            or value['mode']!=('original' if value['path']==0 else 'predictor-math')
            or type(value['pid']) is not int or value['pid']<=0 or not isinstance(value['execution_host'],str) or not 1<=len(value['execution_host'])<=255
            or not isinstance(value['control_sha256'],str) or len(value['control_sha256'])!=64):
        raise ValueError('Exact controlled-VJP cell identity required.')
    return value


def verify_scope(state,rows,scopes,refusals):
    identity(state)
    # This is a coverage validator, not an old boundary worker or passing record.
    from collections import Counter
    if refusals or len(scopes)!=1 or not 0<len(rows)<=4096:raise ValueError('One complete owned GraphTask required.')
    s=scopes[0]
    if (s.get('phase')!='EOT8-before' or s.get('index')!=0 or s.get('state')!='complete' or s.get('hook_removed') is not True
            or s.get('pid')!=state['pid'] or type(s.get('graph_id')) is not int or s['graph_id']<0
            or any(type(s.get(k)) is not int or s[k]<=0 for k in ('creator_thread','runner_thread','root_hook_thread','loss_id','input_id'))
            or not isinstance(s.get('call_threads'),list) or not 1<=len(set(s['call_threads']))<=8
            or len(s['call_threads'])!=len(set(s['call_threads']))):raise ValueError('Incomplete native graph/thread/hook receipt.')
    counts=(2,2) if state['local_tiny'] else (24,12)
    owners={r+'.'+str(i) for r,n in zip(('encoder','predictor'),counts) for i in range(n)}
    forward=Counter();back=Counter()
    for i,r in enumerate(rows):
        if (r.get('index')!=i or r.get('phase')!='EOT8-before' or r.get('pid')!=state['pid'] or r.get('owner') not in owners
                or r['owner']!=str(r.get('role'))+'.'+str(r.get('block')) or r.get('creator_thread')!=s['creator_thread']
                or r.get('entry_flags')!=r.get('exit_flags') or type(r.get('during_backward')) is not bool):raise ValueError('Native call identity/restoration changed.')
        if state['mode']=='predictor-math' and r['role']=='predictor':
            if r.get('effective_flags')!={'math':True,'flash':False,'cudnn':False,'mem_efficient':False}:raise ValueError('Predictor selection context changed.')
        elif r.get('effective_flags')!=r.get('entry_flags'):raise ValueError('Original/encoder call flags changed.')
        if r['during_backward']:
            if (r.get('graph_id')!=s['graph_id'] or r.get('backward_scope')!=0 or r.get('thread') not in s['call_threads']
                    or not(r.get('outcome')=='returned' or r.get('outcome')=='interrupted' and r.get('error_type')=='_StopRecomputationError')):raise ValueError('Unowned native recomputation.')
            back[r['owner']]+=1
        else:
            if r.get('graph_id')!=-1 or r.get('backward_scope') is not None or r.get('thread')!=s['creator_thread'] or r.get('outcome')!='returned':raise ValueError('Forward call changed.')
            forward[r['owner']]+=1
    if forward!=Counter({o:16 for o in owners}) or set(back)!=owners:raise ValueError('Full EOT8/two-family coverage required.')
    return {'kind':'controlled-VJP-one-phase-scope-v1','forward_calls':sum(forward.values()),'backward_calls':sum(back.values()),
            'scopes':1,'owners':len(owners),'hooks_removed':True,'backend_kernel_qualified':False,'full_five_phase_qualified':False}


def semantics(w):
    identity(w)
    if any(k in w for k in ('passed','complete','numerical_passed')):raise ValueError('Ambiguous native-passing flags forbidden in controlled records.')
    if w.get('native_loss_gradient') is not (w['path']==w['cotangent']):raise ValueError('Controlled gradient may not be labelled native loss.')
    if any(w.get(k) is not False for k in ('research_admitted','performance_qualified','backend_kernel_qualified','full_five_phase_qualified')):
        raise ValueError('Attribution cannot claim study/backend/performance qualification.')
    if w.get('attribution_complete'):
        if not all(w.get(k) is True for k in ('capture_complete','forward_reproduces_anchor','observer_restored','adapter_restored','model_unchanged')):
            raise ValueError('Incomplete controlled-VJP evidence.')
        status=w.get('native_numerical_passed');rejection=w.get('native_rejection')
        if w['index']==0:
            if status is not True or rejection is not None or not w.get('input_reproduces_anchor'):raise ValueError('Original diagonal must reproduce passing native values.')
        elif w['index']==1:
            if type(status) is not bool or not w.get('input_reproduces_anchor'):raise ValueError('Candidate diagonal must reproduce its native values.')
            if not w['local_tiny'] and status is not False:raise ValueError('Known target rejection must remain rejected.')
            if not status:
                if rejection!={'type':'ValueError','error':'Original numerical gate rejected; no tolerance change.','kind':'input'}:
                    raise ValueError('Only the preserved canonical native rejection is expected.')
            elif rejection is not None:raise ValueError('Unexpected rejection status.')
        elif status is not None or rejection is not None or w.get('input_reproduces_anchor') is not False:
            raise ValueError('Cross cell cannot claim native-gradient acceptance or exact diagonal reproduction.')


class ControlledRecord:
    def __init__(self,folder):
        self.folder,_=atomic._folder(folder)
        if any(os.path.lexists(self.folder/n) or os.path.lexists(self.folder/(n+'.partial')) for n in LIMITS):raise FileExistsError('Preserve previous controlled records.')
        self.previous={};self.row_hashes=[];self.closed=False;self.poisoned=False

    def write(self,state,route=None,*,final=False):
        if self.closed or self.poisoned:raise ValueError('Controlled record cannot be retried or reused.')
        try:return self._write(state,route,final)
        except BaseException:self.poisoned=True;raise

    def _write(self,state,route,final):
        semantics(state);key=identity(state)
        if any(n in state for n in atomic.INLINE):raise ValueError('External call/scope records required.')
        rows=[] if route is None else route.rows;scopes=[] if route is None else route.scopes;refusals=[] if route is None else route.refusals
        if len(rows)<len(self.row_hashes) or len(rows)>4096 or len(scopes)>1 or len(refusals)>32:raise ValueError('Controlled record extent changed.')
        for n,item in self.previous.items():
            if hashlib.sha256(atomic._read(self.folder,n,item['bytes'])).hexdigest()!=item['sha256']:raise ValueError('Prior record changed.')
        if state.get('attribution_complete'):
            if not final or verify_scope(state,rows,scopes,refusals)!=state.get('scope_check'):raise ValueError('Final attribution requires complete owned scope.')
        hashes=[]
        def lines():
            for i,row in enumerate(rows):
                data=atomic._json(row)
                if row.get('index')!=i or len(data)>2048:raise ValueError('Bounded ordered call rows required.')
                h=hashlib.sha256(data).hexdigest()
                if i<len(self.row_hashes) and h!=self.row_hashes[i]:raise ValueError('Earlier call changed.')
                hashes.append(h);yield data
        calls=atomic._atomic(self.folder,atomic.CALLS,lines(),LIMITS[atomic.CALLS]);calls['records']=len(rows)
        status='attribution-complete' if state.get('attribution_complete') else 'failure' if final else 'running'
        scope=atomic._atomic(self.folder,atomic.SCOPES,[atomic._json({'kind':FORMAT,'identity':key,'status':status,'calls':calls,'scopes':scopes,'refusals':refusals})],LIMITS[atomic.SCOPES])
        header=atomic._atomic(self.folder,atomic.HEADER,[atomic._json({**state,'record_format':FORMAT,'evidence':{'calls':calls,'scopes':scope,'status':status}})],LIMITS[atomic.HEADER])
        if final:
            marker={'kind':FORMAT,'identity':key,'status':status,'header':header}
            if state.get('native_rejection') is not None:
                _atomic(self.folder,REJECTION,[atomic._json({**marker,'status':'native-numerical-rejection','error':state['native_rejection']})],LIMITS[REJECTION])
            _atomic(self.folder,COMPLETE if state.get('attribution_complete') else FAILURE,[atomic._json(marker)],atomic.MAX_MARKER_BYTES)
        self.previous={v['file']:v for v in (calls,scope,header)};self.row_hashes=hashes;self.closed=final

    def failure(self,state,error):
        self.poisoned=True
        value={'kind':FORMAT,'identity':identity(state),'status':'writer-error','original_error':str(state.get('error',''))[:2048],
               'error_type':type(error).__name__,'error':str(error)[:2048]}
        return _atomic(self.folder,FAILURE,[atomic._json(value)],atomic.MAX_MARKER_BYTES)


def load(folder,*,control_sha256,require_complete=True):
    folder=Path(folder);raw=atomic._read(folder,atomic.HEADER,atomic.MAX_HEADER_BYTES);w=atomic._parse(raw);semantics(w);key=identity(w)
    if key['control_sha256']!=control_sha256 or w.get('record_format')!=FORMAT or any(n in w for n in atomic.INLINE):raise ValueError('Controlled identity/format changed.')
    done=w.get('attribution_complete') is True;name=COMPLETE if done else FAILURE
    if done and os.path.lexists(folder/FAILURE):raise ValueError('Failed attribution cannot be complete.')
    marker=atomic._parse(_read(folder,name,atomic.MAX_MARKER_BYTES));status='attribution-complete' if done else 'failure'
    expected={'kind':FORMAT,'identity':key,'status':status,'header':{'file':atomic.HEADER,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}
    if marker!=expected:raise ValueError('Controlled final marker changed.')
    if require_complete and not done:raise ValueError('Complete attribution record required; not a passing-native reference.')
    if w.get('native_rejection') is not None:
        rejected=atomic._parse(_read(folder,REJECTION,atomic.MAX_MARKER_BYTES))
        if rejected!={**expected,'status':'native-numerical-rejection','error':w['native_rejection']}:raise ValueError('Native rejection marker changed.')
    elif os.path.lexists(folder/REJECTION):raise ValueError('Unexpected native rejection marker.')
    evidence=w.get('evidence',{});calls=atomic._verified_file(folder,evidence.get('calls'),atomic.CALLS,atomic.MAX_CALL_BYTES)
    scope=atomic._parse(atomic._verified_file(folder,evidence.get('scopes'),atomic.SCOPES,atomic.MAX_SCOPE_BYTES))
    if scope.get('identity')!=key or scope.get('kind')!=FORMAT or scope.get('status')!=status or scope.get('calls')!=evidence.get('calls') or evidence.get('status')!=status:
        raise ValueError('Controlled scope binding changed.')
    lines=calls.splitlines(keepends=True)
    if (calls and not calls.endswith(b'\n')) or len(lines)!=evidence['calls']['records'] or len(lines)>4096:raise ValueError('Call extent changed.')
    rows=[]
    for i,line in enumerate(lines):
        row=atomic._parse(line)
        if len(line)>2048 or row.get('index')!=i:raise ValueError('Call row/order changed.')
        rows.append(row)
    result={**w,'call_rows':rows,'backward_scopes':scope['scopes'],'ownership_refusals':scope['refusals']}
    if done and verify_scope(w,rows,scope['scopes'],scope['refusals'])!=w.get('scope_check'):raise ValueError('Recorded scope changed.')
    return result

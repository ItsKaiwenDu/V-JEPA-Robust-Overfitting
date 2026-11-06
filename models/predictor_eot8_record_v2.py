"""Bounded, atomic external ownership evidence; no model or transport execution."""
import hashlib
import json
import os
from pathlib import Path
import stat

from models.predictor_eot8_owned_receipt_v1 import verify_ownership

FORMAT='predictor-eot8-split-record-v2'
CALLS='attention-calls.jsonl';SCOPES='attention-scopes.json'
HEADER='worker.json';COMPLETE='worker-complete.json';FAILURE='worker-failure.json'
MAX_ROWS=4096;MAX_ROW_BYTES=2048;MAX_CALL_BYTES=MAX_ROWS*MAX_ROW_BYTES
MAX_SCOPES=5;MAX_REFUSALS=32;MAX_SCOPE_BYTES=64*2**10
MAX_HEADER_BYTES=256*2**10;MAX_MARKER_BYTES=16*2**10
INLINE=('call_rows','backward_scopes','ownership_refusals')
IDENTITY=('kind','execution_host','local_tiny_analogue','mode','replay','index','pid','control_sha256')


def _json(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def _short(value,limit):
    text=str(value)[:2048]
    while len(_json(text))>limit:text=text[:len(text)//2]
    return text


def _parse(data):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('Duplicate evidence JSON key.')
            result[k]=v
        return result
    def bad(value):raise ValueError('Nonfinite evidence JSON value: '+value)
    return json.loads(data,object_pairs_hook=pairs,parse_constant=bad)


def _folder(folder):
    p=Path(folder);s=p.lstat()
    if not stat.S_ISDIR(s.st_mode) or p.is_symlink() or s.st_uid!=os.getuid():
        raise ValueError('Evidence directory ownership/type changed.')
    return p,s


def _read(folder,name,limit):
    if name not in (CALLS,SCOPES,HEADER,COMPLETE,FAILURE):raise ValueError('Fixed evidence filename required.')
    folder,parent=_folder(folder);path=folder/name;before=path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_uid!=os.getuid()
            or before.st_dev!=parent.st_dev or not 0<=before.st_size<=limit):
        raise ValueError('Evidence regular-file/size/ownership boundary.')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        actual=os.fstat(f.fileno())
        if (actual.st_dev,actual.st_ino)!=(before.st_dev,before.st_ino):raise ValueError('Evidence changed while opening.')
        data=f.read(limit+1);after=os.fstat(f.fileno())
    if len(data)>limit or (actual.st_size,actual.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or len(data)!=after.st_size:
        raise ValueError('Evidence changed during bounded read.')
    return data


def _sync_directory(folder):
    descriptor=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def _atomic(folder,name,chunks,limit):
    folder,parent=_folder(folder);path=folder/name;partial=folder/(name+'.partial')
    if os.path.lexists(path):_read(folder,name,limit)
    size=0;digest=hashlib.sha256()
    # An interrupted partial is preserved and refuses reuse; never truncate it.
    with os.fdopen(os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:
        for chunk in chunks:
            if not isinstance(chunk,bytes) or size+len(chunk)>limit:raise ValueError('Evidence file byte ceiling.')
            f.write(chunk);size+=len(chunk);digest.update(chunk)
        f.flush();os.fsync(f.fileno())
    current=folder.lstat()
    if (current.st_dev,current.st_ino)!=(parent.st_dev,parent.st_ino):raise ValueError('Evidence directory changed during write.')
    os.replace(partial,path);_sync_directory(folder)
    return {'file':name,'bytes':size,'sha256':digest.hexdigest()}


def _identity(state):
    result={k:state.get(k) for k in IDENTITY}
    if (result['kind']!='predictor-clean-eot8-owned-worker-v1' or type(result['local_tiny_analogue']) is not bool
            or result['mode'] not in ('original','predictor-math') or type(result['index']) is not int or result['index'] not in range(4)
            or type(result['replay']) is not int or result['replay']!=result['index']//2
            or result['mode']!=('original' if result['index']%2==0 else 'predictor-math')
            or type(result['pid']) is not int or not 0<result['pid']<2**63
            or not isinstance(result['execution_host'],str) or not 1<=len(result['execution_host'])<=255
            or result['control_sha256'] is not None and (not isinstance(result['control_sha256'],str) or len(result['control_sha256'])>128)):
        raise ValueError('Worker evidence identity changed.')
    return result


def _status(state,final):
    if type(state.get('complete')) is not bool or type(state.get('passed')) is not bool or state['complete']!=state['passed']:
        raise ValueError('Explicit matching worker completion flags required.')
    if state['passed'] and not final:raise ValueError('Success is allowed only in the final commit.')
    return 'success' if state['passed'] else 'failure' if final else 'running'


class WorkerRecord:
    """Snapshot bounded call rows separately; commit a small header last."""
    def __init__(self,folder):
        self.folder,_=_folder(folder)
        if any(os.path.lexists(self.folder/n) or os.path.lexists(self.folder/(n+'.partial')) for n in (CALLS,SCOPES,HEADER,COMPLETE,FAILURE)):
            raise FileExistsError('Preserve existing worker evidence; no writer reuse.')
        self.identity=None;self.previous={};self.row_hashes=[];self.closed=False;self.poisoned=False

    def write(self,state,*,route=None,final=False):
        if self.closed or self.poisoned:raise ValueError('Evidence writer closed or failed; no retry.')
        try:return self._write(state,route,final)
        except BaseException:
            self.poisoned=True
            raise

    def _write(self,state,route,final):
        if any(k in state for k in INLINE):raise ValueError('Inline ownership rows are forbidden in the small worker header.')
        identity=_identity(state);status=_status(state,final)
        if self.identity is not None and identity!=self.identity:raise ValueError('Writer identity cannot change.')
        for name,item in self.previous.items():
            data=_read(self.folder,name,item['bytes'])
            if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Earlier evidence snapshot changed.')
        rows=[] if route is None else route.rows
        scopes=[] if route is None else route.scopes
        refusals=[] if route is None else route.refusals
        if (type(rows) is not list or len(rows)>MAX_ROWS or len(rows)<len(self.row_hashes)
                or type(scopes) is not list or len(scopes)>MAX_SCOPES or type(refusals) is not list or len(refusals)>MAX_REFUSALS):
            raise ValueError('Ownership record count ceiling or prefix changed.')
        if any(type(x) is not dict for x in (*scopes,*refusals)):raise ValueError('Scope/refusal record type.')
        if status=='success':
            if state.get('synthetic_capacity_fixture') or route is None or route.active or state.get('adapter_restored') is not True:
                raise ValueError('Final success requires restored actual worker evidence.')
            view={**state,'call_rows':rows,'backward_scopes':scopes,'ownership_refusals':refusals}
            if verify_ownership(view)!=state.get('ownership_check'):raise ValueError('Final ownership validation changed.')
        hashes=[]
        def lines():
            for i,row in enumerate(rows):
                if type(row) is not dict or type(row.get('index')) is not int or row['index']!=i:raise ValueError('Ordered call-row index required.')
                line=_json(row)
                if len(line)>MAX_ROW_BYTES:raise ValueError('Call-row byte ceiling.')
                digest=hashlib.sha256(line).hexdigest()
                if i<len(self.row_hashes) and digest!=self.row_hashes[i]:raise ValueError('Previously saved call row changed.')
                hashes.append(digest);yield line
        calls=_atomic(self.folder,CALLS,lines(),MAX_CALL_BYTES);calls['records']=len(rows)
        body={'kind':FORMAT,'identity':identity,'status':status,'route_present':route is not None,
              'calls':calls,'scopes':scopes,'refusals':refusals}
        scope=_atomic(self.folder,SCOPES,[_json(body)],MAX_SCOPE_BYTES)
        evidence={'kind':FORMAT,'status':status,'calls':calls,'scopes':scope,'route_present':route is not None}
        compact={**state,'record_format':FORMAT,'evidence':evidence}
        header=_atomic(self.folder,HEADER,[_json(compact)],MAX_HEADER_BYTES)
        if final:
            name=COMPLETE if status=='success' else FAILURE
            _atomic(self.folder,name,[_json({'kind':FORMAT,'status':status,'identity':identity,'header':header})],MAX_MARKER_BYTES)
        self.identity=identity;self.row_hashes=hashes;self.previous={x['file']:x for x in (calls,scope,header)}
        self.closed=final
        return compact

    def failure(self,state,error):
        """A separate small refusal marker never overwrites the last safe snapshot."""
        self.poisoned=True
        marker={'kind':FORMAT,'status':'writer-error','identity':_identity(state),'complete':False,'passed':False,
                'error_type':_short(type(error).__name__,128),'error':_short(error,2048),
                'original_error_type':_short(state.get('error_type',''),128),'original_error':_short(state.get('error',''),2048),
                'previous_snapshot_is_not_a_success_receipt':True}
        return _atomic(self.folder,FAILURE,[_json(marker)],MAX_MARKER_BYTES)


def _verified_file(folder,item,name,limit):
    if (not isinstance(item,dict) or item.get('file')!=name or type(item.get('bytes')) is not int
            or not 0<=item['bytes']<=limit or not isinstance(item.get('sha256'),str) or len(item['sha256'])!=64):
        raise ValueError('Bounded evidence file descriptor required.')
    data=_read(folder,name,limit)
    if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Evidence hash/extent mismatch.')
    return data


def load_worker(folder,*,require_passed=True,expected_control=None):
    """Hydrate verified metadata only; never infer or repair a missing receipt."""
    folder=Path(folder);header=_read(folder,HEADER,MAX_HEADER_BYTES);state=_parse(header)
    if (not isinstance(state,dict) or state.get('record_format')!=FORMAT or any(k in state for k in INLINE)):
        raise ValueError('Separate bounded worker record required; legacy/inline records cannot substitute.')
    identity=_identity(state)
    if expected_control is not None and identity['control_sha256']!=expected_control:raise ValueError('Worker control binding changed.')
    if require_passed:
        if state.get('complete') is not True or state.get('passed') is not True or state.get('synthetic_capacity_fixture') or os.path.lexists(folder/FAILURE):
            raise ValueError('Complete passing actual worker required; failed/capacity evidence is not a reference.')
        marker=_parse(_read(folder,COMPLETE,MAX_MARKER_BYTES))
        if marker.get('kind')!=FORMAT or marker.get('status')!='success' or marker.get('identity')!=identity:
            raise ValueError('Missing final worker completion binding.')
        descriptor={'file':HEADER,'bytes':len(header),'sha256':hashlib.sha256(header).hexdigest()}
        if marker.get('header')!=descriptor:raise ValueError('Final completion marker does not bind this worker header.')
    evidence=state.get('evidence')
    if not isinstance(evidence,dict) or evidence.get('kind')!=FORMAT:raise ValueError('Missing external ownership receipt.')
    if require_passed and (evidence.get('status')!='success' or evidence.get('route_present') is not True):raise ValueError('Ownership snapshot not complete.')
    call_data=_verified_file(folder,evidence.get('calls'),CALLS,MAX_CALL_BYTES)
    body=_parse(_verified_file(folder,evidence.get('scopes'),SCOPES,MAX_SCOPE_BYTES))
    if (not isinstance(body,dict) or body.get('kind')!=FORMAT or body.get('identity')!=identity
            or body.get('status')!=evidence.get('status') or body.get('calls')!=evidence.get('calls')
            or type(body.get('route_present')) is not bool or body['route_present']!=evidence.get('route_present')):
        raise ValueError('Ownership snapshot identity/status changed.')
    if call_data and not call_data.endswith(b'\n'):raise ValueError('Incomplete call-row line.')
    lines=call_data.splitlines(keepends=True)
    if len(lines)>MAX_ROWS or type(evidence['calls'].get('records')) is not int or len(lines)!=evidence['calls']['records']:
        raise ValueError('Call-row count changed.')
    rows=[]
    for i,line in enumerate(lines):
        if len(line)>MAX_ROW_BYTES:raise ValueError('Call-row byte ceiling.')
        row=_parse(line)
        if type(row) is not dict or type(row.get('index')) is not int or row['index']!=i:raise ValueError('Call-row order changed.')
        rows.append(row)
    scopes=body.get('scopes');refusals=body.get('refusals')
    if (type(scopes) is not list or len(scopes)>MAX_SCOPES or type(refusals) is not list or len(refusals)>MAX_REFUSALS
            or any(type(x) is not dict for x in (*scopes,*refusals))):raise ValueError('Scope/refusal count/type ceiling.')
    view={**state,'call_rows':rows,'backward_scopes':scopes,'ownership_refusals':refusals}
    if require_passed:
        if state.get('adapter_restored') is not True or verify_ownership(view)!=state.get('ownership_check'):
            raise ValueError('Restoration/ownership qualification missing.')
    return view


def storage_projection():
    # Four serial worker directories plus one active atomic snapshot copy.
    maximum=5*(MAX_CALL_BYTES+MAX_SCOPE_BYTES+MAX_HEADER_BYTES)+4*2*MAX_MARKER_BYTES
    return {'four_workers_plus_one_atomic_copy_bytes':maximum,'existing_metadata_category_bytes':64*2**20,
            'metadata_category_margin_before_other_metadata_bytes':64*2**20-maximum,
            'global_operational_JSON_limit_unchanged':2**20,'new_worker_header_limit':MAX_HEADER_BYTES,
            'all_other_metadata_and_allocated_blocks_still_require_existing_aggregate_gate':True}

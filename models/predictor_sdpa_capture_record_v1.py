"""Separate bounded native SDPA observations; never passing-native worker records."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat

KIND='original-SDPA-one-boundary-observation-v1'
TENSORS=('q','k','v','output','grad_output','dq','dk','dv')
MiB=2**20
LIMITS={'boundary.json':128*1024,'boundary-complete.json':16*1024,'boundary-failed.json':16*1024,
        'attention-calls.jsonl':8*MiB,'attention-scopes.json':64*1024,**{n+'.pt':16*MiB for n in TENSORS}}


def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def folder_identity(folder):
    folder=Path(folder);s=folder.lstat()
    if not stat.S_ISDIR(s.st_mode) or s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)!=0o700:
        raise ValueError('Private regular observation directory required.')
    return folder,(s.st_dev,s.st_ino)


def read_file(folder,name):
    if name not in LIMITS:raise ValueError('Fixed observation filename required.')
    folder,identity=folder_identity(folder);p=folder/name;s=p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=os.getuid() or s.st_dev!=identity[0] or not 0<=s.st_size<=LIMITS[name]:
        raise ValueError('Observation file type/ownership/extent changed.')
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        before=os.fstat(f.fileno());data=f.read(LIMITS[name]+1);after=os.fstat(f.fileno())
    if (before.st_dev,before.st_ino)!=(s.st_dev,s.st_ino) or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or len(data)!=s.st_size:
        raise ValueError('Observation file changed during read.')
    return data


def scope_check(state,rows,scopes,refusals):
    if state.get('kind')!=KIND or type(state.get('local_tiny')) is not bool or state.get('mode')!='original':
        raise ValueError('Native observation identity required.')
    if refusals or len(scopes)!=1 or not 0<len(rows)<=4096:raise ValueError('One complete native GraphTask required.')
    s=scopes[0];pid=state['pid'];creator=state['creator_thread']
    if (s.get('phase')!='EOT8-before' or s.get('index')!=0 or s.get('state')!='complete' or s.get('hook_removed') is not True
            or s.get('pid')!=pid or s.get('creator_thread')!=creator or type(s.get('graph_id')) is not int or s['graph_id']<0
            or any(type(s.get(k)) is not int or s[k]<=0 for k in ('runner_thread','root_hook_thread','loss_id','input_id'))
            or not isinstance(s.get('call_threads'),list) or not 1<=len(s['call_threads'])<=8
            or len(set(s['call_threads']))!=len(s['call_threads']) or any(type(t) is not int or t<=0 for t in s['call_threads'])):
        raise ValueError('Incomplete actual graph/root/thread receipt.')
    counts=(2,2) if state['local_tiny'] else (24,12)
    owners={role+'.'+str(i) for role,n in zip(('encoder','predictor'),counts) for i in range(n)}
    forward=Counter();backward=Counter()
    for index,row in enumerate(rows):
        if (row.get('index')!=index or row.get('phase')!='EOT8-before' or row.get('pid')!=pid or row.get('creator_thread')!=creator
                or row.get('owner') not in owners or row['owner']!=str(row.get('role'))+'.'+str(row.get('block'))
                or row.get('entry_flags')!=row.get('exit_flags') or row.get('effective_flags')!=row.get('entry_flags')
                or type(row.get('during_backward')) is not bool):raise ValueError('Native call identity/flags/order changed.')
        if row['during_backward']:
            if (row.get('graph_id')!=s['graph_id'] or row.get('backward_scope')!=0 or row.get('thread') not in s['call_threads']
                    or not(row.get('outcome')=='returned' or row.get('outcome')=='interrupted' and row.get('error_type')=='_StopRecomputationError')):
                raise ValueError('Unowned native attention recomputation.')
            backward[row['owner']]+=1
        else:
            if row.get('graph_id')!=-1 or row.get('backward_scope') is not None or row.get('thread')!=creator or row.get('outcome')!='returned':
                raise ValueError('Original forward domain changed.')
            forward[row['owner']]+=1
    if forward!=Counter({n:16 for n in owners}) or set(backward)!=owners:
        raise ValueError('All original EOT8 draws/families/owners required.')
    return {'kind':'native-SDPA-observation-one-GraphTask-coverage-v1','forward_calls':sum(forward.values()),
        'backward_calls':sum(backward.values()),'owners':len(owners),'scopes':1,'hooks_removed':True,
        'CUDA_kernel_qualified':False,'five_phase_qualified':False}


class BoundaryStore:
    def __init__(self,folder):
        self.folder,self.identity=folder_identity(folder);self.closed=False;self.poisoned=False;self.files={}
        if any(os.path.lexists(self.folder/n) or os.path.lexists(self.folder/(n+'.partial')) for n in LIMITS):
            raise FileExistsError('Never reuse original boundary evidence.')

    def _check(self):
        if folder_identity(self.folder)[1]!=self.identity:raise ValueError('Observation directory replaced.')
        for name,item in self.files.items():
            data=read_file(self.folder,name)
            if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
                raise ValueError('Earlier observation bytes changed.')

    def _write(self,name,write):
        if self.closed or self.poisoned or name not in LIMITS:raise ValueError('Closed/failed or unknown native observation record.')
        self._check();path=self.folder/name;partial=self.folder/(name+'.partial')
        if name not in self.files and os.path.lexists(path):raise FileExistsError('Preserve an unexpected existing observation file.')
        try:
            with os.fdopen(os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as f:
                write(f)
                if f.tell()>LIMITS[name]:raise ValueError('Observation serialization byte ceiling.')
                f.flush();os.fsync(f.fileno())
            self._check()
            if name not in self.files and os.path.lexists(path):raise FileExistsError('Observation destination appeared during publication.')
            os.replace(partial,path)
            fd=os.open(self.folder,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            item={'file':name,'bytes':path.stat().st_size,'sha256':digest(path)};self.files[name]=item
            return dict(item)
        except BaseException:self.poisoned=True;raise

    def save(self,name,value):
        import torch
        if name not in TENSORS or name+'.pt' in self.files:raise ValueError('Each actual boundary tensor saved once.')
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.ndim!=4 or value.numel()*4+4096>16*MiB:
            raise ValueError('Bounded actual FP32 boundary tensor required.')
        from models.predictor_eot8_capture_v1 import tensor_info
        source=tensor_info(value);copy=value.detach().cpu().contiguous().clone()
        item=self._write(name+'.pt',lambda f:torch.save(copy,f))
        item.update(source=source,stored_stride=list(copy.stride()),stored_device='cpu',logical_copy_is_contiguous=True,
            native_storage_alias_preserved_in_file=False,saved_before_numerical_gate=True)
        return item

    def publish(self,state):
        return self._write('boundary.json',lambda f:f.write(encoded(state)))

    def finish(self,state,route):
        if self.closed:raise ValueError('Observation finalized once.')
        rows=[encoded(r) for r in route.rows]
        if any(len(line)>2048 for line in rows):raise ValueError('Oversize native call row.')
        calls=self._write('attention-calls.jsonl',lambda f:[f.write(line) for line in rows])
        calls['records']=len(rows)
        scopes=self._write('attention-scopes.json',lambda f:f.write(encoded({'scopes':route.scopes,'refusals':route.refusals})))
        state['evidence']={'calls':calls,'scopes':scopes}
        header=self.publish(state);name='boundary-complete.json' if state.get('capture_complete') else 'boundary-failed.json'
        self._write(name,lambda f:f.write(encoded({'kind':KIND,'control_sha256':state['control_sha256'],'header':header,'capture_complete':state['capture_complete']})))
        self.closed=True


def load(folder,*,control_sha256,require_complete=True):
    folder=Path(folder);raw=read_file(folder,'boundary.json');w=json.loads(raw)
    if (w.get('kind')!=KIND or w.get('control_sha256')!=control_sha256 or type(w.get('local_tiny')) is not bool
            or w.get('mode')!='original' or w.get('native_cotangent_replaced') is not False or w.get('native_forward_modified') is not False
            or any(w.get(k) is not False for k in ('research_admitted','performance_qualified','CUDA_kernel_qualified','native_full_model_qualified'))):
        raise ValueError('Original observational identity and no-adoption semantics required.')
    if type(w.get('capture_complete')) is not bool:raise ValueError('Typed native completion status required.')
    done=w['capture_complete'];name='boundary-complete.json' if done else 'boundary-failed.json'
    marker=json.loads(read_file(folder,name))
    if marker!={'kind':KIND,'control_sha256':control_sha256,'header':{'file':'boundary.json','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},'capture_complete':done}:
        raise ValueError('Actual boundary final marker mismatch.')
    if done and os.path.lexists(folder/'boundary-failed.json'):raise ValueError('Failed boundary cannot qualify completion.')
    evidence=w['evidence']
    def verified(item):
        data=read_file(folder,item['file'])
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Bound native observation artifact changed.')
        return data
    rows=[json.loads(line) for line in verified(evidence['calls']).splitlines()]
    if len(rows)!=evidence['calls']['records']:raise ValueError('Call record extent changed.')
    graph=json.loads(verified(evidence['scopes']))
    for key,item in w['artifacts'].items():
        if key not in TENSORS or item['file']!=key+'.pt' or not item.get('saved_before_numerical_gate'):raise ValueError('Actual fixed tensor artifact identity required.')
        verified(item)
    if require_complete and not done:raise ValueError('Complete native boundary required; partial evidence is not a reference.')
    if done:
        if set(w['artifacts'])!=set(TENSORS) or w.get('observer_restored') is not True or w.get('alias_restored') is not True or w.get('tensor_versions_unchanged') is not True:
            raise ValueError('Eight actual tensors, restoration and unchanged versions required.')
        if scope_check(w,rows,graph['scopes'],graph['refusals'])!=w['scope_check']:raise ValueError('Native owned scope changed.')
        scope=graph['scopes'][0]
        selected=w['selected'];row_index=2 if w['local_tiny'] else 24
        if (selected!={'owner':'predictor.0','draw':0,'family':0,'forward_row_index':row_index,'phase':'EOT8-before'}
                or w['selected_call']['actual_call']!=rows[row_index]
                or w['selected_call']['forward_graph_id']!=-1 or w['selected_call']['forward_thread']!=w['creator_thread']
                or w['selected_call']['row_index']!=row_index or w['selected_call']['pid']!=w['pid']):
            raise ValueError('Predetermined first actual forward selection changed.')
        shape=[1,2,32,32] if w['local_tiny'] else [1,12,2048,32]
        if w.get('finite')!=dict.fromkeys(TENSORS,True) or set(w['source_storage'])!=set(TENSORS):
            raise ValueError('Every actual native tensor must be finite and have storage metadata.')
        for n,item in w['artifacts'].items():
            s=item['source'];layout=w['source_storage'][n]
            if (s.get('shape')!=shape or s.get('dtype')!='torch.float32' or s.get('device')!=('cpu' if w['local_tiny'] else 'cuda:0')
                    or len(s.get('stride',[]))!=4 or any(type(x) is not int or x<=0 for x in s['stride'])
                    or item.get('stored_device')!='cpu' or item.get('logical_copy_is_contiguous') is not True
                    or item.get('native_storage_alias_preserved_in_file') is not False
                    or layout['source_logical_hash']!=s['sha256'] or type(layout['alias_group']) is not int or layout['alias_group']<0
                    or type(layout['storage_offset_elements']) is not int or layout['storage_offset_elements']<0
                    or type(layout['version']) is not int or layout['version']<0
                    or layout['requires_grad'] is not (n in ('q','k','v','output'))):
                raise ValueError('Typed source/saved-layout/storage identity changed.')
            if (type(layout['storage_bytes']) is not int or layout['storage_bytes']<=0 or layout['storage_bytes']%4
                    or any(type(layout[k]) is not int or not 0<=layout[k]<256 for k in ('data_address_mod_256','storage_address_mod_256'))
                    or layout['data_address_mod_256']!=(layout['storage_address_mod_256']+4*layout['storage_offset_elements'])%256):
                raise ValueError('Native backing extent or offset/alignment receipt changed.')
            span=4*(1+sum((a-1)*b for a,b in zip(shape,s['stride'])))
            if (span!=layout['positive_stride_span_bytes'] or span>16*MiB
                    or 4*layout['storage_offset_elements']+span>layout['storage_bytes']):
                raise ValueError('Native layout source span/offset exceeds backing storage.')
        groups={}
        for layout in w['source_storage'].values():
            key=layout['alias_group'];signature=(layout['storage_bytes'],layout['storage_address_mod_256'])
            if key in groups and groups[key]!=signature:raise ValueError('Source alias group backing identity changed.')
            groups[key]=signature
        if set(groups)!=set(range(len(groups))):raise ValueError('Bounded ordered source alias groups required.')
        for n,r in w['gradient_observations'].items():
            if n not in ('dq','dk','dv','grad_output') or r['graph_id']!=scope['graph_id'] or r['thread'] not in scope['call_threads'] or r['hook_removed'] is not True or r['pid']!=w['pid']:
                raise ValueError('Native gradient-hook GraphTask ownership changed.')
        if set(w['gradient_observations'])!=set(('dq','dk','dv','grad_output')):raise ValueError('One actual observation per native gradient required.')
    return {**w,'call_rows':rows,'backward_scopes':graph['scopes'],'ownership_refusals':graph['refusals']}


def load_tensor(folder,name,state):
    """Safe CPU deserialization/byte check only; no model or derivative evaluation."""
    import torch
    if name not in TENSORS:raise ValueError('Fixed actual tensor name required.')
    item=state['artifacts'][name];raw=read_file(folder,name+'.pt')
    if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Saved tensor file changed.')
    import io
    value=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
    if (not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=torch.float32
            or value.requires_grad or list(value.shape)!=item['source']['shape'] or list(value.stride())!=item['stored_stride']
            or not value.is_contiguous() or not torch.isfinite(value).all()):
        raise ValueError('Actual saved boundary tensor dtype/rank/layout/finite status changed.')
    actual=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    if actual!=item['source']['sha256']:raise ValueError('Actual native logical tensor bytes changed.')
    return value


def rebuild_cpu_inputs(folder,state):
    """Rebuild recorded layouts/offsets/alias groups; unknown padding is zeroed."""
    import torch
    from models.predictor_eot8_capture_v1 import tensor_info
    groups={};result={}
    for name in ('q','k','v','grad_output'):
        item=state['artifacts'][name];storage=state['source_storage'][name];s=item['source'];key=storage['alias_group']
        size=storage['storage_bytes']
        if type(size) is not int or size<=0 or size%4 or size>16*MiB:raise ValueError('Bounded recorded FP32 source backing required.')
        if key not in groups:groups[key]=torch.zeros(size//4,dtype=torch.float32,device='cpu')
        elif groups[key].numel()*4!=size:raise ValueError('Alias group backing extent changed.')
        value=groups[key].as_strided(s['shape'],s['stride'],storage['storage_offset_elements'])
        value.copy_(load_tensor(folder,name,state));result[name]=value
    # A later alias copy cannot silently alter any earlier logical operand.
    for name,value in result.items():
        got=tensor_info(value);want=state['artifacts'][name]['source']
        if {k:got[k] for k in ('shape','stride','dtype','sha256')}!={k:want[k] for k in ('shape','stride','dtype','sha256')}:
            raise ValueError('Rebuilt logical values/layout or overlapping aliases changed.')
    return result

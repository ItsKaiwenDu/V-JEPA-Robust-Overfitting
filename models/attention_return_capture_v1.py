"""Typed saved returns. Scientific imports occur only when a return is captured."""
import hashlib
import io
from pathlib import Path
import os
import zipfile

KEYS=('output','dq','dk','dv')
MIB=2**20


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def signature(array):
    import numpy as np
    if array.dtype not in (np.dtype('<f4'),np.dtype('<f8')) or not array.flags.c_contiguous:
        raise ValueError('Capture signatures require original contiguous FP32/FP64 logical values.')
    finite=bool(np.isfinite(array).all())
    return {'dtype':array.dtype.str,'shape':list(array.shape),'sha256':hashlib.sha256(array.tobytes()).hexdigest(),
        'finite':finite,'max_abs':float(np.abs(array).max()) if finite and array.size else None}


def save_return(path,values,*,dtype,device,shape,limit):
    import numpy as np
    import torch
    path=Path(path);partial=path.with_name(path.name+'.partial')
    expected={'float32':torch.float32,'float64':torch.float64}[dtype]
    if path.exists() or partial.exists():raise FileExistsError('Preserve existing return capture.')
    if set(values)!=set(KEYS):raise ValueError('Return must contain output and all three derivatives.')
    metadata={};arrays={}
    for name in KEYS:
        tensor=values[name]
        if not isinstance(tensor,torch.Tensor) or tensor.dtype!=expected or tensor.device.type!=device or list(tensor.shape)!=list(shape):
            raise ValueError('Return device, dtype or geometry differs; no fallback/cast.')
        metadata[name]={'dtype':str(tensor.dtype),'device':str(tensor.device),'shape':list(tensor.shape),'stride':list(tensor.stride())}
        arrays[name]=np.array(tensor.detach().cpu().numpy(),order='C',copy=True)
        if arrays[name].dtype!=np.dtype('<f4' if dtype=='float32' else '<f8'):raise ValueError('Return copy changed dtype.')
    if sum(a.nbytes for a in arrays.values())+65536>limit:raise ValueError('Return exceeds its declared capture bound.')
    # Persist nonfinite values too; numerical acceptance belongs to the original worker.
    with partial.open('xb') as f:
        np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    if partial.stat().st_size>limit:raise ValueError('Capture container exceeds bound; partial retained.')
    os.link(partial,path);partial.unlink()
    return {'file':path.name,'bytes':path.stat().st_size,'sha256':digest(path),'source_tensors':metadata,
        'arrays':{n:signature(a) for n,a in arrays.items()},'dtype_preserved':True}


def load_return(folder,item,*,dtype,shape,limit):
    import numpy as np
    path=Path(folder)/item['file']
    if (Path(item['file']).name!=item['file'] or path.is_symlink() or not path.is_file()
            or not 0<item['bytes']<=limit or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']):
        raise ValueError('Saved return path, size or hash differs.')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Return changed while being read.')
    expected=np.dtype('<f4' if dtype=='float32' else '<f8');result={}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries=archive.infolist()
        if len(entries)!=4 or {e.filename for e in entries}!={n+'.npy' for n in KEYS}:raise ValueError('Return array coverage differs.')
        for name in KEYS:
            entry=archive.getinfo(name+'.npy');payload=__import__('math').prod(shape)*expected.itemsize
            if not payload<=entry.file_size<=payload+1024:raise ValueError('Return array extent differs.')
            raw=archive.read(entry);header=io.BytesIO(raw);version=np.lib.format.read_magic(header)
            if version not in ((1,0),(2,0)):raise ValueError('Unsupported array format.')
            dims,fortran,kind=(np.lib.format.read_array_header_1_0(header) if version==(1,0) else np.lib.format.read_array_header_2_0(header))
            if list(dims)!=list(shape) or fortran or kind!=expected or len(raw)-header.tell()!=payload:
                raise ValueError('Return dtype/shape/header changed.')
            array=np.load(io.BytesIO(raw),allow_pickle=False)
            if signature(array)!=item['arrays'][name]:raise ValueError('Return logical signature differs.')
            result[name]=array
    return result


def audit_worker(folder,packet,*,local):
    """Re-read saved arrays, without any SDPA/oracle evaluation or device transfer."""
    import numpy as np
    import torch
    from models.oracle_boundary_observer_v1 import read_record
    from models.attention_saved_evidence import agreement,backend
    from models.attention_component import backend_evidence
    from models.cuda_profile import tensor_signature
    folder=Path(folder);state=read_record(folder/'capture.json');original=read_record(folder/'worker.json')
    manifest=__import__('json').loads((Path(packet)/'FIXED-ATTENTION-HANDOFF.json').read_text())
    fixture=read_record(Path(packet)/manifest['fixtures']['encoder-1']['folder']/'fixture.json')
    shape=fixture['files']['q']['tensor']['shape'];canonical={n:np.load(Path(packet)/'fixtures/encoder-1'/('oracle_'+n+'.npy'),allow_pickle=False) for n in KEYS}
    if not original['passed'] or not original['complete'] or original['stage']!='complete' or not original['within_process_exact']:
        raise ValueError('Original worker did not complete all of its checks.')
    if state['local_tiny_rehearsal'] is not local or state['CPU_oracle_calls']!=1 or state['CPU_oracle_returns']!=1:
        raise ValueError('Wrong capture CPU domain/count.')
    cpu=load_return(folder,state['CPU_return'],dtype='float64',shape=shape,limit=16*MIB)
    for name in KEYS:
        book=original['oracle_bookkeeping'][name]
        if book['rtol']!=1e-12 or book['atol']!=1e-14 or book['passed'] is not True:
            raise ValueError('Original CPU bounds/decision differs.')
        if not bool(torch.allclose(torch.from_numpy(cpu[name]),torch.from_numpy(canonical[name]),rtol=1e-12,atol=1e-14)):
            raise ValueError('Saved CPU return fails original gate.')
    phases=('untimed','profiled','warmup','timed-1','timed-2','timed-3');entries=state['attention_returns']
    if state['attention_calls']!=6 or state['SDPA_calls']!=6 or [x['phase'] for x in entries]!=list(phases):
        raise ValueError('Incomplete attention call/return coverage.')
    values_path=folder/'values.pt'
    if digest(values_path)!=original['artifact']['sha256']:raise ValueError('Original GPU value snapshot changed.')
    values=torch.load(values_path,map_location='cpu',weights_only=True)
    if {n:tensor_signature(t) for n,t in values.items()}!=original['artifact']['tensor_signatures']:
        raise ValueError('Original snapshot logical signatures differ.')
    if any(not np.array_equal(values['oracle_'+n].numpy(),canonical[n]) for n in KEYS):
        raise ValueError('Fixed canonical references differ from the original snapshot.')
    if set(original['evidence'])!={'inputs','device_inputs'}:raise ValueError('Original input snapshots missing.')
    for key,filename in [('inputs','inputs.pt'),('device_inputs','device-inputs.pt')]:
        snapshot=original['evidence'][key]
        if digest(folder/filename)!=snapshot['sha256']:raise ValueError('Original input snapshot changed.')
        tensors=torch.load(folder/filename,map_location='cpu',weights_only=True)
        if {n:tensor_signature(t) for n,t in tensors.items()}!=original['common']['operand_signatures']:
            raise ValueError('Transferred or CPU inputs changed.')
    first=None;comparison_checks=0
    for entry in entries:
        arrays=load_return(folder,entry['artifact'],dtype='float32',shape=shape,limit=8*MIB)
        signatures={n:tensor_signature(torch.from_numpy(a)) for n,a in arrays.items()}
        for name,array in arrays.items():
            if not np.isfinite(array).all() or not np.array_equal(array,values['actual_'+name].numpy()):
                raise ValueError('Actual attention return differs from original untimed values or replay.')
            result=agreement(array,canonical[name],fixture['tolerances'][name]);comparison_checks+=1
            if not result['passed'] or original['oracle_checks'][name]['passed'] is not True:
                raise ValueError('Saved attention return fails original component bounds.')
        if first is None:first=signatures
        elif signatures!=first:raise ValueError('Saved attention returns are not bitwise replay equal.')
        if entry['phase']!='untimed':
            record=next(x for x in original['replay_signatures'] if x['phase']==entry['phase'])
            if record['signatures']!=signatures:raise ValueError('Saved return differs from original phase signature.')
    if [x['phase'] for x in original['replay_signatures']]!=list(phases[1:]) or len(original['timings'])!=3:
        raise ValueError('Original replay/timing coverage missing.')
    observed=backend_evidence(original['backend_profile_events'],original['arm'],'cpu') if local else backend(original['backend_profile_events'],original['arm'])
    if observed!=original['backend_evidence'] or not observed['passed']:raise ValueError('Original actual backend evidence rejected.')
    auth=read_record(folder/'authorization.json');reference_checks=0
    if auth['reference'] is not None:
        ref=auth['reference'];other=folder.parent/ref['worker'];prior=read_record(other/'worker.json')
        if digest(other/'worker.json')!=ref['worker_sha256'] or digest(other/'values.pt')!=ref['values_sha256'] or not prior['passed'] or not prior['complete'] or prior['common']!=original['common']:
            raise ValueError('Fresh completed reference differs.')
        baseline=torch.load(other/'values.pt',map_location='cpu',weights_only=True)
        for name in KEYS:
            if not agreement(values['actual_'+name].numpy(),baseline['actual_'+name].numpy(),fixture['tolerances'][name])['passed']:
                raise ValueError('Saved candidate/reference values fail original bounds.')
            if original['reference_comparison'][name]['passed'] is not True:raise ValueError('Original reference gate rejected.')
            reference_checks+=1
    return {'kind':'original-attention-pair-saved-worker-audit-v1','saved_value_checks_passed':True,
        'original_worker_sha256':digest(folder/'worker.json'),'CPU_gate_checks':4,'component_array_checks':comparison_checks,
        'reference_array_checks':reference_checks,'attention_returns':6,'dtype_preserved':True,
        'original_backend_passed':True,'original_session_qualified':False,'performance_qualified':False,
        'research_admitted':False,'local_tiny_rehearsal':local}

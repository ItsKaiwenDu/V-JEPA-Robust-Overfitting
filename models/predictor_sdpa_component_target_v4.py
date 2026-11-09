"""Target plan/serial/resource wrapper; executor exclusively owns CUDA startup."""
import json,math,os,platform,stat,time
from pathlib import Path

def run_component(args):
    if args.device!='cuda' or platform.node()!='linux10603' or platform.python_version()!='3.11.17':raise ValueError('Exact authorized target component entry required.')
    from models import predictor_sdpa_component_v4 as core
    from models import predictor_sdpa_component_records_v1 as rec
    p=core.validate_plan(core.read(args.plan))
    if rec.sha(args.plan)!=args.plan_sha256 or p['local_tiny'] is not False or args.index not in range(4):raise ValueError('Bound target component plan changed.')
    parent,source,cap,identity=core.capture(p);prior=core.previous(p,args.plan_sha256,identity,args.index)
    ref=Path(p['reference_dir'])
    if args.index:
        core.load_reference(ref,p,args.plan_sha256,identity)
        first=core.load_worker(ref.parent/'0',p,args.plan_sha256,identity,0)
        if first['reference_manifest_sha256']!=rec.sha(ref/'reference.json'):raise ValueError('Original canonical reference changed.')
    elif ref.exists():raise FileExistsError('Never rebuild the canonical reference.')
    if Path(args.output)!=ref.parent/str(args.index):raise ValueError('Exact component output required.')
    import torch
    start=time.monotonic();w=core._execute(args,p,parent,source,cap,identity,prior)
    if w.get('complete') is not True or w.get('pid')!=os.getpid():raise ValueError('Completed own worker required before resource receipt.')
    from models.predictor_sdpa_component_startup_v2 import validate
    validate(w.get('startup'),pid=os.getpid(),device='cuda')
    peak=torch.cuda.max_memory_allocated()
    if type(peak) is not int or not 0<peak<=7*2**30 or time.time()>args.deadline_unix:raise ValueError('Component resource limit rejected after returns; preserve evidence.')
    from models.cpu_oracle_scratch import write_json
    previous_mask=os.umask(0o077)
    try:
        write_json(Path(args.output)/'component-resource.json',{'kind':'native-fixed-SDPA-component-resource-v4','plan_sha256':args.plan_sha256,
            'worker_sha256':rec.sha(Path(args.output)/'worker.json'),'pid':os.getpid(),'index':args.index,
            'peak_allocator_bytes':peak,'startup':w['startup'],'post_entry_wall_seconds':time.monotonic()-start,'research_admitted':False,'performance_qualified':False})
    finally:os.umask(previous_mask)
    load_resource(Path(args.output),plan_sha256=args.plan_sha256,index=args.index,worker=w)
    return w



def load_resource(path,*,plan_sha256,index,worker):
    """Bounded receipt readback; no CUDA, tensor or scientific work."""
    from models import predictor_sdpa_component_records_v1 as rec
    from models.predictor_sdpa_component_startup_v2 import validate
    path=Path(path);rec.folder(path);file=path/'component-resource.json';s=file.lstat()
    if (file.is_symlink() or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=os.getuid()
            or stat.S_IMODE(s.st_mode)!=0o600 or not 0<s.st_size<=16*1024):
        raise ValueError('Private bounded component resource receipt required.')
    with os.fdopen(os.open(file,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        before=os.fstat(f.fileno());data=f.read(16*1024+1);after=os.fstat(f.fileno())
    if ((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
            or (after.st_size,after.st_mtime_ns,after.st_nlink)!=(s.st_size,s.st_mtime_ns,1) or len(data)!=s.st_size):
        raise ValueError('Component resource receipt changed during read.')
    r=json.loads(data)
    if (not isinstance(r,dict) or set(r)!={'kind','plan_sha256','worker_sha256','pid','index','peak_allocator_bytes',
            'startup','post_entry_wall_seconds','research_admitted','performance_qualified'}
            or r['kind']!='native-fixed-SDPA-component-resource-v4' or r['plan_sha256']!=plan_sha256
            or type(index) is not int or index not in range(4) or type(r['index']) is not int or r['index']!=index
            or worker.get('complete') is not True or worker.get('index')!=index or worker.get('plan_sha256')!=plan_sha256
            or type(r['pid']) is not int or r['pid']<=0 or r['pid']!=worker.get('pid')
            or r['worker_sha256']!=rec.sha(path/'worker.json') or r['startup']!=worker.get('startup')
            or type(r['peak_allocator_bytes']) is not int or not 0<r['peak_allocator_bytes']<=7*2**30
            or type(r['post_entry_wall_seconds']) not in (int,float) or not math.isfinite(r['post_entry_wall_seconds'])
            or not 0<=r['post_entry_wall_seconds']<=120 or r['research_admitted'] is not False or r['performance_qualified'] is not False):
        raise ValueError('Exact completed-worker resource receipt required.')
    validate(r['startup'],pid=r['pid'],device='cuda')
    return r

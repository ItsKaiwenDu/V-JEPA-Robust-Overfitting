"""Target plan/serial/resource wrapper; executor exclusively owns CUDA startup."""
import os,platform,time
from pathlib import Path

def run_component(args):
    if args.device!='cuda' or platform.node()!='linux10603' or platform.python_version()!='3.11.17':raise ValueError('Exact authorized target component entry required.')
    from models import predictor_sdpa_component_v3 as core
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
    if not 0<peak<=7*2**30 or time.time()>args.deadline_unix:raise ValueError('Component resource limit rejected after returns; preserve evidence.')
    from models.cpu_oracle_scratch import write_json
    write_json(Path(args.output)/'component-resource.json',{'kind':'native-fixed-SDPA-component-resource-v3','plan_sha256':args.plan_sha256,
        'worker_sha256':rec.sha(Path(args.output)/'worker.json'),'pid':os.getpid(),'index':args.index,
        'peak_allocator_bytes':peak,'startup':w['startup'],'post_entry_wall_seconds':time.monotonic()-start,'research_admitted':False,'performance_qualified':False})
    return w


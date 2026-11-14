"""Separate saved-metadata contract for a narrowly recognized profiler overhead.

Retains every raw row and original index. No scientific or profiler work starts.
The frozen v1 schema/validator and historical refusals remain unchanged. Native
activity kind is not exposed by the pinned Python API; classification is an
explicit, source-supported signature, never an authenticated native-kind witness.
"""
import hashlib
import math
from models import predictor_sdpa_launch_evidence_v1 as frozen

KIND=frozen.KIND
API=frozen.API
POLICY='fixed-SDPA-profiler-overhead-signature-v1'
MAX_EVENTS=frozen.MAX_EVENTS
MAX_BYTES=frozen.MAX_BYTES
MAX_INTEGER=frozen.MAX_INTEGER
FIELDS=frozen.FIELDS
FAMILIES=dict(frozen.FAMILIES)
FORWARD=frozen.FORWARD
BACKWARD=frozen.BACKWARD
associated_regions=frozen.associated_regions
encoded=frozen.encoded
OVERHEAD_NAME='Activity Buffer Request'
OVERHEAD_RUNTIME='2.8.0+cu128'
KINETO_PIN='5e7501833f1021ce6f618572d3baf657b6319658'


def record_role(row,*,torch_version):
    # Called only after exact schema, field types, ranges and clocks validate.
    if row['device_index']!=-1:
        if row['device']=='cuda' and row['user_annotation']:
            if (torch_version!=OVERHEAD_RUNTIME or row['name'] not in (FORWARD,BACKWARD)
                    or row['device_index']!=0 or row['linked_correlation_id']!=0 or row['correlation_id']<=0
                    or row['is_async'] or row['thread']!=1 or row['end_thread']!=1
                    or row['fwd_thread']!=0 or row['sequence_nr']!=-1 or row['scope']!=0
                    or row['legacy_cuda_elapsed_us']!=-1):
                raise ValueError('Unknown or malformed GPU user range; never treat annotations as kernels.')
            return 'profiler-GPU-range'
        if row['device']=='cuda':return 'CUDA-activity'
        return 'frontend-CPU' if row['linked_correlation_id']==0 else 'linked-runtime'
    if (torch_version!=OVERHEAD_RUNTIME or row['device']!='cpu' or row['name']!=OVERHEAD_NAME
            or row['resource_id']!=0 or row['thread']!=1 or row['end_thread']!=1
            or row['fwd_thread']!=0 or row['sequence_nr']!=-1 or row['scope']!=0
            or row['linked_correlation_id']!=0 or row['correlation_id']<=0
            or row['is_async'] or row['user_annotation'] or row['legacy_cuda_elapsed_us']!=-1):
        raise ValueError('Unknown or malformed negative-device profiler record; no name-only exclusion.')
    return 'profiler-overhead-signature'


def project(trace):
    """Validate raw rows and derive a separate CPU tree and exact device links.

    Matches v2.8 linked-ID ownership and CPU containment, retaining duplicate
    CPU nodes rather than lifting kernels from a deleted owner. No stored row is
    relabeled. CUDA streams, intervals and IDs remain recorded provenance.
    """
    if (not isinstance(trace,dict) or set(trace)!={'kind','source_api','torch_version','trace_start_ns','records'}
            or trace.get('kind')!=KIND or trace.get('source_api')!=API
            or not isinstance(trace.get('torch_version'),str) or not 0<len(trace['torch_version'])<=128
            or type(trace.get('trace_start_ns')) is not int or not 0<=trace['trace_start_ns']<=MAX_INTEGER
            or not isinstance(trace.get('records'),list) or not 0<len(trace['records'])<=MAX_EVENTS):
        raise ValueError('Exact raw correlation schema required; old traces have no recoverable IDs.')
    raw=trace['records'];origin=trace['trace_start_ns'];seen=set();owners={};roles=[]
    for i,r in enumerate(raw):
        if not isinstance(r,dict) or set(r)!=FIELDS or type(r['index']) is not int or r['index']!=i:
            raise ValueError('Complete ordered typed raw records required.')
        if (not isinstance(r['name'],str) or not 0<len(r['name'])<=4096 or r['device'] not in ('cpu','cuda')
                or type(r['device_index']) is not int or not -1<=r['device_index']<=MAX_INTEGER
                or any(type(r[k]) is not int or not 0<=r[k]<=MAX_INTEGER for k in
                    ('resource_id','thread','end_thread','fwd_thread','scope','start_ns','end_ns','correlation_id','linked_correlation_id'))
                or type(r['sequence_nr']) is not int or not -1<=r['sequence_nr']<=MAX_INTEGER
                or any(type(r[k]) is not bool for k in ('is_async','user_annotation'))
                or type(r['legacy_cuda_elapsed_us']) not in (int,float) or not math.isfinite(r['legacy_cuda_elapsed_us'])
                or r['legacy_cuda_elapsed_us']>0 or not origin<=r['start_ns']<=r['end_ns']):
            raise ValueError('Invalid raw event fields or legacy CUDA fallback.')
        role=record_role(r,torch_version=trace['torch_version']);roles.append(role)
        identity=tuple(r[k] for k in ('device','correlation_id','linked_correlation_id','device_index','resource_id','thread','start_ns','end_ns','name'))
        if identity in seen:raise ValueError('Duplicate raw event identity; preserve and refuse.')
        seen.add(identity)
        if role!='profiler-overhead-signature' and r['device']=='cpu' and r['linked_correlation_id']==0:
            key=r['correlation_id']
            if key<=0 or key in owners:raise ValueError('Unique positive frontend CPU correlation IDs required.')
            owners[key]=i
    if len(encoded(trace))>MAX_BYTES:raise ValueError('Raw correlation trace byte ceiling exceeded; no truncation.')
    if len({raw[i]['device_index'] for i in owners.values()})!=1:
        raise ValueError('One recorded frontend CPU process required.')
    # Overhead can inherit its real parent's correlation ID in PyTorch 2.8.
    # Only role-qualified frontend rows enter owners; overhead never does.
    annotation_hosts={};annotation_keys=set()
    for i,r in enumerate(raw):
        if roles[i]!='profiler-GPU-range':continue
        owner=owners.get(r['correlation_id'])
        if owner is None:raise ValueError('GPU range lacks its recorded frontend phase owner.')
        host=raw[owner]
        if (host['name']!=r['name'] or not host['user_annotation'] or host['is_async']
                or host['thread']<=0 or host['thread']!=host['end_thread']):
            raise ValueError('GPU range needs its exact synchronous CPU phase annotation.')
        key=(r['correlation_id'],r['device_index'],r['resource_id'])
        if key in annotation_keys:raise ValueError('Duplicate GPU range for a phase and stream.')
        annotation_keys.add(key);annotation_hosts[i]=owner
    rows=[];cpu_groups={}
    for i,r in enumerate(raw):
        linked=r['linked_correlation_id'];owner=owners.get(linked)
        if linked and owner is None:raise ValueError('Unknown linked CPU correlation ID; no fallback.')
        if r['device']=='cuda' and roles[i]!='profiler-GPU-range' and (r['device_index']!=0 or r['user_annotation'] or not linked):
            raise ValueError('Actual device-0 non-annotation CUDA event with CPU correlation required.')
        if linked:
            o=raw[owner]
            if o['is_async'] or o['thread']!=o['end_thread'] or o['thread']<=0:
                raise ValueError('Linked CPU owner is asynchronous or has no stable thread.')
            if r['device']=='cpu' and not o['start_ns']<=r['start_ns']<=r['end_ns']<=o['end_ns']:
                raise ValueError('Linked runtime event lies outside its recorded CPU owner.')
        thread=raw[owner]['thread'] if linked and r['device']=='cpu' else r['thread']
        row={'index':i,'name':r['name'],'device':r['device'],'device_index':r['device_index'],
            'thread':thread,'fwd_thread':r['fwd_thread'],'sequence_nr':r['sequence_nr'],'scope':r['scope'],
            'user_annotation':r['user_annotation'],'parent':None,'start_us':(r['start_ns']-origin)/1000,
            'end_us':(r['end_ns']-origin)/1000,'legacy':False,'linked_kernels':[],
            'correlation_id':r['correlation_id'],'linked_correlation_id':linked,'resource_id':r['resource_id'],
            'raw_thread':r['thread'],'record_role':roles[i],'annotation_host_event_index':annotation_hosts.get(i),'start_ns':r['start_ns'],'end_ns':r['end_ns']}
        rows.append(row)
        if roles[i]!='profiler-overhead-signature' and r['device']=='cpu' and not r['is_async'] and r['thread']==r['end_thread']:
            cpu_groups.setdefault(thread,[]).append(i)
    # Interval containment is used only for CPU ancestry, never GPU launch choice.
    for indices in cpu_groups.values():
        intervals=[(raw[i]['start_ns'],raw[i]['end_ns']) for i in indices]
        if len(intervals)!=len(set(intervals)):
            raise ValueError('Indistinguishable CPU intervals; no arbitrary ancestry tie-break.')
        stack=[]
        for i in sorted(indices,key=lambda j:(raw[j]['start_ns'],-raw[j]['end_ns'],j)):
            r=raw[i]
            while stack and (r['start_ns']>=raw[stack[-1]]['end_ns'] or r['end_ns']>raw[stack[-1]]['end_ns']):stack.pop()
            if stack:rows[i]['parent']=stack[-1]
            stack.append(i)
    for i,r in enumerate(raw):
        if roles[i]=='CUDA-activity' and r['device']=='cuda' and r['end_ns']>r['start_ns']:
            owner=owners[r['linked_correlation_id']]
            rows[owner]['linked_kernels'].append({'name':r['name'],'device':r['device_index'],
                'duration_us':(r['end_ns']-r['start_ns'])/1000,'event_index':i})
    return rows


def _verify_view(trace,*,arm,device):
    rows=project(trace)
    if arm not in ('original','math') or device not in ('cpu','cuda') or not isinstance(rows,list) or not 0<len(rows)<=MAX_EVENTS:
        raise ValueError('Exact component trace/domain required.')
    def chain(i):
        seen=set()
        while i is not None:
            if type(i) is not int or not 0<=i<len(rows) or i in seen:raise ValueError('Invalid component parent graph.')
            if not isinstance(rows[i],dict) or 'parent' not in rows[i]:raise ValueError('Invalid component parent record.')
            seen.add(i);yield i;i=rows[i]['parent']
    ancestry={};labels={}
    for i,r in enumerate(rows):
        if not isinstance(r,dict):raise ValueError('Typed event object required.')
        if (type(r.get('index')) is not int or r.get('index')!=i or r.get('device') not in ('cpu','cuda') or not isinstance(r.get('name'),str)
                or any(type(r.get(k)) is not int for k in ('device_index','thread','scope','fwd_thread','sequence_nr')) or r.get('legacy') is not False
                or type(r.get('user_annotation')) is not bool or any(type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) for k in ('start_us','end_us'))
                or r['end_us']<r['start_us']):raise ValueError('Invalid device-typed component event.')
        if not isinstance(r.get('linked_kernels'),list):raise ValueError('Actual linked-kernel list required.')
        c=list(chain(i));ancestry[i]=c
        if r['parent'] is not None:
            parent=rows[r['parent']]
            if r['device']!='cpu' or parent['device']!='cpu' or parent['thread']!=r['thread'] or parent['start_us']>r['start_us'] or parent['end_us']<r['end_us']:
                raise ValueError('Broken CPU event containment.')
        if r['device']=='cpu' and r['name'] in (FORWARD,BACKWARD):
            if not r['user_annotation'] or r['name'] in labels:raise ValueError('Unique actual component phase labels required.')
            labels[r['name']]=i
        for k in r.get('linked_kernels',[]):
            if not isinstance(k,dict):raise ValueError('Typed linked kernel required.')
            if not isinstance(k.get('name'),str) or type(k.get('duration_us')) not in (int,float) or not math.isfinite(k['duration_us']) or k['duration_us']<0 or type(k.get('device')) is not int:
                raise ValueError('Malformed actual linked kernel.')
    if set(labels)!={FORWARD,BACKWARD}:raise ValueError('Missing distinct actual forward/backward phases.')
    regions,anchors=associated_regions(rows,ancestry,labels)
    if set(regions[FORWARD])&set(regions[BACKWARD]):raise ValueError('Ambiguous overlapping component phases.')
    def operators(indices):return {rows[i]['name'] for i in indices if rows[i]['scope']==0 and not rows[i]['user_annotation']}
    fops=operators(regions[FORWARD]);bops=operators(regions[BACKWARD]);allops=fops|bops
    detected=[family for family,(f,b) in FAMILIES.items() if f in fops]
    if len(detected)!=1:raise ValueError('Exactly one observed component forward backend required.')
    family=detected[0];fop,bop=FAMILIES[family]
    if bop not in bops or any('_cudnn_attention' in n or '_flash_attention' in n and '_for_cpu' not in n for n in allops):
        raise ValueError('Unrecognized/mixed or missing actual component backward backend; no fallback.')
    if any(b in bops for other,(f,b) in FAMILIES.items() if other!=family):raise ValueError('Mixed component backward families observed.')
    if arm=='math' and family!='math':raise ValueError('Candidate math selection was not observed.')
    if device=='cpu' and family=='efficient' or device=='cuda' and family=='cpu-flash':raise ValueError('Observed backend belongs to another device domain.')
    if family=='math' and 'aten::bmm' not in bops and 'aten::mm' not in bops:raise ValueError('Math VJP matrix operations not observed.')
    kernel_witnesses={}
    kernels={r['name'] for r in rows if r['device']=='cuda' and r['end_us']>r['start_us'] and not r['user_annotation']}
    if device=='cpu':
        if kernels or any(r['device']=='cuda' or r.get('linked_kernels') for r in rows):raise ValueError('CPU proof cannot carry CUDA activity.')
        phase_kernels={FORWARD:[],BACKWARD:[]}
    else:
        if not kernels:raise ValueError('Missing actual non-annotation CUDA activity.')
        if any(r['device']=='cuda' and r['device_index']!=0 for r in rows):raise ValueError('Unexpected CUDA device index.')
        def correlated(op,region):
            nodes=[i for i in regions[region] if rows[i]['name']==op and not rows[i]['user_annotation']]
            found=set()
            for i in regions[region]:
                if not any(n in ancestry[i] for n in nodes):continue
                for link_index,k in enumerate(rows[i]['linked_kernels']):
                    if k['device']!=0:raise ValueError('Linked kernel belongs to another CUDA device.')
                    if k['duration_us']<=0:continue
                    gpu=k['event_index'];r=rows[gpu]
                    if (r['device']!='cuda' or r['device_index']!=k['device'] or r['name']!=k['name']
                            or r['linked_correlation_id']!=rows[i]['correlation_id'] or r['end_ns']<=r['start_ns']):
                        raise ValueError('Exact recorded GPU-to-host correlation required.')
                    previous=kernel_witnesses.get(gpu)
                    if previous and i not in ancestry[previous['host_event_index']] and previous['host_event_index'] not in ancestry[i]:
                        raise ValueError('One kernel witness claimed by unrelated host events.')
                    if not previous or previous['host_event_index'] in ancestry[i]:
                        kernel_witnesses[gpu]={'host_event_index':i,'host_link_index':link_index,'CUDA_event_index':gpu,
                            'name':k['name'],'device':0,'host_correlation_id':rows[i]['correlation_id'],
                            'GPU_correlation_id':r['correlation_id'],'GPU_linked_correlation_id':r['linked_correlation_id'],
                            'resource_id':r['resource_id'],'start_ns':r['start_ns'],'end_ns':r['end_ns']}
                    found.add(k['name'])
            if not found:raise ValueError('Required actual operator has no correlated positive CUDA kernel.')
            return sorted(found)
        phase_kernels={FORWARD:correlated(fop,FORWARD),BACKWARD:correlated(bop,BACKWARD)}
        if family=='efficient':
            if (not any(n.startswith('fmha_cutlassF_f32_') or 'AttentionKernel<' in n for n in phase_kernels[FORWARD])
                    or not any(n.startswith('fmha_cutlassB_f32_') or 'AttentionBackwardKernel<' in n for n in phase_kernels[BACKWARD])):
                raise ValueError('Associated efficient forward/backward kernel family was not recognized.')
        else:
            matrix={}
            for region,ops in ((FORWARD,fops),(BACKWARD,bops)):
                matmul='aten::bmm' if 'aten::bmm' in ops else 'aten::mm'
                matrix[region]=correlated(matmul,region)
                if not any('gemm' in n.lower() for n in matrix[region]):raise ValueError('Associated math matrix CUDA kernel was not recognized.')
                phase_kernels[region]=sorted(set(phase_kernels[region])|set(matrix[region]))
            softmax=correlated('aten::_softmax',FORWARD)
            if (not any('softmax' in n.lower() for n in softmax)
                    or not any('SoftMaxBackward' in n or 'softmax_warp_backward<' in n for n in phase_kernels[BACKWARD])):
                raise ValueError('Associated math forward/backward softmax CUDA kernels were not recognized.')
            phase_kernels[FORWARD]=sorted(set(phase_kernels[FORWARD])|set(softmax))
    return {'kind':'one-fixed-SDPA-correlated-backend-v3','observed_family':family,'selection':arm,'device':device,
        'CPU_only':device=='cpu','CUDA_forward_and_backward_qualified':device=='cuda',
        'phase_correlated_CUDA_kernel_names':phase_kernels,'timings_usable_for_speed_or_cost':False,
        'association_mode':'CPU-parent-and-autograd-identity' if anchors else 'CPU-parent',
        'cross_thread_anchors':anchors,'kernel_witnesses':[kernel_witnesses[k] for k in sorted(kernel_witnesses)],
        'kernel_match_fields':['raw_event_index','recorded_CPU_correlation_id','recorded_GPU_linked_correlation_id'],
        'raw_trace_kind':KIND,'recorded_host_correlation_links_available':True,
        'per_kernel_global_unique_correlation_id_assumed':False,'kernel_name_or_duration_used_as_identity':False,
        'metadata_only':True,'actual_CUDA_execution_admitted':False,'worker_completion_inferred':False,'full_model_candidate_adopted':False,
        'default_family_was_assumed':False,'unsupported_backend_action':'refuse and preserve; never force a fallback'}


def verify(trace,*,arm,device):
    """Metadata-only assessment under this separate policy; never closes a worker."""
    proof=_verify_view(trace,arm=arm,device=device)
    rows=project(trace)
    indices=[i for i,r in enumerate(rows) if r['record_role']=='profiler-overhead-signature']
    annotations=[i for i,r in enumerate(rows) if r['record_role']=='profiler-GPU-range']
    proof.update(kind='one-fixed-SDPA-correlated-backend-with-overhead-v1',
        base_proof_kind=proof['kind'],metadata_CUDA_links_valid=proof['CUDA_forward_and_backward_qualified'],
        CUDA_forward_and_backward_qualified=False,metadata_policy=POLICY,raw_trace_sha256=hashlib.sha256(encoded(trace)).hexdigest(),
        raw_records=len(rows),projected_records=len(rows),retained_overhead_indices=indices,
        retained_GPU_annotation_indices=annotations,annotations_qualify_as_kernels=False,
        overhead_interpretation='source-supported strict signature; native activity kind not recorded',
        overhead_can_own_launches=False,overhead_can_parent_CPU_operations=False,raw_rows_removed_or_rewritten=False,
        native_activity_kind_observed=False,installed_binary_source_commit_verified=False,
        pinned_Kineto_source=KINETO_PIN,historical_worker_completion_inferred=False,
        fresh_process_replay_inferred=False)
    return proof

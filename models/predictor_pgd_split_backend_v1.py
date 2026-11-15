"""Metadata-only split-profile reader; never runs a model, profiler or CUDA.

The frozen raw projection preserves overhead/GPU-annotation distinctions. This
NEW association policy joins outer custom and inner math producers separately.
A matching recipe is not native ownership, actual execution or worker admission.
"""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import stat

from models import predictor_sdpa_launch_evidence_v2 as frozen

POLICY='pgd-native-forward-math-backward-metadata-v1'
DESIGN_SHA='c5daae30cbe4043cf4d5265de2a2a7420d015ab1e244cc3d74fcc93b5f0186af'
MAX_BYTES=4*2**20
MAX_EVENTS=20000
MAX_DEPTH=128
MAX_ANCHORS=16
MAX_PROOF_BYTES=128*1024
RUNTIME='2.8.0+cu128'
FORWARD='fixed-SDPA/forward'
BACKWARD='fixed-SDPA/backward'
ENGINE='autograd::engine::evaluate_function: '
EFFICIENT='aten::_scaled_dot_product_efficient_attention'
EFFICIENT_BACKWARD=EFFICIENT+'_backward'
MATH='aten::_scaled_dot_product_attention_math'
CUSTOM='_Attention'
CUSTOM_BACKWARD='_AttentionBackward'
MATRIX={'BmmBackward0':frozenset(('aten::bmm',)), 'MmBackward0':frozenset(('aten::mm',))}
SOFTMAX={'SoftmaxBackward0':frozenset(('aten::_softmax','aten::softmax')),
         'SafeSoftmaxBackward0':frozenset(('aten::_safe_softmax',))}
PINNED={
 'predictor_sdpa_launch_evidence_v2.py':'05f99a96ca34d65c4167cdbd37f97600b14468aa08fbfe5aa9b581b72e5fdddb',
 'predictor_sdpa_launch_evidence_v1.py':'c9749407f3d237050252ec7e9d4831c901df813a35b3b4a38ad8d693e0e1899f',
 'predictor_sdpa_component_events_v2.py':'ebde9c0f32078979cfa5920503004bb79ca697fdc05d7adacbfc59bde41f515a',
 'predictor_sdpa_component_events_v1.py':'c1397e06ca89d1299a7d853e6c88124d1cc7874aae3f5763f018fbc67bd195fc'}
_PROJECT=frozen.project
_ROLE=frozen.record_role
_ENCODED=frozen.encoded


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def _sha(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def source_gate():
    for name,want in PINNED.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=want:
            raise ValueError('Frozen raw projection source changed: '+name)
    if (frozen.project is not _PROJECT or frozen.record_role is not _ROLE or frozen.encoded is not _ENCODED
            or frozen.MAX_EVENTS!=20000 or frozen.MAX_BYTES!=8*2**20 or frozen.FORWARD!=FORWARD
            or frozen.BACKWARD!=BACKWARD or frozen.OVERHEAD_RUNTIME!=RUNTIME):
        raise ValueError('Frozen raw projection binding changed.')


def _pairs(items):
    out={}
    for key,value in items:
        if key in out:raise ValueError('Duplicate JSON key; preserve and refuse.')
        out[key]=value
    return out


def parse(data):
    if type(data) is not bytes or not 0<len(data)<=MAX_BYTES:
        raise ValueError('One complete raw trace within 4 MiB required; no truncation.')
    def invalid(unused):raise ValueError('Nonfinite JSON constant refused.')
    trace=json.loads(data,object_pairs_hook=_pairs,parse_constant=invalid)
    if (type(trace) is not dict or trace.get('torch_version')!=RUNTIME
            or type(trace.get('records')) is not list or not 0<len(trace['records'])<=MAX_EVENTS):
        raise ValueError('Exact pinned runtime and bounded raw records required.')
    return trace


class _View:
    def __init__(self,trace):
        self.trace=trace
        self.rows=_PROJECT(trace)
        self.raw=trace['records']
        self.ancestry={}
        for i,row in enumerate(self.rows):
            chain=[];parent=i
            while parent is not None:
                if type(parent) is not int or not 0<=parent<len(self.rows) or parent in chain or len(chain)>=MAX_DEPTH:
                    raise ValueError('Bounded unambiguous CPU ancestry required.')
                chain.append(parent);parent=self.rows[parent]['parent']
            self.ancestry[i]=tuple(chain)
        self.cpu={i for i in range(len(self.rows)) if self.frontend(i)}
        labels={name:[i for i in self.cpu if self.rows[i]['name']==name] for name in (FORWARD,BACKWARD)}
        if any(len(v)!=1 for v in labels.values()):raise ValueError('Two unique CPU phase annotations required.')
        self.f,self.b=(labels[name][0] for name in (FORWARD,BACKWARD))
        f,b=(self.rows[i] for i in (self.f,self.b))
        if (any(not r['user_annotation'] or r['scope']!=7 for r in (f,b))
                or f['thread']!=b['thread'] or f['device_index']!=b['device_index']
                or not f['start_ns']<f['end_ns']<=b['start_ns']<b['end_ns']):
            raise ValueError('Ordered synchronous phases on one calling thread/process required.')
        self.forward=self.desc(self.f)
        self.backward=self.desc(self.b)
        self.ops={i for i in self.cpu if self.rows[i]['scope']==0 and not self.rows[i]['user_annotation']}
        critical={EFFICIENT,EFFICIENT_BACKWARD,MATH,CUSTOM,'aten::mm','aten::bmm','aten::matmul',
                  'aten::_softmax','aten::softmax','aten::_safe_softmax','aten::_softmax_backward_data'}
        if any(r['device']=='cpu' and r['name'] in critical and i not in self.ops for i,r in enumerate(self.rows)):
            raise ValueError('Critical operator lacks synchronous frontend CPU identity.')
        self.nodes=defaultdict(list);self.engines=defaultdict(list);self.producers=defaultdict(list)
        for i in self.cpu:
            r=self.rows[i];key=(r['fwd_thread'],r['sequence_nr'])
            if r['scope']==1 and not r['user_annotation']:self.nodes[key].append(i)
            if i in self.ops:
                if r['name'].startswith(ENGINE):self.engines[(r['name'],*key)].append(i)
                elif r['sequence_nr']>=0:self.producers[(r['thread'],r['sequence_nr'])].append(i)
        self.anchors=[];self.bound_nodes=set();self.used_keys=set();self.witnesses={}

    def frontend(self,i):
        r=self.rows[i];raw=self.raw[i]
        return (r['device']=='cpu' and r['record_role']=='frontend-CPU' and not raw['is_async']
                and raw['thread']==raw['end_thread'] and raw['thread']>0)

    def desc(self,index):
        return {i for i,r in enumerate(self.rows) if r['device']=='cpu' and index in self.ancestry[i]}

    def named(self,name):
        return [i for i in self.ops if self.rows[i]['name']==name]

    def exactly(self,name,count):
        found=self.named(name)
        if len(found)!=count:raise ValueError('Exact component operator count required: '+name)
        return sorted(found)

    def bind(self,index,pool,allowed,container,stage):
        """Join by recorded source thread+sequence and real engine/node parent."""
        node=self.rows[index];key=(node['fwd_thread'],node['sequence_nr'])
        if (len(self.anchors)>=MAX_ANCHORS or index in self.bound_nodes or key in self.used_keys or key[0]<=0 or key[1]<0
                or self.nodes[key]!=[index] or node['name'] not in allowed):
            raise ValueError('Unique recorded autograd node identity required.')
        engine_index=node['parent']
        if engine_index is None or engine_index not in self.ops:raise ValueError('Direct actual engine parent required.')
        engine=self.rows[engine_index];scope=self.rows[container]
        if (self.engines[(ENGINE+node['name'],*key)]!=[engine_index]
                or engine['name']!=ENGINE+node['name']
                or (engine['thread'],engine['fwd_thread'],engine['sequence_nr'])!=(node['thread'],*key)
                or engine['device_index']!=scope['device_index'] or node['device_index']!=scope['device_index']
                or not scope['start_ns']<=engine['start_ns']<engine['end_ns']<=scope['end_ns']):
            raise ValueError('Exact engine/node identity and enclosing CPU interval required.')
        candidates=self.producers[key]
        matches=[i for i in candidates if i in pool]
        if not 0<len(matches)<=MAX_DEPTH:raise ValueError('Bounded recorded producer chain required.')
        deepest=[i for i in matches if all(j in self.ancestry[i] for j in matches)]
        if len(deepest)!=1:raise ValueError('Missing or ambiguous recorded producer chain.')
        producer_index=deepest[0];producer=self.rows[producer_index]
        if producer['name'] not in allowed[node['name']] or producer['end_ns']>engine['start_ns']:
            raise ValueError('Autograd producer type/order is outside the selected computation.')
        if any(i not in matches and self.rows[i]['start_ns']<engine['start_ns'] for i in candidates):
            raise ValueError('Recorded producer identity reused outside its selected scope.')
        # CPU containment and a source identity must agree. A detached foreign
        # thread may join only at a root or through the caller's backward scope.
        permitted={container} if self.rows[container]['scope']==1 else set()
        if any(self.rows[i]['scope']==1 and i not in permitted for i in self.ancestry[engine_index]):
            raise ValueError('Engine lies under an unrelated autograd node.')
        if engine_index not in self.desc(container):
            if engine['thread']==producer['thread']:
                raise ValueError('Detached same-thread backward cannot be repaired by names/time.')
            parent=engine['parent']
            if parent is not None and parent not in self.backward:
                raise ValueError('Foreign engine parent outside the recorded backward scope.')
        self.bound_nodes.add(index);self.used_keys.add(key)
        anchor={'stage':stage,'node_index':index,'node_name':node['name'],'engine_index':engine_index,
            'producer_index':producer_index,'producer_name':producer['name'],'producer_chain':sorted(matches),
            'forward_thread':key[0],'sequence_nr':key[1],'backward_thread':node['thread'],
            'container_index':container,'cross_thread':node['thread']!=key[0],
            'CPU_interval_checked':True,'GraphTask_ID_observed':False}
        self.anchors.append(anchor)
        return self.desc(index)

    def correlated(self,indices,group,predicate):
        """Resolve GPU rows only by their recorded link to qualified CPU hosts."""
        indices=set(indices);found=set()
        for i in sorted(self.ops):
            if not indices.intersection(self.ancestry[i]):continue
            for link_number,link in enumerate(self.rows[i]['linked_kernels']):
                gpu=link['event_index'];raw=self.raw[gpu];host=self.rows[i]
                if (self.rows[gpu]['record_role']!='CUDA-activity' or raw['user_annotation']
                        or raw['device_index']!=0 or raw['linked_correlation_id']!=host['correlation_id']
                        or raw['end_ns']<=raw['start_ns'] or link['device']!=0 or link['name']!=raw['name']):
                    raise ValueError('Exact positive non-annotation GPU-to-CPU correlation required.')
                if not predicate(raw['name']):continue
                previous=self.witnesses.get(gpu)
                if previous and previous['host_event_index']!=i:
                    raise ValueError('One device event cannot acquire a second host owner.')
                if not previous:
                    self.witnesses[gpu]={'host_event_index':i,'host_link_index':link_number,'CUDA_event_index':gpu,
                        'host_correlation_id':host['correlation_id'],'GPU_correlation_id':raw['correlation_id'],
                        'GPU_linked_correlation_id':raw['linked_correlation_id'],'device':0,'resource_id':raw['resource_id'],
                        'name':raw['name'],'start_ns':raw['start_ns'],'end_ns':raw['end_ns'],'groups':[]}
                if group not in self.witnesses[gpu]['groups']:self.witnesses[gpu]['groups'].append(group)
                found.add(gpu)
        if not found:raise ValueError('Missing correlated positive CUDA kernel for '+group)
        return sorted(found)


def _assess(trace,role,binding,raw_sha):
    view=_View(trace);rows=view.rows
    efficient,=view.exactly(EFFICIENT,1)
    if efficient not in view.forward:raise ValueError('Efficient forward must belong to its forward phase.')
    forbidden=[i for i in view.ops if any(token in rows[i]['name'] for token in ('_cudnn_attention','_flash_attention'))]
    if forbidden:raise ValueError('Unsupported attention backend; no fallback.')
    view.correlated([efficient],'efficient-forward',lambda n:n.startswith('fmha_cutlassF_f32_') or 'AttentionKernel<' in n)
    counts={'efficient_forward':1,'efficient_backward':0,'math_forward_in_backward':0}
    if role=='native':
        view.exactly(MATH,0)
        backward,=view.exactly(EFFICIENT_BACKWARD,1)
        nodes=[i for i in view.cpu if rows[i]['scope']==1 and rows[i]['name']=='ScaledDotProductEfficientAttentionBackward0' and not rows[i]['user_annotation']]
        if len(nodes)!=1:raise ValueError('One actual efficient autograd node required.')
        owned=view.bind(nodes[0],view.forward,{'ScaledDotProductEfficientAttentionBackward0':frozenset((EFFICIENT,))},view.b,'native')
        if backward not in owned:raise ValueError('Native backward operator outside its producer-bound node.')
        if view.named(CUSTOM) or any(rows[i]['scope']==1 and rows[i]['name'] in set(MATRIX)|set(SOFTMAX)|{CUSTOM_BACKWARD} for i in view.cpu):
            raise ValueError('Native probe cannot carry a custom attention route.')
        view.correlated([backward],'efficient-backward',lambda n:n.startswith('fmha_cutlassB_f32_') or 'AttentionBackwardKernel<' in n)
        counts['efficient_backward']=1
        matrix_region=set();math_region=set();softmax_region=set()
    else:
        view.exactly(EFFICIENT_BACKWARD,0)
        if any(rows[i]['scope']==1 and rows[i]['name']=='ScaledDotProductEfficientAttentionBackward0' for i in view.cpu):
            raise ValueError('Hybrid cannot execute an unused native efficient backward node.')
        custom,=view.exactly(CUSTOM,1)
        if custom not in view.forward or custom not in view.ancestry[efficient]:
            raise ValueError('Original efficient forward must descend from its actual custom forward.')
        nodes=[i for i in view.cpu if rows[i]['scope']==1 and rows[i]['name']==CUSTOM_BACKWARD and not rows[i]['user_annotation']]
        if len(nodes)!=1:raise ValueError('One actual custom backward node required.')
        outer=nodes[0]
        custom_region=view.bind(outer,view.forward,{CUSTOM_BACKWARD:frozenset((CUSTOM,))},view.b,'custom-outer')
        math_index,=view.exactly(MATH,1)
        if math_index not in custom_region:raise ValueError('Math recomputation must be inside the producer-bound custom backward.')
        math_region=view.desc(math_index)
        matrix_region=set();softmax_region=set()
        for family,mapping in [('matrix',MATRIX),('softmax',SOFTMAX)]:
            nodes=sorted(i for i in view.cpu if rows[i]['scope']==1 and rows[i]['name'] in mapping and not rows[i]['user_annotation'])
            if not nodes:raise ValueError('Recorded inner '+family+' backward producers required.')
            for node in nodes:
                owned=view.bind(node,math_region,mapping,outer,'inner-'+family)
                if family=='matrix':
                    ops={i for i in owned&view.ops if rows[i]['name'] in ('aten::mm','aten::bmm')}
                    if not ops:raise ValueError('Inner matrix node has no actual matrix operation.')
                    for op in sorted(ops):view.correlated([op],'math-GEMM-backward',lambda n:'gemm' in n.lower())
                    matrix_region.update(owned)
                else:
                    ops={i for i in owned&view.ops if rows[i]['name']=='aten::_softmax_backward_data'}
                    if not ops:raise ValueError('Inner softmax node has no actual softmax backward.')
                    for op in sorted(ops):view.correlated([op],'math-softmax-backward',lambda n:'SoftMaxBackward' in n or 'softmax_warp_backward<' in n)
                    softmax_region.update(owned)
        matrix_ops={i for i in math_region&view.ops if rows[i]['name'] in ('aten::mm','aten::bmm')}
        softmax_ops={i for i in math_region&view.ops if rows[i]['name']=='aten::_softmax'}
        if not matrix_ops or not softmax_ops:raise ValueError('Actual math matrix and softmax forwards required.')
        for op in sorted(matrix_ops):view.correlated([op],'math-GEMM-forward',lambda n:'gemm' in n.lower())
        for op in sorted(softmax_ops):view.correlated([op],'math-softmax-forward',lambda n:'softmax' in n.lower())
        counts['math_forward_in_backward']=1
    # Do not let extra math operators outside their source-bound scopes hide in
    # a broad backward interval or be rescued by a similarly named GPU launch.
    for i in view.ops:
        name=rows[i]['name']
        if name in ('aten::mm','aten::bmm','aten::matmul') and i not in math_region|matrix_region:
            raise ValueError('Unassociated or extra math matrix operation.')
        if name in ('aten::_softmax','aten::softmax','aten::_safe_softmax') and i not in math_region:
            raise ValueError('Unassociated or extra math softmax forward.')
        if name=='aten::_softmax_backward_data' and i not in softmax_region:
            raise ValueError('Unassociated or extra math softmax backward.')
    proof={'kind':'pgd-split-profile-metadata-assessment-v1','policy':POLICY,'binding':dict(binding),'role':role,
        'metadata_matches_recipe':True,'metadata_only':True,'raw_trace_sha256':raw_sha,
        'raw_records':len(trace['records']),'projected_records':len(rows),'operator_counts':counts,
        'producer_anchors':view.anchors,'kernel_witnesses':[view.witnesses[i] for i in sorted(view.witnesses)],
        'retained_overhead_indices':[i for i,r in enumerate(rows) if r['record_role']=='profiler-overhead-signature'],
        'retained_GPU_annotation_indices':[i for i,r in enumerate(rows) if r['record_role']=='profiler-GPU-range'],
        'raw_rows_removed_or_rewritten':False,'kernel_name_or_duration_used_as_identity':False,
        'per_kernel_global_unique_correlation_id_assumed':False,'overhead_or_annotation_qualifies_as_kernel':False,
        'binding_lineage_authenticated':False,'tensor_geometry_or_accuracy_verified':False,
        'component_root_leaf_or_GraphTask_ownership_verified':False,'actual_CUDA_execution_admitted':False,
        'worker_completion_inferred':False,'full_EOT8_backend_qualified':False,
        'fresh_replay_performance_cost_or_training_qualified':False}
    if len(encoded(proof))>MAX_PROOF_BYTES:raise ValueError('Bounded metadata assessment exceeded; no witness truncation.')
    return proof


def assess_bytes(data,*,role,binding):
    if role not in ('native','forward-preserving'):raise ValueError('Exact proposed split-profile selection required.')
    fields={'design_sha256','control_sha256','boundary_sha256','probe_index'}
    if (type(binding) is not dict or set(binding)!=fields or binding['design_sha256']!=DESIGN_SHA
            or not all(_sha(binding[k]) for k in fields-{'probe_index'})
            or type(binding['probe_index']) is not int or binding['probe_index']!=(0 if role=='native' else 1)):
        raise ValueError('Exact external design/control/boundary/probe binding required; not authenticated here.')
    trace=parse(data);source_gate()
    before=hashlib.sha256(encoded(trace)).hexdigest()
    result=_assess(trace,role,binding,hashlib.sha256(data).hexdigest())
    if hashlib.sha256(encoded(trace)).hexdigest()!=before:raise ValueError('Source metadata changed during assessment.')
    return result


def _identity(s):
    return (s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode),s.st_nlink,s.st_size,s.st_mtime_ns)


def read_private_trace(path,expected_sha256):
    """Read a held private source once; no tensor/archive fallback or repair."""
    if not _sha(expected_sha256):raise ValueError('Expected raw trace digest required.')
    path=Path(path).absolute();parent=path.parent
    if path.name!='events.json':raise ValueError('Exact proposed raw events.json input only; no tensor/archive fallback.')
    if parent.resolve(strict=True)!=parent:raise ValueError('Canonical private source parent required.')
    directory=parent.lstat();s=path.lstat()
    if (not stat.S_ISDIR(directory.st_mode) or directory.st_uid!=os.getuid() or stat.S_IMODE(directory.st_mode)!=0o700
            or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=os.getuid()
            or stat.S_IMODE(s.st_mode)!=0o600 or not 0<s.st_size<=MAX_BYTES):
        raise ValueError('Owned private single-link bounded raw trace required.')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        before=os.fstat(f.fileno());data=f.read(MAX_BYTES+1);after=os.fstat(f.fileno())
    now=parent.lstat()
    if (_identity(s)!=_identity(before) or _identity(before)!=_identity(after) or _identity(path.lstat())!=_identity(after)
            or (directory.st_dev,directory.st_ino,directory.st_uid,stat.S_IMODE(directory.st_mode))
               !=(now.st_dev,now.st_ino,now.st_uid,stat.S_IMODE(now.st_mode)) or len(data)!=s.st_size):
        raise ValueError('Held source or parent changed during read.')
    if hashlib.sha256(data).hexdigest()!=expected_sha256:raise ValueError('Expected private trace bytes changed.')
    return data

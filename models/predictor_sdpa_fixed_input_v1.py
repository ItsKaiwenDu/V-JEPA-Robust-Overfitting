"""Offline proposal and small local CPU capture proof, never a live operator."""
import copy
import math
import platform
import threading

KIND = 'predictor-same-qkv-cotangent-proposal-v1'
PREDECESSOR = '9066c0afb97cb9e92f02e047806124aab73fff36cc9cdb46b4c31e28d167e94c'
MiB, GiB = 2**20, 2**30
SHAPE = [1, 12, 2048, 32]
RULE = 'first original predictor call in frozen execution order; no magnitude ranking'
WORKERS = ['native-original-capture', 'fixed-original-0', 'fixed-math-0', 'fixed-original-1', 'fixed-math-1']
RESOURCE = {'host': 'linux10603', 'parent': '/var/tmp', 'work_seconds': 1500,
    'setup_seconds': 180, 'write_proof_bytes': 5*GiB//2, 'write_proof_seconds': 60,
    'capture_worker_seconds': 420, 'component_workers': 4, 'component_worker_seconds': 120,
    'receipt_reserve_seconds': 120, 'static_reserved_seconds': 1200,
    'setup_bytes': 3*GiB, 'capture_bytes': 9*GiB//4, 'raw_bytes': 512*MiB,
    'metadata_bytes': 64*MiB, 'input_bytes': 128*MiB, 'code_bytes': 128*MiB,
    'cache_bytes': 32*MiB, 'emergency_bytes': 16*MiB,
    'tensor_file_bytes': 16*MiB, 'allocator_bytes': 7*GiB,
    'collection_seconds': 600, 'collection_bytes': 3*GiB}
TOLERANCES = {'output': {'rtol': 1e-4, 'atol': 2e-5},
              **{k: {'rtol': .002, 'atol': 2e-7} for k in ('dq', 'dk', 'dv')}}


def extent(shape, stride):
    if (not isinstance(shape, list) or not isinstance(stride, list) or len(shape) != len(stride)
            or not 0 < len(shape) <= 4 or any(type(n) is not int or n <= 0 for n in shape)
            or any(type(n) is not int or n <= 0 for n in stride)):
        raise ValueError('Positive bounded shape and stride required.')
    return 4 * (1 + sum((n-1)*s for n, s in zip(shape, stride)))


def proposal(review, original, rows):
    """Derive a fixed first-call selection from reviewed records, not rankings."""
    if (review.get('kind') != 'predictor-common-cotangent-v1-target-saved-review'
            or review.get('saved_review_passed') is not True or review.get('attribution_complete') is not True
            or review.get('candidate_remains_rejected') is not True
            or review.get('candidate_native_numerical_passed') is not False
            or review.get('consumed_attempt_sha256') != PREDECESSOR):
        raise ValueError('Exact reviewed rejected predecessor required.')
    if (original.get('cell') != '00' or original.get('path') != 0 or original.get('cotangent') != 0
            or original.get('native_numerical_passed') is not True or original.get('local_tiny') is not False
            or original.get('execution_host') != 'linux10603' or original.get('mode') != 'original'
            or original.get('baseline') is None or original.get('attribution_complete') is not True):
        raise ValueError('Passing original target value control required.')
    forward = [r for r in rows if r.get('during_backward') is False]
    owners = ['encoder.'+str(i) for i in range(24)] + ['predictor.'+str(i) for i in range(12)]
    if len(rows) != 1152 or [r.get('owner') for r in forward] != owners*16:
        raise ValueError('Full ordered eight-draw/two-family call coverage required.')
    selected = forward[24]
    if (selected['index'] != 24 or selected.get('owner') != 'predictor.0' or selected.get('phase') != 'EOT8-before'
            or selected.get('outcome') != 'returned' or selected.get('graph_id') != -1
            or selected.get('pid') != original.get('pid') or selected.get('backward_scope') is not None
            or selected.get('entry_flags') != selected.get('effective_flags')
            or selected.get('entry_flags') != selected.get('exit_flags')):
        raise ValueError('Original first forward attention-call identity changed.')
    for k in ('q', 'k', 'v'):
        t = selected[k]
        if (t.get('shape') != SHAPE or t.get('dtype') != 'torch.float32'
                or t.get('device') != 'cuda:0' or t.get('requires_grad') is not True
                or extent(t['shape'], t['stride']) > 16*MiB):
            raise ValueError('Recorded pretrained post-RoPE layout changed.')
    nbytes = 4*math.prod(SHAPE)
    plan = {
        'kind': KIND, 'proposal_only': True, 'authorized': False,
        'new_target_execution_permitted': False, 'operator_present': False,
        'prior_consumed_claim_sha256': PREDECESSOR,
        'candidate_native_numerical_passed': False, 'numerical_tolerances_changed': False,
        'research_admitted': False, 'performance_qualified': False, 'paid_compute': False,
        'automatic_retries': 0, 'cleanup': False, 'reset_old_claim_or_clock': False,
        'parallel_host': False, 'standalone_retrieval': False,
        'selection': {'source_cell': '00', 'owner': 'predictor.0', 'block': 0,
            'draw': 0, 'family': 0, 'phase': 'EOT8-before', 'forward_row_index': 24,
            'rule': RULE,
            'kwargs': {'dropout_p': 0., 'is_causal': False, 'attn_mask': None},
            'layout': {k: copy.deepcopy(selected[k]) for k in ('q', 'k', 'v')}},
        'missing_actual_target_values': ['q', 'k', 'v', 'output', 'grad_output', 'dq', 'dk', 'dv'],
        'existing_output_cotangent_is_at_predictor_output_not_SDPA_output': True,
        'capture_is_native_observation_no_cotangent_replacement': True,
        'capture_original_full_loss_input_must_replay_exactly': True,
        'all_component_cells_same_QKV_and_grad_output_logical_hashes_and_layouts': True,
        'component_original_must_match_captured_native_output_and_QKV_gradients_exactly': True,
        'fresh_process_replay_requires_exact_per_arm_outputs_and_QKV_gradients': True,
        'independent_FP64_analytical_oracle_required_before_acceptance': True,
        'component_bounds': copy.deepcopy(TOLERANCES),
        'original_full_model_bounds': {'loss': {'rtol': 1e-5, 'atol': 2e-6}, 'input': {'rtol': .002, 'atol': 2e-7}},
        'oracle_bookkeeping_bounds': {'rtol': 1e-12, 'atol': 1e-14},
        'actual_device_typed_CUDA_forward_and_backward_evidence_required': True,
        'original_full_model_kernel_identity_proved': False,
        'same_QKV_and_grad_output_imply_same_backend_internal_saved_state': False,
        'pure_backward_arithmetic_isolated': False,
        'private_backend_backward_API_or_synthetic_saved_state_permitted': False,
        'internal_forward_rounding_remains_in_public_SDPA_VJP_comparison': True,
        'failure_action': 'preserve values and stop before next worker; no fallback or retries',
        'proposed_serial_workers': list(WORKERS),
        'scientific_call_proposal': {'full_model_backward': 1, 'full_model_mask_draws': 8,
            'full_model_mask_families': 2, 'independent_FP64_oracle': 1,
            'component_untimed_iterations': 4, 'component_profiled_iterations': 4,
            'component_timed_or_warmup_iterations': 0},
        'raw_projection': {'elements_per_array': math.prod(SHAPE), 'FP32_bytes_per_array': nbytes,
            'capture_eight_FP32_arrays_bytes': 8*nbytes, 'four_FP64_reference_arrays_bytes': 8*nbytes,
            'four_workers_two_phases_four_FP32_returns_bytes': 32*nbytes,
            'original_full_model_loss_input_targets_bytes': 20*MiB+4,
            'steady_logical_tensor_bytes': 48*nbytes+20*MiB+4,
            'single_atomic_tensor_allowance_bytes': 16*MiB,
            'input_layout_storage_span_bytes': {k: extent(selected[k]['shape'], selected[k]['stride']) for k in ('q', 'k', 'v')},
            'estimate_is_not_actual_storage_or_allocator_peak': True},
        'resource_proposal': dict(RESOURCE),
        'financial_state_unchanged': copy.deepcopy(review['financial_state_unchanged']),
    }
    validate(plan)
    return plan


def validate(plan):
    if (plan.get('kind') != KIND or plan.get('proposal_only') is not True
            or plan.get('prior_consumed_claim_sha256') != PREDECESSOR
            or any(plan.get(k) is not False for k in ('authorized', 'new_target_execution_permitted', 'operator_present',
                'candidate_native_numerical_passed', 'numerical_tolerances_changed', 'research_admitted',
                'performance_qualified', 'paid_compute', 'cleanup', 'reset_old_claim_or_clock', 'parallel_host',
                'standalone_retrieval', 'pure_backward_arithmetic_isolated',
                'same_QKV_and_grad_output_imply_same_backend_internal_saved_state',
                'private_backend_backward_API_or_synthetic_saved_state_permitted'))
            or type(plan.get('automatic_retries')) is not int or plan['automatic_retries'] != 0):
        raise ValueError('Inert diagnostic-only proposal required; no execution grant or backward-only claim.')
    if (plan['component_bounds'] != TOLERANCES
            or plan['oracle_bookkeeping_bounds'] != {'rtol': 1e-12, 'atol': 1e-14}
            or plan['original_full_model_bounds'] != {'loss': {'rtol': 1e-5, 'atol': 2e-6}, 'input': {'rtol': .002, 'atol': 2e-7}}):
        raise ValueError('Original numerical bounds must remain unchanged.')
    resource=plan['resource_proposal']
    if resource != RESOURCE or any(type(resource[k]) is not type(v) for k,v in RESOURCE.items()):
        raise ValueError('Exact inert resource proposal required.')
    s=plan['selection']
    if ({k:s.get(k) for k in ('source_cell','owner','block','draw','family','phase','forward_row_index','rule','kwargs')}
            != {'source_cell':'00','owner':'predictor.0','block':0,'draw':0,'family':0,'phase':'EOT8-before','forward_row_index':24,'rule':RULE,
                'kwargs':{'dropout_p':0.,'is_causal':False,'attn_mask':None}}
            or plan['proposed_serial_workers'] != WORKERS):
        raise ValueError('Predetermined first-call selection and serial worker order required.')
    static=resource['setup_seconds']+resource['capture_worker_seconds']+resource['component_workers']*resource['component_worker_seconds']+resource['receipt_reserve_seconds']
    if static != resource['static_reserved_seconds'] or static >= resource['work_seconds']:
        raise ValueError('Static worker/proof/setup/receipt envelope does not fit.')
    raw=plan['raw_projection']
    if raw['steady_logical_tensor_bytes']+raw['single_atomic_tensor_allowance_bytes'] >= resource['raw_bytes']:
        raise ValueError('Conservative logical tensor projection exceeds proposed raw category.')
    return True


class LocalSDPABoundary:
    """Single selected local tiny CPU boundary; observational tensor hooks only.

    This prototype has no SSH, disk serialization or target model route. Repeated
    calls during activation recomputation use the unchanged builtin and are not
    mistaken for a second capture. All forward results and hook cotangents pass
    through untouched. The caller owns the normal forward/backward operations.
    """
    def __init__(self):
        if platform.node() in ('linux10603', 'linux10609'):
            raise ValueError('Local tiny CPU prototype only; no university execution.')
        self.thread=threading.get_ident(); self.calls=0; self.handles=[]; self.values={}; self.active=False

    def __enter__(self):
        if self.active or self.values or self.handles or self.calls:
            raise ValueError('Local observer is single use.')
        self.active=True
        return self

    def __call__(self,q,k,v):
        import torch
        if not self.active or threading.get_ident()!=self.thread:
            raise ValueError('Unowned local call.')
        tensors=(q,k,v)
        if (any(t.device.type!='cpu' or t.dtype!=torch.float32 or not t.requires_grad or t.ndim!=4 for t in tensors)
                or q.shape!=k.shape or q.shape!=v.shape or q.shape[0]!=1 or not 0<q.shape[1]<=2
                or not 0<q.shape[2]<=8 or not 0<q.shape[3]<=4
                or any(not torch.isfinite(t).all().item() for t in tensors)):
            raise ValueError('Finite FP32 CPU operands in tiny-only domain required.')
        index=self.calls;self.calls+=1
        out=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False,attn_mask=None)
        if index:
            return out
        def snapshot(t):
            return {'array':t.detach().cpu().numpy().copy(), 'shape':list(t.shape), 'stride':list(t.stride()),
                'dtype':str(t.dtype), 'device':str(t.device), 'requires_grad':t.requires_grad,
                'version':t._version}
        self.values.update({n:snapshot(t) for n,t in zip(('q','k','v','output'),(*tensors,out))})
        def hook(name):
            def observe(g):
                if name in self.values:
                    raise ValueError('Repeated local gradient observation.')
                self.values[name]=snapshot(g)
                return None
            return observe
        for name,t in zip(('dq','dk','dv','grad_output'),(*tensors,out)):
            self.handles.append(t.register_hook(hook(name)))
        return out

    def __exit__(self,kind,value,tb):
        for h in self.handles:h.remove()
        self.handles=[];self.active=False
        if kind is None and set(self.values)!={'q','k','v','output','grad_output','dq','dk','dv'}:
            raise ValueError('Incomplete local native boundary; no replay control.')


def tiny_same_input_comparison(capture):
    """Actual local CPU VJPs vs an independent analytical oracle, not CUDA."""
    import numpy as np
    import torch
    from torch.nn.attention import SDPBackend,sdpa_kernel
    from models.attention_saved_evidence import oracle,agreement
    names={'q','k','v','output','grad_output','dq','dk','dv'}
    if platform.node() in ('linux10603','linux10609') or set(capture)!=names:
        raise ValueError('Complete local CPU capture required.')
    for item in capture.values():
        a=item['array']
        if (item['device']!='cpu' or a.dtype!=np.float32 or a.ndim!=4 or a.shape[0]!=1
                or not 0<a.shape[1]<=2 or not 0<a.shape[2]<=8 or not 0<a.shape[3]<=4 or not np.isfinite(a).all()):
            raise ValueError('Only tiny finite FP32 CPU captured arrays are supported.')
    shape=capture['q']['array'].shape
    if any(item['array'].shape!=shape for item in capture.values()):
        raise ValueError('All actual native boundary arrays must have matching shapes.')
    # Preserve recorded positive non-dense strides, with zeroed unused storage.
    operands=[]
    for name in ('q','k','v','grad_output'):
        item=capture[name];span=extent(list(shape),item['stride'])
        backing=torch.zeros(span//4,dtype=torch.float32,device='cpu')
        t=backing.as_strided(shape,item['stride'])
        t.copy_(torch.from_numpy(item['array']))
        operands.append(t.requires_grad_(name!='grad_output'))
    q,k,v,g=operands
    with sdpa_kernel(SDPBackend.MATH):
        output=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False,attn_mask=None)
        gradients=torch.autograd.grad(output,(q,k,v),g)
    actual={n:t.detach().numpy() for n,t in zip(('output','dq','dk','dv'),(output,*gradients))}
    reference=oracle(*(capture[n]['array'] for n in ('q','k','v','grad_output')),chunk=8)
    exact={n:bool(np.array_equal(actual[n].view(np.uint32),capture[n]['array'].view(np.uint32))) for n in actual}
    checks={n:agreement(actual[n],reference[n],TOLERANCES[n]) for n in actual}
    if not all(exact.values()) or not all(x['passed'] for x in checks.values()):
        raise ValueError('Local same-input native replay or analytical check rejected.')
    return {'kind':'local-tiny-CPU-same-QKV-grad-output-proof-v1','native_replay_exact':exact,
        'independent_FP64_checks':checks,'non_dense_V_stride_preserved':list(v.stride())==capture['v']['stride'],
        'CUDA_qualified':False,'pretrained_target_values_captured':False,'pure_backward_arithmetic_isolated':False,
        'calls_observed_in_capture':1}

"""Synthetic post-RoPE SDPA probes; no pretrained model or study integration."""
from contextlib import contextmanager
import importlib
import math
import time

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend

from models.cuda_convolution import agreement,TOLERANCES as PATCH_TOLERANCES
from models.native_diagnostic import TOLERANCES as MODEL_TOLERANCES
from models.runtime import synchronize
from models.vjepa2 import sample_masks,use_upstream

KIND='fp32-attention-component-diagnostic-v1'
CASES=('predictor','encoder-0','encoder-1')
ARMS=('efficient','math')
TOLERANCES={'output':dict(PATCH_TOLERANCES['output']),
            **{name:dict(MODEL_TOLERANCES['input_gradient']) for name in ('dq','dk','dv')}}
ORACLE_ID='cpu-fp64-query-chunked-softmax-analytical-vjp-v1'


def sdp_flags():
    return {'flash':torch.backends.cuda.flash_sdp_enabled(),'efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),
            'math':torch.backends.cuda.math_sdp_enabled(),'cudnn':torch.backends.cuda.cudnn_sdp_enabled()}


@contextmanager
def force_backend(arm):
    if arm not in ARMS:raise ValueError('Unknown diagnostic-only attention backend.')
    before=sdp_flags()
    try:
        with sdpa_kernel(SDPBackend.MATH if arm=='math' else SDPBackend.EFFICIENT_ATTENTION):
            inside=sdp_flags()
            if inside!={name:name==arm for name in ('flash','efficient','math','cudnn')}:
                raise ValueError('The requested backend was not exclusively selected.')
            yield inside
    finally:
        if sdp_flags()!=before:raise RuntimeError('Attention backend flags did not restore.')


def positions_for(upstream,case,reference,tiny):
    if case not in CASES:raise ValueError('Unknown attention case.')
    if tiny:
        if case=='predictor':return torch.arange(16).view(1,-1)
        n=12 if case=='encoder-0' else 8
        return torch.randperm(16,generator=torch.Generator().manual_seed(20261030))[:n].view(1,-1)
    masks=sample_masks(upstream,batch_size=1,seed=reference['attack_seed']%(2**31-1))
    if case=='predictor':
        joined=torch.cat((masks[0].context,masks[0].target),dim=1).sort(dim=1).values
        if not torch.equal(joined,torch.arange(2048).view(1,-1)):raise ValueError('Predictor positions must cover the original sorted token sequence.')
        return joined
    return masks[int(case[-1])].context


def operands(upstream,case,reference,tiny):
    """Capture CPU-only upstream operands at its SDPA boundary, then restore F."""
    use_upstream(upstream);ids=positions_for(upstream,case,reference,tiny)
    # Tiny D=12 keeps each rotary subspace at four channels. D=8 hits the
    # upstream squeeze(-1) singleton behavior; do not alter pinned RoPE code.
    heads,dim,grid=(2,12,4) if tiny else (12,32,16) if case=='predictor' else (16,64,16)
    width=heads*dim;seed=20261030+CASES.index(case)
    cls=importlib.import_module('src.models.utils.modules').RoPEAttention
    saved=F.scaled_dot_product_attention;before=sdp_flags();captured=[]
    class Captured(Exception):pass
    def boundary(q,k,v,**kwargs):
        if kwargs!={'dropout_p':0.,'is_causal':False,'attn_mask':None}:raise ValueError('Upstream attention arguments changed.')
        captured.append((q.detach(),k.detach(),v.detach()));raise Captured()
    try:
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(seed)
            module=cls(dim=width,num_heads=heads,qkv_bias=True,attn_drop=0.,proj_drop=0.,use_sdpa=True,grid_size=grid).eval().requires_grad_(False)
            x=torch.randn(1,ids.shape[1],width)
            F.scaled_dot_product_attention=boundary
            try:module(x,mask=ids,T=1 if tiny else 8,H_patches=grid,W_patches=grid)
            except Captured:pass
    finally:
        F.scaled_dot_product_attention=saved
        if sdp_flags()!=before:raise RuntimeError('Operand capture leaked upstream backend flags.')
    if len(captured)!=1:raise ValueError('Exactly one CPU operand boundary must be captured.')
    q,k,v=captured[0]
    expected=(1,heads,ids.shape[1],dim)
    if any(t.device.type!='cpu' or t.dtype!=torch.float32 or tuple(t.shape)!=expected for t in (q,k,v)):
        raise ValueError('Unexpected synthetic post-RoPE operand geometry.')
    cotangent=torch.randn(expected,generator=torch.Generator().manual_seed(seed+100))
    metadata={'case':case,'tiny_fixture':tiny,'seed':seed,'heads':heads,'head_dim':dim,'tokens':ids.shape[1],
        'width':width,'grid_size':grid,'frames':1 if tiny else 8,'positions':ids,'SDPA_arguments':{'dropout_p':0.,'is_causal':False,'attn_mask':None,'scale':None},
        'operand_source':'Pinned upstream RoPEAttention, synthetic CPU weights/input, captured immediately before SDPA; no trained activations.',
        'strides':{name:list(t.stride()) for name,t in zip(('q','k','v'),(q,k,v))}}
    return (q,k,v,cotangent),metadata


def clone_layout(tensor,device,*,requires_grad=False):
    """Retain non-dense post-RoPE V strides; zero unused storage before copying."""
    if tensor.dtype!=torch.float32 or any(s<=0 for s in tensor.stride()):raise ValueError('Require positive-stride FP32 operands.')
    count=1+sum((n-1)*s for n,s in zip(tensor.shape,tensor.stride()))
    if count*4>128*2**20:raise ValueError('Unexpected component storage extent.')
    storage=torch.zeros(count,dtype=torch.float32,device=device)
    result=storage.as_strided(tensor.shape,tensor.stride());result.copy_(tensor)
    return result.detach().requires_grad_(requires_grad)


def fp64_oracle(q,k,v,cotangent,*,chunk=128):
    """Stable softmax and analytical dQ/dK/dV; query chunks bound CPU memory."""
    if type(chunk) is not int or not 1<=chunk<=128:raise ValueError('Oracle query chunk must be in [1,128].')
    q,k,v,g=(x.detach().cpu().double() for x in (q,k,v,cotangent))
    if q.shape!=k.shape or q.shape!=v.shape or q.shape!=g.shape or q.ndim!=4 or min(q.shape)<=0:
        raise ValueError('Require matching nonempty self-attention operands.')
    if not all(torch.isfinite(x).all() for x in (q,k,v,g)):raise ValueError('Nonfinite oracle operands.')
    out=torch.empty_like(q);dq=torch.zeros_like(q);dk=torch.zeros_like(k);dv=torch.zeros_like(v);scale=1/math.sqrt(q.shape[-1])
    with torch.no_grad():
        for b in range(q.shape[0]):
            for h in range(q.shape[1]):
                for start in range(0,q.shape[-2],chunk):
                    stop=min(start+chunk,q.shape[-2]);x=q[b,h,start:stop];dy=g[b,h,start:stop]
                    scores=(x@k[b,h].T)*scale;scores=scores-scores.max(dim=-1,keepdim=True).values
                    prob=scores.exp();prob=prob/prob.sum(dim=-1,keepdim=True)
                    out[b,h,start:stop]=prob@v[b,h]
                    dp=dy@v[b,h].T;ds=prob*(dp-(dp*prob).sum(dim=-1,keepdim=True))
                    dq[b,h,start:stop]=(ds@k[b,h])*scale
                    dk[b,h].add_((ds.T@x)*scale);dv[b,h].add_(prob.T@dy)
    return {'output':out,'dq':dq,'dk':dk,'dv':dv}


def iteration(qkv,cotangent,arm,device):
    with force_backend(arm):
        synchronize(device);tick=time.monotonic()
        output=F.scaled_dot_product_attention(*qkv,dropout_p=0.,is_causal=False,attn_mask=None)
        synchronize(device);middle=time.monotonic()
        gradients=torch.autograd.grad(output,qkv,grad_outputs=cotangent)
        synchronize(device);end=time.monotonic()
    return ({'output':output.detach(),**{n:t.detach() for n,t in zip(('dq','dk','dv'),gradients)}},
            {'forward_seconds':middle-tick,'backward_seconds':end-middle,'total_seconds':end-tick})


def compare(actual,expected):
    return {name:agreement(actual[name],expected[name],TOLERANCES[name]) for name in TOLERANCES}


def backend_evidence(rows,arm,device):
    if arm not in ARMS or device not in ('cpu','cuda') or not isinstance(rows,list) or not rows or any(
            not isinstance(r,dict) or not isinstance(r.get('name'),str) or r.get('device_type') not in ('cpu','cuda')
            or type(r.get('duration_us')) not in (int,float) or not math.isfinite(r['duration_us']) or r['duration_us']<0 for r in rows):
        raise ValueError('Invalid device-typed attention backend observations.')
    operators={r['name'] for r in rows if r['device_type']=='cpu'}
    kernels={r['name'] for r in rows if r['device_type']=='cuda' and r['duration_us']>0}
    if arm=='efficient':
        passed=(device=='cuda' and 'aten::_scaled_dot_product_efficient_attention' in operators
                and 'aten::_scaled_dot_product_efficient_attention_backward' in operators
                and any('AttentionKernel<' in k or k.startswith('fmha_cutlassF_') for k in kernels)
                and any('AttentionBackwardKernel<' in k or k.startswith('fmha_cutlassB_') for k in kernels)
                and not any('scaled_dot_product_attention_math' in k or 'scaled_dot_product_flash_attention' in k for k in operators))
    else:
        passed=('aten::_scaled_dot_product_attention_math' in operators
                and not any('scaled_dot_product_efficient_attention' in k or 'scaled_dot_product_flash_attention' in k for k in operators)
                and (device=='cpu' and not kernels or device=='cuda' and any('gemm' in k.lower() for k in kernels)
                     and not any('fmha_cutlass' in k or 'AttentionBackwardKernel<' in k for k in kernels)))
    return {'arm':arm,'device':device,'passed':bool(passed),'CPU_operators':sorted(operators),'CUDA_activity_names':sorted(kernels),
            'actual_CUDA_backend_observed':device=='cuda' and bool(passed)}

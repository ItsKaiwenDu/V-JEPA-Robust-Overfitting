"""Independent NumPy audit of saved operands, not a new execution policy.

Cross-platform FP32 regeneration is intentionally not asserted here. The
original generator/reviewer gates and failures remain separate evidence.
"""
import math

import numpy as np


KIND='attention-saved-operands-numpy-audit-v1'


def oracle(q,k,v,cotangent,*,chunk=128):
    if type(chunk) is not int or not 1<=chunk<=128:raise ValueError('Invalid oracle chunk bound.')
    q,k,v,g=[np.asarray(t,dtype=np.float64) for t in (q,k,v,cotangent)]
    if (q.ndim!=4 or min(q.shape)<=0 or any(t.shape!=q.shape for t in (k,v,g))
            or any(not np.isfinite(t).all() for t in (q,k,v,g))):
        raise ValueError('Require matching finite self-attention operands.')
    result={n:np.zeros(q.shape,dtype=np.float64) for n in ('output','dq','dk','dv')}
    scale=1/math.sqrt(q.shape[-1])
    for batch in range(q.shape[0]):
        for head in range(q.shape[1]):
            key=k[batch,head];value=v[batch,head]
            for start in range(0,q.shape[-2],chunk):
                end=min(start+chunk,q.shape[-2]);query=q[batch,head,start:end];dy=g[batch,head,start:end]
                scores=(query@key.T)*scale;scores-=np.max(scores,axis=-1,keepdims=True)
                probability=np.exp(scores);probability/=np.sum(probability,axis=-1,keepdims=True)
                result['output'][batch,head,start:end]=probability@value
                dprob=dy@value.T
                dscores=probability*(dprob-np.sum(dprob*probability,axis=-1,keepdims=True))
                result['dq'][batch,head,start:end]=(dscores@key)*scale
                result['dk'][batch,head]+=(dscores.T@query)*scale
                result['dv'][batch,head]+=probability.T@dy
    return result


def agreement(actual,expected,tolerance):
    actual=np.asarray(actual,dtype=np.float64);expected=np.asarray(expected,dtype=np.float64)
    if (actual.shape!=expected.shape or actual.size==0 or not np.isfinite(actual).all() or not np.isfinite(expected).all()
            or set(tolerance)!={'rtol','atol'} or any(not math.isfinite(v) or v<0 for v in tolerance.values())
            or tolerance['atol']<=0):raise ValueError('Invalid saved-tensor comparison.')
    error=np.abs(actual-expected);allowed=tolerance['atol']+tolerance['rtol']*np.abs(expected)
    return {'passed':bool(np.all(error<=allowed)),'failed_elements':int(np.count_nonzero(error>allowed)),
            'shape':list(actual.shape),'max_absolute_error':float(error.max()),
            'max_tolerance_ratio':float((error/allowed).max()),
            'relative_l2_error':float(np.linalg.norm(error.ravel())/max(float(np.linalg.norm(expected.ravel())),1e-300)),
            **tolerance}


def backend(rows,arm):
    if arm not in ('efficient','math') or not isinstance(rows,list) or not rows or len(rows)>20000 or any(
            not isinstance(r,dict) or not isinstance(r.get('name'),str) or r.get('device_type') not in ('cpu','cuda')
            or type(r.get('duration_us')) not in (float,int) or not math.isfinite(r['duration_us']) or r['duration_us']<0 for r in rows):
        raise ValueError('Malformed device-typed backend observations.')
    ops={r['name'] for r in rows if r['device_type']=='cpu'}
    kernels={r['name'] for r in rows if r['device_type']=='cuda' and r['duration_us']>0}
    if arm=='efficient':
        passed=(all(n in ops for n in ('aten::_scaled_dot_product_efficient_attention','aten::_scaled_dot_product_efficient_attention_backward'))
            and any(n.startswith('fmha_cutlassF_f32_') or 'AttentionKernel<' in n for n in kernels)
            and any(n.startswith('fmha_cutlassB_f32_') or 'AttentionBackwardKernel<' in n for n in kernels)
            and not any('scaled_dot_product_attention_math' in n or 'scaled_dot_product_flash_attention' in n for n in ops))
    else:
        passed=('aten::_scaled_dot_product_attention_math' in ops and 'aten::_softmax_backward_data' in ops
            and any('gemm' in n.lower() for n in kernels)
            and any('SoftMaxBackward' in n or 'softmax_warp_backward<' in n for n in kernels)
            and not any('scaled_dot_product_efficient_attention' in n or 'scaled_dot_product_flash_attention' in n for n in ops)
            and not any('fmha_cutlass' in n or 'AttentionBackwardKernel<' in n for n in kernels))
    return {'arm':arm,'passed':bool(passed),'CUDA_forward_and_backward_observed':bool(passed),
            'CUDA_kernel_names':sorted(kernels),'CPU_operators':sorted(ops)}


def differences(left,right,path=''):
    if isinstance(left,dict) and isinstance(right,dict):
        result=[]
        for key in sorted(set(left)|set(right)):
            if key not in left or key not in right:result.append({'path':path+'/'+key,'left':left.get(key),'right':right.get(key)})
            else:result+=differences(left[key],right[key],path+'/'+key)
        return result
    if isinstance(left,list) and isinstance(right,list) and len(left)==len(right):
        return [r for i,(a,b) in enumerate(zip(left,right)) for r in differences(a,b,path+'/'+str(i))]
    return [] if type(left)==type(right) and left==right else [{'path':path,'left':left,'right':right}]

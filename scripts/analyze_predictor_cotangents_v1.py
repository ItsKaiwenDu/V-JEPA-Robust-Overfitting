#!/usr/bin/env python3
"""Local saved-data analysis; no model, autograd, CUDA, transport or launch."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    if platform.node() in ('linux10603','linux10609'):
        raise ValueError('This local saved analysis has no university execution grant.')
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'runs/pretrained/diagnostic') or output.exists():
        raise ValueError('Fresh local raw diagnostic output required; preserve older evidence.')
    import torch
    def blocked(*a,**kw):raise RuntimeError('Saved analysis forbids model/autograd/SDPA/CUDA execution.')
    for obj,name in ((torch.nn.Module,'__call__'),(torch.autograd,'grad'),(torch.autograd,'backward'),
        (torch.Tensor,'backward'),(torch.nn.functional,'scaled_dot_product_attention'),(torch.cuda,'init'),(torch.cuda,'_lazy_init')):setattr(obj,name,blocked)
    def deny(event,args):
        if event.startswith(('subprocess.','socket.','os.fork','os.posix_spawn','os.exec')) or event=='os.system':raise RuntimeError('Saved analysis forbids process/network.')
    sys.addaudithook(deny)
    from models.predictor_cotangent_attribution_v1 import analyze_saved_target
    bundle,summary=analyze_saved_target(ROOT)
    output.mkdir(mode=0o700)
    data=(json.dumps(bundle,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    (output/'sparse-values.json').write_bytes(data)
    summary['sparse_values']={'path':str((output/'sparse-values.json').relative_to(ROOT)),
        'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    names=('models/predictor_cotangent_attribution_v1.py','scripts/analyze_predictor_cotangents_v1.py',
           'tests/test_predictor_cotangent_attribution_v1.py')
    summary['source_sha256']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    assert not torch.cuda.is_initialized()
    (output/'analysis.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'saved_analysis_complete':True,'changes':summary['changed_entries'],
        'full_array_comparisons':64,'sparse_values_bytes':len(data),'pins_reverified':summary['prior_inputs_reverified'],
        'target_VJP_cells_computed':False}))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Local-only concrete preparation; never writes authorization/claim or uses SSH."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models import predictor_pgd_target_operator_v1 as op
from models import predictor_pgd_target_client_v1 as client


def resource_bound(plan):
    lengths=[f[1]['shape'][1] for draw in plan['reference']['full_mask_tensor_identities'] for f in draw]
    # 37 typed arrays per objective, including actual pixels and full derivative.
    per=2*sum(lengths)*1024*4+(2048*1024+2*3*16*256*256)*4+8*2*2048*8+37*4096
    # Clean, selected adversary, two separately bounded CPU restart-noise arrays,
    # scoring masks/scalars and one full 16 MiB partial, all included concurrently.
    root=4*3*16*256*256*4+4*2*2048*8+9*4096
    raw=8*per+root+16*op.MiB
    r=plan['resource'];capture=sum(r[k] for k in ('raw_bytes','input_bytes','code_bytes','metadata_bytes','cache_bytes','emergency_bytes'))
    proof=r['proof_bytes']+r['code_bytes']+r['metadata_bytes']+r['emergency_bytes']
    if raw>r['raw_bytes'] or capture>r['capture_bytes'] or proof>r['setup_bytes']:
        raise ValueError('Full typed/partial/category/proof bound exceeds envelope.')
    return {'eight_objectives_plus_rollout_and_partial_bytes':raw,'raw_limit':r['raw_bytes'],
            'all_capture_category_caps_bytes':capture,'capture_ceiling':r['capture_bytes'],
            'proof_phase_category_bound_bytes':proof,'setup_ceiling':r['setup_bytes']}


def prepare(root,folder):
    root,folder=Path(root).resolve(),Path(folder).resolve()
    if folder.exists():raise FileExistsError('Preserve prior preparation.')
    text=(root/op.CONTRACT).read_text();plan=json.loads(text)
    if hashlib.sha256(text.encode()).hexdigest()!=op.CONTRACT_SHA:raise ValueError('Frozen proposal changed.')
    previous=root/plan['prior_inputs_manifest'];pins=json.loads(previous.read_bytes())
    if len(pins)!=plan['prior_sealed_files_count']:raise ValueError('Prior source coverage changed.')
    for n,w in pins.items():
        if op.old.digest(root/n)!=w:raise ValueError('Consumed source/evidence changed: '+n)
    for n,w in plan['prior_consumed_files'].items():
        if op.old.digest(root/n)!=w:raise ValueError('Consumed predecessor changed.')
    archive=plan['reference']['existing_packet_archive'];spec=json.loads((root/Path(archive).parent/'vjepa-eot-profile.tar.gz.manifest.json').read_bytes())
    if op.old.digest(root/archive)!=op.PACKET_SHA:raise ValueError('Unchanged packet required.')
    sources={n:op.VIRTUAL[n] if n in op.VIRTUAL else (root/n).read_bytes() for n in op.REMOTE_NAMES}
    key=(root/'runs/pretrained/diagnostic/attention-replay1-v1-20261105/linux10603/known-hosts').read_text()
    fp='SHA256:'+base64.b64encode(hashlib.sha256(base64.b64decode(key.split()[2],validate=True)).digest()).decode().rstrip('=')
    if fp!=op.base.SSH_FINGERPRINT:raise ValueError('Pinned public host key changed.')
    c={'kind':op.KIND,'host':'linux10603','parent':'/var/tmp','expected_device':66307,'expected_mount':'/',
       'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f','expected_driver':'580.178.04','expected_ssh_fingerprint':fp,
       'contract_sha256':op.CONTRACT_SHA,'contract_text':text,'contract':plan,'resource':plan['resource'],'paid_compute':False,'research_admitted':False,
       'authorization_present_at_preparation':False,'automatic_retries':0,'repository':str(root),'local_handoff_directory':str(folder),
       'archive_path':archive,'archive_manifest':spec,'known_hosts_sha256':hashlib.sha256(key.encode()).hexdigest(),
       'local_source_sha256':{n:op.old.digest(root/n) for n in sorted((set(op.REMOTE_NAMES)-set(op.VIRTUAL))|set(op.LOCAL_EXTRA))},
       'remote_source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in sources.items()},
       'prior_inputs_manifest':str(previous.relative_to(root)),'prior_inputs_manifest_sha256':op.old.digest(previous),
       'host_binding':{'prior_consumed_files':plan['prior_consumed_files']}}
    client.transport.ensure_previous_unarmed(root);op.validate_control(c);bootstrap=client.payload(c);bounds=resource_bound(plan)
    previous_mask=os.umask(0o077)
    try:
        folder.mkdir(mode=0o700,parents=True);op.old.write_json(folder/'CONTROL.json',c)
        (folder/'known-hosts').write_text(key)
        script='#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.quote(str(root/'.venv/bin/python'))+' -B '+shlex.quote(str(root/'scripts/run_predictor_pgd_target_v1.py'))+' --handoff '+shlex.quote(str(folder))+'\n'
        (folder/'run-attended.sh').write_text(script);(folder/'run-attended.sh').chmod(0o700)
        info={'kind':op.KIND,'control_sha256':op.old.digest(folder/'CONTROL.json'),'packet_bytes':spec['bytes'],
              'overlay_files':len(sources),'overlay_bytes':sum(map(len,sources.values())),'bootstrap_bytes':len(bootstrap),'storage_bound':bounds,
              'prior_sealed_files_verified':len(pins),'authorization_present':False,'attempt_present':False,'new_remote_work':False,
              'no_new_model_or_gradient_objective':True,'fresh_rollout_replay_in_scope':False}
        op.old.write_json(folder/'payload-review.json',info)
    finally:os.umask(previous_mask)
    return c,info

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(json.dumps(prepare(Path(__file__).resolve().parents[1],p.parse_args().output)[1],sort_keys=True))

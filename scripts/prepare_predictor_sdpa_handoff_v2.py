"""Prepare one unarmed native SDPA handoff locally. Never authorize or launch it."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models import predictor_sdpa_handoff_operator_v2 as op
from models import predictor_sdpa_handoff_client_v2 as client


def prepare(root,folder):
    root=Path(root).resolve();folder=Path(folder).resolve()
    if folder.exists():raise FileExistsError('Preserve previous prepared/claimed handoffs.')
    contract_path=root/'configs/predictor_sdpa_handoff_v2.json';text=contract_path.read_text();contract=json.loads(text)
    if op.old.digest(contract_path)!=op.CONTRACT_SHA:raise ValueError('Reviewed target contract changed.')
    for item in [contract[k] for k in ('prior_manifest','prior_consumed_claim','local_component_review','approved_original_worker')]+list(contract['repair_bindings'].values()):
        if op.old.digest(root/item['path'])!=item['sha256']:raise ValueError('Previous reviewed binding changed: '+item['path'])
    pins=json.loads((root/contract['prior_manifest']['path']).read_text())
    if len(pins)!=21507:raise ValueError('Prior sealed evidence coverage changed.')
    for name,want in pins.items():
        if op.old.digest(root/name)!=want:raise ValueError('Frozen source/evidence changed: '+name)
    client.pending_unarmed(root)
    proposal_text=(root/'configs/predictor_sdpa_fixed_input_v1.json').read_text()
    scientific=json.loads((root/'configs/predictor_attention_eot8_proposal_v1.json').read_text())
    archive=scientific['reference']['existing_packet_archive'];spec=json.loads((root/Path(archive).parent/'vjepa-eot-profile.tar.gz.manifest.json').read_text())
    if op.old.digest(root/archive)!=op.PACKET_SHA:raise ValueError('Original packet changed.')
    sources={n:op.INIT if n=='models/__init__.py' else (root/n).read_bytes() for n in op.REMOTE_NAMES}
    key=(root/'runs/pretrained/diagnostic/attention-replay1-v1-20261105/linux10603/known-hosts').read_text()
    local=set(op.REMOTE_NAMES)-{'models/__init__.py'}|set(op.LOCAL_EXTRA)
    c={'kind':op.KIND,'host':'linux10603','parent':'/var/tmp','expected_device':66307,'expected_mount':'/',
        'expected_gpu':'NVIDIA GeForce RTX 3070','expected_gpu_uuid':'GPU-e54fae90-feeb-8106-d8c0-1974fe48555f','expected_driver':'580.178.04',
        'expected_ssh_fingerprint':op.base.SSH_FINGERPRINT,'contract_sha256':op.CONTRACT_SHA,'prior_consumed_claim_sha256':op.PRIOR_CLAIM_SHA,
        'paid_compute':False,'research_admitted':False,'authorization_present_at_preparation':False,'automatic_retries':0,'resource':op.RESOURCE,
        'contract_text':text,'contract':contract,'proposal_text':proposal_text,'proposal':json.loads(proposal_text),
        'repository':str(root),'local_handoff_directory':str(folder),'archive_path':archive,'archive_manifest':spec,
        'known_hosts_sha256':hashlib.sha256(key.encode()).hexdigest(),
        'local_source_sha256':{n:op.old.digest(root/n) for n in sorted(local)},
        'remote_source_sha256':{n:hashlib.sha256(data).hexdigest() for n,data in sources.items()}}
    op.validate_control(c);bootstrap=client.payload(c)
    folder.mkdir(mode=0o700,parents=True);op.old.write_json(folder/'CONTROL.json',c);(folder/'known-hosts').write_text(key)
    script='#!/usr/bin/env bash\nset -euo pipefail\nexec caffeinate -i '+client.shlex.quote(str(root/'.venv/bin/python'))+' -B '+client.shlex.quote(str(root/'scripts/run_predictor_sdpa_handoff_v2.py'))+' --handoff '+client.shlex.quote(str(folder))+'\n'
    (folder/'run-attended.sh').write_text(script);(folder/'run-attended.sh').chmod(0o700)
    info={'kind':'unarmed-native-fixed-SDPA-payload-review-v2','control_sha256':op.old.digest(folder/'CONTROL.json'),
        'packet_bytes':17192095,'packet_members':99,'packet_uncompressed_bytes':26446958,
        'overlay_files':len(sources),'overlay_bytes':sum(map(len,sources.values())),'stdin_bootstrap_bytes':len(bootstrap),
        'prior_pins_verified':len(pins),'authorization_present':False,'attempt_present':False,'new_remote_or_university_work':False}
    op.old.write_json(folder/'payload-review.json',info);client.check_local(folder)
    return c,info


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1],a.output)[1],sort_keys=True))

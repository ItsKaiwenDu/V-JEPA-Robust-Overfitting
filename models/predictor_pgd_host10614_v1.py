"""Cold new-host PGD source/handoff assembly; never grants or starts target work."""
import ast
import base64
import copy
import hashlib
import os
from pathlib import Path
import stat
from models import predictor_pgd_close_handoff_v1 as helpers
from models import predictor_pgd_host10609_v1 as literals

IDENTITY='configs/university_host10614_identity_v1.json'
BASE='runs/pretrained/diagnostic/pgd-close-handoff-v1-20261117/handoff-parent/linux10609-final-unarmed'
FROZEN={BASE+'/CONTROL-CANDIDATE.json':'9c6670f7495fe9cec5faf219d0ef7664b2b49bc89a197f23ad4315f74a7a112b',
        BASE+'/SOURCE-PAYLOAD.json':'2ffd2f2de2a56a3c844d2f0ca7d51d0bfdef8f72d0f9fa9de52d38d06b4620e5',
        'models/predictor_pgd_close_outer_v1.py':'ab1b588cca7e3d76f34c54a802bd6681db80416c32570c63532198c0d760f684',
        'models/predictor_pgd_close_client_v1.py':'3a39abad4f7c2a730f7061317042198e140068c4c352692affdf2b99d9e7cde4',
        'scripts/run_predictor_pgd_close_v1.py':'83e5be906445017ee998a5553257be506b955ddfdd4cec829816ec4ebfcb441a'}
HOST='linux10614';OLD_HOST='linux10609';OLD_UUID='GPU-4e2a7c77-19bd-689f-6c08-60eb774c0d52'
NEW={'builder':'models/predictor_pgd_host10614_v1.py','outer':'models/predictor_pgd_host10614_outer_v1.py',
     'client':'models/predictor_pgd_host10614_client_v1.py','cli':'scripts/run_predictor_pgd_host10614_v1.py',
     'boot':'models/predictor_pgd_host10614_bootstrap_v1.py','scratch':'models/predictor_pgd_host10614_scratch_v1.py',
     'collection':'models/predictor_pgd_host10614_collection_v1.py'}
CPU='overlay/models/predictor_forward_preserving_cpu_v1.py'
UPSTREAM='overlay/models/predictor_forward_preserving_upstream_v1.py'
TARGET='overlay/models/predictor_forward_preserving_target_v1.py'
COORD='supervisor/predictor_pgd_split_coordinator_v1.py';SCRATCH='supervisor/predictor_pgd_split_scratch_v1.py';BOOT='supervisor/bootstrap.py'
EXPECTED_RUNTIME={'overlay/attacks/pgd_pretrained_v1.py','overlay/models/_pgd_split_facade_v1.py',
 'overlay/models/_pgd_split_numerical_v1.py','overlay/models/_pgd_split_observer_v1.py','overlay/models/_pgd_split_records_v1.py',
 'overlay/models/_pgd_split_route_v1.py',CPU,UPSTREAM,TARGET,'overlay/models/predictor_pgd_split_lineage_v1.py'}
CONTROL_KIND='pgd-host10614-outer-control-v1';AUTH_KIND='pgd-host10614-outer-university-exception-v1'
encoded,digest=helpers.encoded,helpers.digest


def identity(repository):
    data=literals.read(Path(repository)/IDENTITY,cap=16384);v=literals.parsed(data)
    expected={'hostname':HOST,'python':'3.11.17','uid':2170,'scratch_device':66307,'scratch_owner':0,
              'scratch_mode':0o1777,'scratch_path':'/var/tmp','filesystem':'xfs','mount':'/',
              'gpu_name':'NVIDIA GeForce RTX 3070','memory_total_MiB':8192,'driver':'580.178.04'}
    if any(type(v.get(k)) is not type(w) or v[k]!=w for k,w in expected.items()):raise ValueError('Exact supplied 10614 identity required.')
    if type(v.get('home_device')) is not int or v['home_device']==v['scratch_device']:raise ValueError('Separate home and local scratch required.')
    uuid=v.get('gpu_uuid');key=v.get('public_key','').split()
    if type(uuid) is not str or len(uuid)!=40 or not uuid.startswith('GPU-') or any(c not in '0123456789abcdef-' for c in uuid[4:]):raise ValueError('Typed new GPU UUID required.')
    if len(key)!=2 or key[0]!='ssh-ed25519':raise ValueError('One supplied public host key required.')
    blob=base64.b64decode(key[1],validate=True)
    if len(blob)!=51 or blob[:19]!=b'\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20':raise ValueError('Exact ED25519 public-key encoding required.')
    fp='SHA256:'+base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip('=')
    return v,fp,digest(data)


def retarget_runtime(data):
    """Change host literals and strengthen only positive tiny-host exclusions.

    No NotIn allowlist is expanded. The separate old projection launcher stays
    unchanged and cannot acquire authorization for a new host by this recipe.
    """
    tree=ast.parse(data);parents={id(c):n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    lines=data.splitlines(keepends=True);offset=[0]
    for line in lines:offset.append(offset[-1]+len(line))
    edits=[];covered=set()
    def edit(node,replacement,kind):
        begin=offset[node.lineno-1]+node.col_offset;end=offset[node.end_lineno-1]+node.end_col_offset
        edits.append((begin,end,data[begin:end],replacement,kind))
    for node in ast.walk(tree):
        if not isinstance(node,(ast.Tuple,ast.List)):continue
        if not all(isinstance(n,ast.Constant) and type(n.value) is str for n in node.elts):continue
        names=[n.value for n in node.elts]
        if set(names)!={'linux10603','linux10609'}:continue
        parent=parents[id(node)]
        if not isinstance(parent,ast.Compare) or len(parent.ops)!=1 or not isinstance(parent.ops[0],ast.In) or node not in parent.comparators:
            raise ValueError('Only positive university exclusions may expand; never broaden an allowlist.')
        replacement=repr(tuple(names+[HOST]) if isinstance(node,ast.Tuple) else names+[HOST]).encode()
        edit(node,replacement,'tiny-host-exclusion');covered.update(id(v) for v in node.elts)
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and id(node) not in covered and type(node.value) is str and OLD_HOST in node.value:
            edit(node,repr(node.value.replace(OLD_HOST,HOST)).encode(),'execution-host-literal')
    edits.sort();parts=[];cursor=0;newpos=0;trace=[]
    for begin,end,old,new,kind in edits:
        if begin<cursor:raise ValueError('Overlapping host edits.')
        prefix=data[cursor:begin];parts.extend((prefix,new));newpos+=len(prefix)
        trace.append({'new_begin':newpos,'new_end':newpos+len(new),'old_literal':old.decode(),'new_literal':new.decode(),'kind':kind})
        newpos+=len(new);cursor=end
    parts.append(data[cursor:]);result=b''.join(parts);compile(result,'<UNEXECUTED-HOST10614-RUNTIME>','exec')
    reverse=[];cursor=0
    for e in trace:reverse.extend((result[cursor:e['new_begin']],e['old_literal'].encode()));cursor=e['new_end']
    reverse.append(result[cursor:])
    if b''.join(reverse)!=data:raise ValueError('Exact runtime host byte inverse failed.')
    return result,trace


def assemble(repository):
    root=Path(repository);host,fp,identity_sha=identity(root)
    files={n:literals.read(root/n,wanted=w) for n,w in FROZEN.items()}
    c=literals.parsed(files[BASE+'/CONTROL-CANDIDATE.json']);b=literals.parsed(c['supervised_control_text']);p=literals.parsed(b['policy_text'])
    descriptors={'supervisor/'+n:d for n,d in b['supervisor_sources'].items()};descriptors.update({'overlay/'+n:d for n,d in c['runtime_files'].items()})
    original={n:base64.b64decode(v,validate=True) for n,v in literals.parsed(files[BASE+'/SOURCE-PAYLOAD.json']).items()}
    if set(original)!=set(descriptors) or len(original)!=69:raise ValueError('Exact prior 69-source namespace required.')
    for n,data in original.items():
        if descriptors[n]!={'bytes':len(data),'sha256':digest(data)}:raise ValueError('Prior source changed: '+n)
    sources=dict(original);traces={}
    for n in sorted(EXPECTED_RUNTIME):
        sources[n],traces[n]=retarget_runtime(original[n])
        if not traces[n]:raise ValueError('Expected new host guard missing: '+n)
    # CPU-only exclusion changes alter source hashes used by target ownership
    # guards. Rebind this two-edge chain, without changing attention/value math.
    sources[UPSTREAM],proto_trace=literals.constants(sources[UPSTREAM],{digest(original[CPU]):digest(sources[CPU])})
    sources[TARGET],route_trace=literals.constants(sources[TARGET],{digest(original[UPSTREAM]):digest(sources[UPSTREAM])})
    if len(proto_trace)!=1 or len(route_trace)!=1:raise ValueError('Exact two source-ownership pin edges required.')
    sources[COORD],coord_trace=literals.constants(original[COORD],{OLD_HOST:HOST,OLD_UUID:host['gpu_uuid']})
    oldct=literals.parsed(p['scratch_contract_text']);ct=copy.deepcopy(oldct)
    runtime={n.removeprefix('overlay/'):{'bytes':len(v),'sha256':digest(v)} for n,v in sources.items() if n.startswith('overlay/')}
    manifest={'kind':'pgd-host10614-runtime-manifest-v1','prior_runtime_manifest_sha256':oldct['runtime_manifest_sha256'],'runtime_files':runtime}
    ct['runtime_files']=runtime;ct['runtime_manifest_sha256']=digest(encoded(manifest))
    sources[SCRATCH],scratch_trace=literals.constants(original[SCRATCH],{p['scratch_contract_sha256']:digest(encoded(ct))})
    policy=copy.deepcopy(p);policy['scope']['host']['hostname']=HOST;policy['scope']['host']['gpu_uuid']=host['gpu_uuid']
    policy['scratch_contract_text']=encoded(ct).decode();policy['scratch_contract_sha256']=digest(encoded(ct))
    policy['runtime_repair']['runtime_manifest_text']=encoded(manifest).decode()
    policy['host10614_binding']={'kind':'pgd-host10614-execution-binding-v1','prior_policy_sha256':b['policy_sha256'],
       'host_identity_sha256':identity_sha,'execution_host':HOST,'GPU_UUID':host['gpu_uuid'],'public_host_fingerprint':fp,
       'current_runtime_manifest_sha256':digest(encoded(manifest)),'historical_host_change_is_provenance':True,'no_old_claims_or_results_relabelled':True}
    policy_bytes=encoded(policy)
    sources[BOOT],boot_trace=literals.constants(original[BOOT],{b['policy_sha256']:digest(policy_bytes)})
    control=encoded({'kind':b['kind'],'policy_text':policy_bytes.decode(),'policy_sha256':digest(policy_bytes),
                     'supervisor_sources':{n.removeprefix('supervisor/'):{'bytes':len(v),'sha256':digest(v)} for n,v in sources.items() if n.startswith('supervisor/')}})
    changed=sorted(n for n in sources if sources[n]!=original[n])
    if set(changed)!=EXPECTED_RUNTIME|{COORD,SCRATCH,BOOT}:raise ValueError('Unexpected host-change source surface.')
    if policy['base_text']!=p['base_text'] or policy['design_text']!=p['design_text'] or policy['fixed_support']!=p['fixed_support']:
        raise ValueError('Scientific/financial reference bytes or verified monitor changed.')
    scope=copy.deepcopy(policy['scope']);scope['host']=p['scope']['host']
    if scope!=p['scope']:raise ValueError('Only current host/UUID authority may change in scope.')
    for n,data in sources.items():compile(data,'<UNEXECUTED-HOST10614-'+n+'>','exec')
    readers={NEW['boot']:sources[BOOT],NEW['scratch']:sources[SCRATCH]}
    collector=original['supervisor/predictor_pgd_split_collection_v1.py']
    for old,new in ((b'predictor_pgd_split_bootstrap_v1 as boot',b'predictor_pgd_host10614_bootstrap_v1 as boot'),(b'predictor_pgd_split_scratch_v1 as scratch',b'predictor_pgd_host10614_scratch_v1 as scratch')):
        collector=helpers.replace_once(collector,old,new)
    readers[NEW['collection']]=collector
    info={'kind':'pgd-host10614-source-binding-v1','identity_sha256':identity_sha,'fingerprint':fp,'runtime_edits':traces,
          'prototype_pin_edits':proto_trace,'target_pin_edits':route_trace,'coordinator_edits':coord_trace,'scratch_edits':scratch_trace,'bootstrap_edits':boot_trace,
          'policy_sha256':digest(policy_bytes),'bootstrap_sha256':digest(sources[BOOT]),'scratch_contract_sha256':digest(encoded(ct)),
          'supervised_control_sha256':digest(control),'runtime_manifest_sha256':digest(encoded(manifest)),
          'changed_uploaded_sources':changed,'unchanged_uploaded_sources':69-len(changed),'source_files':69,'source_bytes':sum(map(len,sources.values())),
          'tiny_guard_exclusions_extended_without_broadening_projection_allowlist':True,'new_authorization_or_target_work':False,
          'scientific_qualified':False,'training_admitted':False}
    return control,sources,readers,manifest,info,files


def build_local(repository):
    control,sources,readers,manifest,info,files=assemble(repository);host,fp,_=identity(repository)
    old_outer=files['models/predictor_pgd_close_outer_v1.py'];old_client=files['models/predictor_pgd_close_client_v1.py'];old_cli=files['scripts/run_predictor_pgd_close_v1.py']
    old_policy=literals.parsed(literals.parsed(literals.parsed(files[BASE+'/CONTROL-CANDIDATE.json'])['supervised_control_text'])['policy_text'])
    old_key='linux10609.dc.engr.scu.edu '+host['public_key']+'\n';new_key=HOST+'.dc.engr.scu.edu '+host['public_key']+'\n'
    # This university image uses the same supplied ED25519 public key; it is
    # checked from user-provided public data, not assumed from the old machine.
    if helpers.assignment(old_outer,'KNOWN_HOSTS') is None:raise ValueError('Known host seed missing.')
    oldb=literals.parsed(literals.parsed(files[BASE+'/CONTROL-CANDIDATE.json'])['supervised_control_text'])
    replacements={oldb['policy_sha256']:info['policy_sha256'],oldb['supervisor_sources']['bootstrap.py']['sha256']:info['bootstrap_sha256'],
                  'pgd-close-outer-control-v1':CONTROL_KIND,'pgd-close-outer-university-exception-v1':AUTH_KIND,
                  'linux10609.dc.engr.scu.edu':HOST+'.dc.engr.scu.edu',old_key:new_key,OLD_HOST:HOST,
                  'models/predictor_pgd_close_outer_v1.py':NEW['outer']}
    outer,outer_trace=literals.constants(old_outer,replacements)
    previous_pins=ast.literal_eval(helpers.assignment(old_outer,'LOCAL_SOURCES').value)
    local_sources=(NEW['outer'],NEW['client'],NEW['cli'],NEW['builder'],NEW['boot'],NEW['scratch'],NEW['collection'],IDENTITY,
                   BASE+'/CONTROL-CANDIDATE.json',BASE+'/SOURCE-PAYLOAD.json')+previous_pins
    if len(set(local_sources))!=len(local_sources):raise ValueError('Distinct complete local pins required.')
    outer,value_trace=helpers.replace_value(outer,'LOCAL_SOURCES',local_sources)
    outer=helpers.replace_once(outer,b'from models import predictor_pgd_close_handoff_v1 as binding',b'from models import predictor_pgd_host10614_v1 as binding')
    client=old_client
    for old,new in ((b'predictor_pgd_close_outer_v1 as outer',b'predictor_pgd_host10614_outer_v1 as outer'),
                    (b'predictor_pgd_close_bootstrap_v1 as boot',b'predictor_pgd_host10614_bootstrap_v1 as boot'),
                    (b'predictor_pgd_close_collection_v1 as collection',b'predictor_pgd_host10614_collection_v1 as collection')):
        client=helpers.replace_once(client,old,new)
    cli=helpers.replace_once(old_cli,b'from models.predictor_pgd_close_client_v1 import run',b'from models.predictor_pgd_host10614_client_v1 import run')
    generated={**readers,NEW['outer']:outer,NEW['client']:client,NEW['cli']:cli}
    for n,data in generated.items():compile(data,'<UNEXECUTED-HOST10614-LOCAL-'+n+'>','exec')
    info.update(outer_literal_edits=outer_trace,local_source_assignment=value_trace,local_source_files=len(local_sources))
    return generated,info


def prepare_package(repository,destination):
    from models import predictor_pgd_host10614_outer_v1 as outer
    from models import predictor_pgd_split_source_bundle_v1 as reader
    root,dest=Path(repository),Path(destination);parent=dest.parent.lstat()
    if (root.resolve(strict=True)!=root or not root.is_absolute() or not dest.is_absolute() or dest.parent.resolve(strict=True)!=dest.parent
         or os.path.lexists(dest) or parent.st_uid!=os.getuid() or stat.S_IMODE(parent.st_mode)!=0o700):raise ValueError('Fresh private local handoff required.')
    generated,meta=build_local(root)
    for n,data in generated.items():literals.read(root/n,wanted=digest(data))
    control,sources,readers,manifest,details,_=assemble(root);b=literals.parsed(control);p=literals.parsed(b['policy_text'])
    outer.verify_directory_policy_sources(root,reader);outer.verify_refusal_policy_sources(root,reader)
    pins={n:digest(literals.read(root/n)) for n in outer.LOCAL_SOURCES}
    c={'kind':CONTROL_KIND,'supervised_control_text':control.decode(),'supervised_control_sha256':digest(control),
       'outer_source_sha256':pins[NEW['outer']],'local_source_sha256':pins,'runtime_files':manifest['runtime_files'],
       'repository':str(root),'local_handoff':str(dest),'packet_path':outer.PACKET_PATH,'known_hosts_text':outer.KNOWN_HOSTS,
       'known_hosts_sha256':digest(outer.KNOWN_HOSTS.encode()),'expected_ssh_fingerprint':outer.FINGERPRINT,
       'financial_state_unchanged':True,'authorized_at_preparation':False,'target_helper_reviewed':False}
    raw=encoded(c);outer.validate_control(raw,pins[NEW['outer']])
    payload=encoded({n:base64.b64encode(v).decode() for n,v in sources.items()});outer.decode_sources(literals.parsed(payload),c,b)
    if len(payload)+len(raw)+65536>2*2**20:raise ValueError('Original upload frame cap exceeded.')
    result={'kind':'pgd-host10614-unarmed-package-v1','outer_control_sha256':digest(raw),'supervised_control_sha256':digest(control),
            'source_files':69,'source_bytes':sum(map(len,sources.values())),'source_envelope_bytes':len(payload),'local_source_files':len(pins),
            'source_binding':meta,'scope':p['scope'],'authorization_created':False,'target_work_or_SSH_executed':False,
            'scientific_qualified':False,'training_admitted':False}
    dest.mkdir(mode=0o700)
    for n,data in (('CONTROL-CANDIDATE.json',raw),('SOURCE-PAYLOAD.json',payload),('known-hosts',outer.KNOWN_HOSTS.encode()),('PACKAGE.json',encoded(result))):
        outer.private_write(dest,n,data,lambda:None)
    return result

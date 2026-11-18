"""Cold integration of the reviewed JSON scanner; no target or launch API.

Complete 69-source/control/reader candidates only. No packet/checkpoint contents,
scientific imports, source activation, authorization, process, proof or network.
"""
import ast
import base64
import copy
import hashlib
from pathlib import Path
from models import predictor_pgd_host10609_v1 as literals
from models import predictor_pgd_publication_scan_v1 as scanner

FOLDER='runs/pretrained/diagnostic/pgd-host10614-v1-20261117/handoff-parent/linux10614-final-unarmed'
CONTROL=FOLDER+'/CONTROL-CANDIDATE.json';PAYLOAD=FOLDER+'/SOURCE-PAYLOAD.json'
TEMPLATE='models/predictor_pgd_publication_scratch_v1.py'
FROZEN={CONTROL:'251fb58811892d8c76faf2f29cbe87b1526ebaa82e69d44cd069a8c40bf79712',
 PAYLOAD:'673641e170e918fa13b69dfe95aa91b2557d773c8df60b26582b8a87881c4f3a',
 TEMPLATE:'361725052784e02c5e2a18c8eb4f3d05c0a25db125890ce08f5bebf1f5e0d102',
 'models/predictor_pgd_publication_scan_v1.py':'c2e7ae0d6a157a57bb602162cc411039a3eaafc4cc614d62c24247295829fff5'}
SCRATCH='supervisor/predictor_pgd_split_scratch_v1.py';BOOT='supervisor/bootstrap.py'
COLLECTION='supervisor/predictor_pgd_split_collection_v1.py';COORDINATOR='supervisor/predictor_pgd_split_coordinator_v1.py'
READERS={'bootstrap':'models/predictor_pgd_publication_bootstrap_v1.py',
 'scratch':'models/predictor_pgd_publication_bound_scratch_v1.py',
 'collection':'models/predictor_pgd_publication_collection_v1.py'}
RULE={'kind':'bounded-mutable-json-publication-v1','phase':'capture','initial':False,'settled':False,
 'allowed_spaces':['metadata','result'],'category':'metadata','suffix':'.json.partial',
 'mutable_required':True,'partial_allowed_required':True,'maximum_typed_bytes':262144,
 'maximum_resolutions_per_scan':2,'directory_walks':1,'directory_rescans_or_polling':0,
 'listed_inode_matches_held_and_named_final':True,'private_same_device_uid_regular_0600_single_link_required':True,
 'held_named_size_mtime_and_ancestry_before_after_required':True,'absent_partial_required':True,
 'verified_source_pins_excluded':True,'another_publication_or_reappearance_refuses':True,
 'logical_reserve':'full typed metadata cap per resolution',
 'allocated_reserve':'max(typed cap,observed held/named/superseded/partial allocation)',
 'metadata_partial_slots_shared':2,'extra_metadata_bytes_shared':524288,'capture_ceiling_bytes':2415919104,
 'metadata_plus_trace_bytes_shared':67108864,'file_slots_including_resolutions':1024,'walk_node_cap':1280,
 'collection_setup_proof_and_settled_scans_strict':True,'accounting_field_is_capture_only':'publication_accounting',
 'atomic_snapshot_kernel_quota_continuous_bound_or_computation_authentication':False}
encoded,digest=literals.encoded,literals.digest


def rewrite(source,mapping):
    result,trace=literals.constants(source,mapping)
    if {ast.literal_eval(e['old_literal']) for e in trace}!=set(mapping):raise ValueError('Every dependent literal pin, and only those pins, must change.')
    return result,trace


def load_frozen(repository):
    root=Path(repository);files={n:literals.read(root/n,wanted=w) for n,w in FROZEN.items()}
    c=literals.parsed(files[CONTROL]);bc=literals.parsed(c['supervised_control_text']);p=literals.parsed(bc['policy_text'])
    if (digest(c['supervised_control_text'].encode())!=c['supervised_control_sha256']
            or digest(bc['policy_text'].encode())!=bc['policy_sha256']):raise ValueError('Consumed control joins changed.')
    descriptors={'overlay/'+n:d for n,d in c['runtime_files'].items()}
    descriptors.update({'supervisor/'+n:d for n,d in bc['supervisor_sources'].items()})
    mapping=literals.parsed(files[PAYLOAD])
    if set(mapping)!=set(descriptors) or len(mapping)!=69:raise ValueError('Exact prior63runtime/6supervisor namespace required.')
    sources={}
    for name,value in mapping.items():
        data=base64.b64decode(value,validate=True)
        if descriptors[name]!={'bytes':len(data),'sha256':digest(data)}:raise ValueError('Consumed source changed: '+name)
        sources[name]=data
    return c,bc,p,sources,files


def assemble(repository):
    c,bc,oldp,original,files=load_frozen(repository);sources=dict(original)
    template,scan_meta=scanner.build(original[SCRATCH])
    if template!=files[TEMPLATE]:raise ValueError('Exact reviewed full scanner template required.')
    contract=literals.parsed(oldp['scratch_contract_text']);prior_contract=copy.deepcopy(contract)
    contract['publication_observation']=copy.deepcopy(RULE)
    contract_bytes=encoded(contract);contract_sha=digest(contract_bytes)
    bound,edit=rewrite(template,{oldp['scratch_contract_sha256']:contract_sha})
    sources[SCRATCH]=bound
    policy=copy.deepcopy(oldp);policy['scratch_contract_text']=contract_bytes.decode();policy['scratch_contract_sha256']=contract_sha
    policy['fixed_support']['predictor_pgd_split_scratch_v1.py']=digest(bound)
    policy['publication_scan_repair']={'kind':'pgd-publication-integration-v1','prior_policy_sha256':bc['policy_sha256'],
      'frozen_scanner_template_sha256':FROZEN[TEMPLATE],'bound_scanner_sha256':digest(bound),
      'recipe_sha256':FROZEN['models/predictor_pgd_publication_scan_v1.py'],
      'current_contract_sha256':contract_sha,'rules':copy.deepcopy(RULE),
      'all_numerical_host_resource_clock_and_financial_scope_unchanged':True,
      'journal_and_collector_bytes_unchanged':True,'only_source_binding_no_authorization':True}
    policy_bytes=encoded(policy);policy_sha=digest(policy_bytes)
    sources[BOOT],boot_edit=rewrite(original[BOOT],{bc['policy_sha256']:policy_sha})
    expected={SCRATCH,BOOT};changed={n for n in sources if sources[n]!=original[n]}
    if changed!=expected:raise ValueError('Only scanner and bootstrap dependent pin may change.')
    for key in oldp:
        if key not in ('scratch_contract_text','scratch_contract_sha256','fixed_support') and policy[key]!=oldp[key]:
            raise ValueError('Prior scientific/reference/financial scope changed: '+key)
    fixed=dict(policy['fixed_support']);fixed.pop('predictor_pgd_split_scratch_v1.py')
    if fixed!=oldp['fixed_support']:raise ValueError('Original monitor/proof support pins changed.')
    comparable=copy.deepcopy(contract);comparable.pop('publication_observation')
    if comparable!=prior_contract:raise ValueError('Original whole-scratch ceilings or source catalogue changed.')
    control=encoded({'kind':bc['kind'],'policy_text':policy_bytes.decode(),'policy_sha256':policy_sha,
      'supervisor_sources':{n.removeprefix('supervisor/'):{'bytes':len(v),'sha256':digest(v)} for n,v in sources.items() if n.startswith('supervisor/')}})
    if len(control)>256*1024 or sum(map(len,sources.values()))>2*2**20:raise ValueError('Original control/source envelope caps exceeded.')
    readers={READERS['bootstrap']:sources[BOOT],READERS['scratch']:sources[SCRATCH]}
    collector=sources[COLLECTION]
    for old,new in ((b'predictor_pgd_split_bootstrap_v1 as boot',b'predictor_pgd_publication_bootstrap_v1 as boot'),
                    (b'predictor_pgd_split_scratch_v1 as scratch',b'predictor_pgd_publication_bound_scratch_v1 as scratch')):
        if collector.count(old)!=1:raise ValueError('Exact local collector import bridge required.')
        collector=collector.replace(old,new,1)
    readers[READERS['collection']]=collector
    for name,data in {**sources,**readers}.items():compile(data,'<UNEXECUTED-PUBLICATION-BINDING-'+name+'>','exec')
    metadata={'kind':'pgd-publication-integration-v1-source-candidate','prior_outer_control_sha256':FROZEN[CONTROL],
     'prior_supervised_control_sha256':c['supervised_control_sha256'],'prior_policy_sha256':bc['policy_sha256'],
     'supervised_control_sha256':digest(control),'policy_sha256':policy_sha,'scratch_contract_sha256':contract_sha,
     'scanner_template_sha256':digest(template),'bound_scanner_sha256':digest(bound),'bootstrap_sha256':digest(sources[BOOT]),
     'source_literal_edits':{SCRATCH:edit,BOOT:boot_edit},'scanner_derivation':scan_meta,
     'changed_uploaded_sources':sorted(changed),'unchanged_uploaded_sources':67,'source_files':69,'source_bytes':sum(map(len,sources.values())),
     'runtime_files':63,'reader_pins':{n:digest(v) for n,v in readers.items()},
     'full_scope_and_reference_financial_bytes_unchanged':True,'journal_collection_monitor_writer_numerical_proof_bytes_unchanged':True,
     'runtime_manifest_unchanged':True,'new_fixed_scanner_support_pin':True,'source_bundle_ready':True,
     'runnable_outer_client_or_target_handoff_prepared':False,'authorized':False,'scientific_qualified':False,'training_admitted':False}
    return control,sources,readers,literals.parsed(policy['runtime_repair']['runtime_manifest_text']),metadata


def verify_readers(repository,readers):
    if set(readers)!=set(READERS.values()):raise ValueError('Exact three integrated local readers required.')
    for name,data in readers.items():
        if literals.read(Path(repository)/name,wanted=digest(data))!=data:raise ValueError('Local reader differs from exact server bytes: '+name)
    return {'kind':'pgd-publication-local-reader-binding-v1','reader_files':3,'source_identity_verified':True,
     'scientific_qualified':False,'training_admitted':False}

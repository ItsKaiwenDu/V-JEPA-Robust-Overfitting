"""Cold complete source binding for raw observation and priority collection.

No packet/data decode, scientific import, grant, process, target or network API.
The six supervisor slots and 63 runtime files remain the full pinned graph.
"""
import ast,base64,copy,io,tokenize
from pathlib import Path
from models import predictor_pgd_host10609_v1 as literals
from models import predictor_pgd_raw_publication_scan_v1 as scanner
FOLDER='runs/pretrained/diagnostic/pgd-publication-handoff-v1-20261117/handoff-parent/linux10614-final-unarmed'
CONTROL=FOLDER+'/CONTROL-CANDIDATE.json';PAYLOAD=FOLDER+'/SOURCE-PAYLOAD.json'
TEMPLATE='models/predictor_pgd_raw_publication_scratch_v1.py'
PROTOCOL='models/predictor_pgd_priority_collection_v1.py'
FROZEN={'runs/pretrained/diagnostic/pgd-publication-handoff-v1-20261117/handoff-parent/linux10614-final-unarmed/CONTROL-CANDIDATE.json': '3f7b6ee518807ee36e330805082f28ea44a069bdf6cd59593804ea745a3ceb16', 'runs/pretrained/diagnostic/pgd-publication-handoff-v1-20261117/handoff-parent/linux10614-final-unarmed/SOURCE-PAYLOAD.json': 'b152a652de9c603eae5c770d768a780a8ebbad70d82d8e1336dc13d06a740022', 'models/predictor_pgd_raw_publication_scan_v1.py': '81bafe2a9293a4d8dc5cee6d63e859404a4ed212a80ee72b7b761676b5cfb24d', 'models/predictor_pgd_raw_publication_scratch_v1.py': 'a85a55bab3e76229e7dc84aadbaf919fbe08b86853200b9d646a9d1687698c57', 'models/predictor_pgd_priority_collection_v1.py': 'a59eda60794b83deaded19ba43b4292826491f78e5ae6eeda904e29a42e589f3'}
SCRATCH='supervisor/predictor_pgd_split_scratch_v1.py';BOOT='supervisor/bootstrap.py'
COLLECTION='supervisor/predictor_pgd_split_collection_v1.py';COORDINATOR='supervisor/predictor_pgd_split_coordinator_v1.py'
READERS={'bootstrap':'models/predictor_pgd_joint_bootstrap_v1.py','scratch':'models/predictor_pgd_joint_scratch_v1.py','collection':'models/predictor_pgd_joint_collection_v1.py'}
encoded,digest=literals.encoded,literals.digest
RAW_RULE={'kind':'bounded-immutable-raw-publication-v1','phase':'capture','initial':False,'settled':False,
 'category':'raw','space':'result','mutable':False,'partial_allowed':True,'suffixes':['.npy.partial','.pt.partial'],
 'maximum_typed_bytes':16*2**20,'maximum_raw_resolutions':1,'maximum_JSON_resolutions':2,'one_directory_walk_no_polling':True,
 'cached_inode_private_held_named_identity_and_ancestry_required':True,'exact_two_name_two_link_to_single_final_only':True,
 'partial_cannot_reappear_after_absence':True,'one_link_partial_growth_bounded_no_shrink':True,'linked_final_bytes_mtime_immutable':True,
 'full_typed_cap_extra_logical_and_max_observed_allocation_reserved':True,'one_raw_two_metadata_slots_shared':True,
 'original_category_file_logical_allocated_caps_unchanged':True,'setup_proof_collection_settled_strict':True,
 'sampling_not_atomic_kernel_quota_or_authentication':True}
PROTOCOL_RULE={'kind':'pgd-priority-evidence-stream-v1','magic':'VJEPA-PGD-PRIORITY-EVIDENCE-1\n',
 'record_magic':'VJEPA-PGD-PRIORITY-FILE-1\n','end_magic':'VJEPA-PGD-PRIORITY-END-1\n',
 'critical_order':['session.json','capture/results/final-receipt.json','capture/results/collection-status.json','control.json','authorization.json','installation.json','proposal.json'],
 'reference_policy':'earlier first DATA with identical SHA256 and byte count; no chains',
 'sender_rehashes_each_reference_source':True,'receiver_independent_private_single_link_copies':True,
 'header_cap':2**20,'record_cap':4096,'trailer_cap':16384,'wire_expansion_disk_cap':3*2**30,
 'journal_reserve':8*2**20,'progress_cap':4*2**20,'journal_inside_existing_metadata_cap':64*2**20,
 'preparation_seconds':120,'collection_seconds':600,'connection_seconds':3000,
 'validated_header_and_incomplete_hash_chained_prefix_journal':True,'journal_never_claims_complete_transport':True,
 'terminal_EOF_tree_journal_identity_clock_and_disk_checks_required':True,'hard_blocking_IO_alarms_still_outer_owned':True}
EXPORTS=('boot','scratch','Window','WireReader','WireWriter','FileReader','Collector','validate_header','verify_local','_row','_directory','RECEIPT','STATUS','work_receipt','publish','excluded')


def rewrite(source,mapping):
 result,trace=literals.constants(source,mapping)
 if {ast.literal_eval(e['old_literal']) for e in trace}!=set(mapping):raise ValueError('Exact dependent literal pins only.')
 return result,trace

def load_frozen(repository):
 root=Path(repository);files={n:literals.read(root/n,wanted=w) for n,w in FROZEN.items()}
 c=literals.parsed(files[CONTROL]);bc=literals.parsed(c['supervised_control_text']);p=literals.parsed(bc['policy_text'])
 if digest(c['supervised_control_text'].encode())!=c['supervised_control_sha256'] or digest(bc['policy_text'].encode())!=bc['policy_sha256']:raise ValueError('Frozen control joins changed.')
 descriptors={'overlay/'+n:d for n,d in c['runtime_files'].items()};descriptors.update({'supervisor/'+n:d for n,d in bc['supervisor_sources'].items()})
 payload=literals.parsed(files[PAYLOAD])
 if set(payload)!=set(descriptors) or len(payload)!=69:raise ValueError('Exact69-source namespace required.')
 sources={}
 for name,value in payload.items():
  data=base64.b64decode(value,validate=True)
  if descriptors[name]!={'bytes':len(data),'sha256':digest(data)}:raise ValueError('Frozen source changed: '+name)
  sources[name]=data
 return c,bc,p,sources,files


def bind_collection(legacy,prototype):
 if digest(legacy)!='50e1deede0e0d9bf9a0f5a7ad9f1879920f4d01b74015666adcc89106b2523a1' or digest(prototype)!=FROZEN[PROTOCOL]:raise ValueError('Exact reviewed collector/prototype required.')
 # Lexical closure preserves legacy global bindings without a seventh import,
 # runtime exec/string compilation, or fallback to files from a local checkout.
 prefix='"""Bound six-slot collector; frozen validator closure plus paired priority codec."""\n'
 preserved=set()
 for token in tokenize.tokenize(io.BytesIO(legacy).readline):
  if token.type==tokenize.STRING and token.start[0]!=token.end[0]:preserved.update(range(token.start[0]+1,token.end[0]+1))
 indented=''.join(line if index in preserved else '    '+line for index,line in enumerate(legacy.decode().splitlines(keepends=True),1))
 prefix+='def _frozen_legacy_scope():\n'+indented
 prefix+='    from types import SimpleNamespace as _Namespace\n    return _Namespace('+','.join(n+'='+n for n in EXPORTS)+')\nlegacy=_frozen_legacy_scope()\n\n'
 bridge=b'from models import predictor_pgd_publication_collection_v1 as legacy\n'
 if prototype.count(bridge)!=1:raise ValueError('Exact prototype legacy import bridge required.')
 body=prototype.replace(bridge,b'',1)
 before=b'  self.prepared=self.streamed=False;self.header=None'
 after=b'  self.window=self.original.window\n'+before
 if body.count(before)!=1:raise ValueError('Original collector clock adapter location required.')
 body=body.replace(before,after,1)
 aliases=b'\n# Exact existing outer/client metadata APIs; no new clocks or source files.\nWindow=legacy.Window\nWireReader=legacy.WireReader\nWireWriter=legacy.WireWriter\nFileReader=legacy.FileReader\nRECEIPT=legacy.RECEIPT\nSTATUS=legacy.STATUS\nwork_receipt=legacy.work_receipt\n'
 result=prefix.encode()+body+aliases
 tree=ast.parse(result);factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_frozen_legacy_scope')
 original=ast.parse(legacy)
 if ast.dump(ast.Module(body=factory.body[:len(original.body)],type_ignores=[]),include_attributes=False)!=ast.dump(original,include_attributes=False):raise ValueError('Legacy validator closure AST must stay exact.')
 restored=body.replace(after,before,1).replace(b'from pathlib import Path,PurePosixPath\n',b'from pathlib import Path,PurePosixPath\n'+bridge,1)
 if restored!=prototype:raise ValueError('Priority prototype inverse required.')
 compile(result,'<UNEXECUTED-BOUND-PAIRED-COLLECTOR>','exec')
 return result,{'kind':'collector-closure-binding-v1','legacy_sha256':digest(legacy),'prototype_sha256':digest(prototype),'bound_sha256':digest(result),
  'bound_bytes':len(result),'legacy_body_nodes':len(original.body),'legacy_string_lines_kept':sorted(preserved),'legacy_AST_exact':True,'prototype_inverse_verified':True,'exports':list(EXPORTS),
  'legacy_import_resolved_lexically_no_extra_project_source':True,'original_clock_exposed_without_reset':True}


def assemble(repository):
 c,bc,oldp,original,files=load_frozen(repository);sources=dict(original)
 template,raw_meta=scanner.build(original[SCRATCH])
 if template!=files[TEMPLATE]:raise ValueError('Reviewed raw scanner template required.')
 sources[COLLECTION],collection_meta=bind_collection(original[COLLECTION],files[PROTOCOL])
 contract=literals.parsed(oldp['scratch_contract_text']);old_contract=copy.deepcopy(contract)
 contract['immutable_raw_publication_observation']=copy.deepcopy(RAW_RULE)
 contract['priority_collection_protocol']=dict(PROTOCOL_RULE,source_sha256=digest(sources[COLLECTION]),prototype_sha256=FROZEN[PROTOCOL])
 cb=encoded(contract);cs=digest(cb);sources[SCRATCH],scan_edit=rewrite(template,{oldp['scratch_contract_sha256']:cs})
 p=copy.deepcopy(oldp);p['scratch_contract_text']=cb.decode();p['scratch_contract_sha256']=cs
 p['fixed_support']['predictor_pgd_split_scratch_v1.py']=digest(sources[SCRATCH]);p['fixed_support']['predictor_pgd_split_collection_v1.py']=digest(sources[COLLECTION])
 p['publication_scan_repair']={'kind':'pgd-joint-evidence-binding-v1','historical_JSON_only_repair':copy.deepcopy(oldp['publication_scan_repair']),
  'prior_policy_sha256':bc['policy_sha256'],'current_contract_sha256':cs,'bound_scanner_sha256':digest(sources[SCRATCH]),'bound_collection_sha256':digest(sources[COLLECTION]),
  'raw_scanner_template_sha256':FROZEN[TEMPLATE],'priority_prototype_sha256':FROZEN[PROTOCOL],
  'journal_and_collector_bytes_unchanged':False,'numerical_writer_monitor_coordinator_proof_bytes_unchanged':True,'only_source_binding_no_authorization':True}
 pb=encoded(p);ps=digest(pb);sources[BOOT],boot_edit=rewrite(original[BOOT],{bc['policy_sha256']:ps})
 changed={n for n in sources if sources[n]!=original[n]}
 if changed!={SCRATCH,BOOT,COLLECTION}:raise ValueError('Only scanner/collector/bootstrap may change.')
 for key in oldp:
  if key not in ('scratch_contract_text','scratch_contract_sha256','fixed_support','publication_scan_repair') and p[key]!=oldp[key]:raise ValueError('Original scope/science/finance changed: '+key)
 comp=copy.deepcopy(contract);comp.pop('immutable_raw_publication_observation');comp.pop('priority_collection_protocol')
 if comp!=old_contract:raise ValueError('Original limits/layout/reference contract changed.')
 fixed=dict(p['fixed_support']);fixed.pop('predictor_pgd_split_collection_v1.py');fixed['predictor_pgd_split_scratch_v1.py']=oldp['fixed_support']['predictor_pgd_split_scratch_v1.py']
 if fixed!=oldp['fixed_support']:raise ValueError('Original monitor/proof pins changed.')
 control=encoded({'kind':bc['kind'],'policy_text':pb.decode(),'policy_sha256':ps,
  'supervisor_sources':{n.removeprefix('supervisor/'):{'bytes':len(v),'sha256':digest(v)} for n,v in sources.items() if n.startswith('supervisor/')}})
 if len(control)>256*1024 or sum(map(len,sources.values()))>2*2**20:raise ValueError('Original control/source caps exceeded.')
 readers={READERS['bootstrap']:sources[BOOT],READERS['scratch']:sources[SCRATCH]}
 local=sources[COLLECTION]
 for old,new in ((b'predictor_pgd_split_bootstrap_v1 as boot',b'predictor_pgd_joint_bootstrap_v1 as boot'),(b'predictor_pgd_split_scratch_v1 as scratch',b'predictor_pgd_joint_scratch_v1 as scratch')):
  if local.count(old)!=1:raise ValueError('Exactly two reader import bridges required.')
  local=local.replace(old,new,1)
 readers[READERS['collection']]=local
 for name,data in {**sources,**readers}.items():compile(data,'<UNEXECUTED-JOINT-BINDING-'+name+'>','exec')
 metadata={'kind':'pgd-joint-evidence-binding-v1-source-candidate','prior_outer_control_sha256':FROZEN[CONTROL],'prior_supervised_control_sha256':c['supervised_control_sha256'],
  'supervised_control_sha256':digest(control),'policy_sha256':ps,'scratch_contract_sha256':cs,'bound_scanner_sha256':digest(sources[SCRATCH]),'bound_collection_sha256':digest(sources[COLLECTION]),'bootstrap_sha256':digest(sources[BOOT]),
  'source_literal_edits':{SCRATCH:scan_edit,BOOT:boot_edit},'scanner_derivation':raw_meta,'collector_derivation':collection_meta,
  'changed_uploaded_sources':sorted(changed),'unchanged_uploaded_sources':66,'source_files':69,'source_bytes':sum(map(len,sources.values())),'runtime_files':63,
  'runtime_manifest_unchanged':True,'full_scope_scientific_financial_resource_bytes_unchanged':True,'reader_pins':{n:digest(v) for n,v in readers.items()},
  'outer_client_handoff_prepared':False,'authorized':False,'scientific_qualified':False,'training_admitted':False}
 return control,sources,readers,literals.parsed(p['runtime_repair']['runtime_manifest_text']),metadata


def verify_readers(repository,readers):
 if set(readers)!=set(READERS.values()):raise ValueError('Exact three joint readers required.')
 for name,data in readers.items():literals.read(Path(repository)/name,wanted=digest(data))
 return {'kind':'pgd-joint-reader-binding-v1','reader_files':3,'source_identity_verified':True,'scientific_qualified':False,'training_admitted':False}

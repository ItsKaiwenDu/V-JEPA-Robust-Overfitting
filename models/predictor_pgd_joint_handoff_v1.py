"""Fresh local complete handoff binding; no arming or target/packet work."""
import ast,base64,os,stat
from pathlib import Path
from models import predictor_pgd_joint_integration_v1 as integration
from models import predictor_pgd_close_handoff_v1 as helpers
from models import predictor_pgd_host10609_v1 as literals
OLD={'outer': 'models/predictor_pgd_publication_outer_v1.py', 'client': 'models/predictor_pgd_publication_client_v1.py', 'cli': 'scripts/run_predictor_pgd_publication_v1.py'}
FROZEN={'models/predictor_pgd_publication_outer_v1.py': '9af62ffbb15f244ffea0eb2adafa6fa5d39762e93c54eb65217246be36b0fa6a', 'models/predictor_pgd_publication_client_v1.py': '9e6c23e21c237033c7b6942cc8b1a6f0b45ec873090ca0ef83957a6870a277be', 'scripts/run_predictor_pgd_publication_v1.py': '1fd1fe981296ed3f6d165693fb73d2454283cfe547335854387d5f7c64626e83'}
NEW={'outer':'models/predictor_pgd_joint_outer_v1.py','client':'models/predictor_pgd_joint_client_v1.py','cli':'scripts/run_predictor_pgd_joint_v1.py','builder':'models/predictor_pgd_joint_handoff_v1.py'}
CONTROL_KIND='pgd-joint-evidence-outer-control-v1';AUTH_KIND='pgd-joint-evidence-university-exception-v1'
encoded,digest=integration.encoded,integration.digest


def function_edit(raw,name,change):
 tree=ast.parse(raw);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);lines=raw.splitlines(keepends=True)
 begin=sum(map(len,lines[:fn.lineno-1]));end=sum(map(len,lines[:fn.end_lineno]));old=raw[begin:end];new=change(old)
 if old==new:raise ValueError('Explicit function edit required: '+name)
 return raw[:begin]+new+raw[end:],{'begin':begin,'new_end':begin+len(new),'old_method':old.decode(),'new_method':new.decode(),'function':name}


def build_local(repository):
 root=Path(repository);old={name:literals.read(root/path,wanted=FROZEN[path]) for name,path in OLD.items()}
 control,sources,readers,manifest,meta=integration.assemble(root);integration.verify_readers(root,readers)
 changes={'9f1d10b01fc48e09711084be515f1193fa5140244e1d07eb1848b0fde5164920':meta['policy_sha256'],
  '763a9d1d0e609583e1356392991ad93a52b50309099baeb417751183d4847239':meta['bootstrap_sha256'],
  '50e1deede0e0d9bf9a0f5a7ad9f1879920f4d01b74015666adcc89106b2523a1':meta['bound_collection_sha256'],
  'pgd-publication-outer-control-v1':CONTROL_KIND,'pgd-publication-outer-university-exception-v1':AUTH_KIND,OLD['outer']:NEW['outer']}
 outer,trace=literals.constants(old['outer'],changes)
 if {ast.literal_eval(e['old_literal']) for e in trace}!=set(changes):raise ValueError('Exact new outer literal bindings required.')
 previous=ast.literal_eval(helpers.assignment(old['outer'],'LOCAL_SOURCES').value)
 additions=(NEW['outer'],NEW['client'],NEW['cli'],NEW['builder'],'models/predictor_pgd_joint_integration_v1.py',
  'models/predictor_pgd_raw_publication_scan_v1.py',integration.TEMPLATE,integration.PROTOCOL,*integration.READERS.values(),integration.CONTROL,integration.PAYLOAD)
 local_sources=additions+previous
 if len(set(local_sources))!=len(local_sources) or len(local_sources)!=69:raise ValueError('Distinct complete69local source/input pins required.')
 outer,value_trace=helpers.replace_value(outer,'LOCAL_SOURCES',local_sources)
 outer=helpers.replace_once(outer,b'from models import predictor_pgd_publication_handoff_v1 as binding',b'from models import predictor_pgd_joint_handoff_v1 as binding')
 client=old['client']
 for before,after in ((b'predictor_pgd_publication_outer_v1 as outer',b'predictor_pgd_joint_outer_v1 as outer'),(b'predictor_pgd_publication_bootstrap_v1 as boot',b'predictor_pgd_joint_bootstrap_v1 as boot'),(b'predictor_pgd_publication_collection_v1 as collection',b'predictor_pgd_joint_collection_v1 as collection')):client=helpers.replace_once(client,before,after)
 def bound_header(raw):
  raw=helpers.replace_once(raw,b'    if (h[',b"    base=h['base']\n    if (base[")
  return raw.replace(b"h['root_binding']",b"base['root_binding']").replace(b"h['connection_started']",b"base['connection_started']").replace(b"h['collection_started']",b"base['collection_started']").replace(b"h['collection_deadline']",b"base['collection_deadline']")
 client,header_edit=function_edit(client,'bound_collection_source',bound_header)
 before=b"        alarms.arm(window.deadline,hard_deadline=started+3000,phase='client-collection')\n"
 after=before+b"        outer.private_write(session,'COLLECTION-READY.json',outer.encoded(ready),window.check)\n"
 client=helpers.replace_once(client,before,after)
 cli=helpers.replace_once(old['cli'],b'from models.predictor_pgd_publication_client_v1 import run',b'from models.predictor_pgd_joint_client_v1 import run')
 generated={NEW['outer']:outer,NEW['client']:client,NEW['cli']:cli}
 for name,data in generated.items():compile(data,'<UNEXECUTED-JOINT-HANDOFF-'+name+'>','exec')
 metadata={'kind':'pgd-joint-local-operator-derivation-v1','outer_literal_edits':trace,'local_source_assignment':value_trace,
  'client_bound_header_edit':header_edit,'client_ready_receipt_edit':{'old':before.decode(),'new':after.decode()},
  'local_source_files':len(local_sources),'supervised_control_sha256':digest(control),'policy_sha256':meta['policy_sha256'],
  'control_kind':CONTROL_KIND,'authorization_kind':AUTH_KIND,'outer_remote_session_clocks_claims_setup_work_collection_AST_unchanged':True,
  'client_changes_only_imports_nested_header_and_ready_record':True,'authorized':False,'scientific_qualified':False,'training_admitted':False}
 return generated,metadata


def prepare_package(repository,destination):
 from models import predictor_pgd_joint_outer_v1 as outer
 from models import predictor_pgd_split_source_bundle_v1 as source
 root,dest=Path(repository),Path(destination);parent=dest.parent.lstat()
 if not root.is_absolute() or root.resolve(strict=True)!=root or not dest.is_absolute() or dest.parent.resolve(strict=True)!=dest.parent or os.path.lexists(dest) or not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode)!=0o700 or parent.st_uid!=os.getuid():raise ValueError('Fresh canonical private handoff required.')
 generated,derivation=build_local(root)
 for name,data in generated.items():literals.read(root/name,wanted=digest(data))
 control,sources,readers,manifest,meta=integration.assemble(root);integration.verify_readers(root,readers)
 b=literals.parsed(control);p=literals.parsed(b['policy_text'])
 outer.verify_directory_policy_sources(root,source);outer.verify_refusal_policy_sources(root,source)
 pins={name:digest(literals.read(root/name)) for name in outer.LOCAL_SOURCES}
 c={'kind':CONTROL_KIND,'supervised_control_text':control.decode(),'supervised_control_sha256':digest(control),'outer_source_sha256':pins[NEW['outer']],
  'local_source_sha256':pins,'runtime_files':manifest['runtime_files'],'repository':str(root),'local_handoff':str(dest),'packet_path':outer.PACKET_PATH,
  'known_hosts_text':outer.KNOWN_HOSTS,'known_hosts_sha256':digest(outer.KNOWN_HOSTS.encode()),'expected_ssh_fingerprint':outer.FINGERPRINT,
  'financial_state_unchanged':True,'authorized_at_preparation':False,'target_helper_reviewed':False}
 raw=encoded(c);outer.validate_control(raw,pins[NEW['outer']]);payload=encoded({name:base64.b64encode(data).decode() for name,data in sources.items()})
 outer.decode_sources(literals.parsed(payload),c,b)
 if len(payload)+len(raw)+65536>2*2**20:raise ValueError('Original source/control frame cap exceeded.')
 package={'kind':'pgd-joint-evidence-unarmed-package-v1','outer_control_sha256':digest(raw),'supervised_control_sha256':digest(control),
  'source_files':69,'source_bytes':sum(map(len,sources.values())),'source_envelope_bytes':len(payload),'local_source_files':len(pins),
  'derivation':derivation,'limits':p['scope'],'authorization_created':False,'target_work_or_SSH_executed':False,'scientific_qualified':False,'training_admitted':False}
 dest.mkdir(mode=0o700)
 for name,data in (('CONTROL-CANDIDATE.json',raw),('SOURCE-PAYLOAD.json',payload),('known-hosts',outer.KNOWN_HOSTS.encode()),('PACKAGE.json',encoded(package))):outer.private_write(dest,name,data,lambda:None)
 return package

"""Source-only fresh attended scanner handoff packaging; no grants or launches."""
import ast
import base64
import os
from pathlib import Path
import stat
from models import predictor_pgd_publication_integration_v1 as integration
from models import predictor_pgd_close_handoff_v1 as helpers
from models import predictor_pgd_host10609_v1 as literals

OLD={'outer':'models/predictor_pgd_host10614_outer_v1.py','client':'models/predictor_pgd_host10614_client_v1.py','cli':'scripts/run_predictor_pgd_host10614_v1.py'}
FROZEN={OLD['outer']:'68d54b5018465c111b1abe53ae78be5b019fe5fd5efa4df40ab8a74a6bee96a1',
 OLD['client']:'10232a88dd5f9d219d6ce1e168f363a2ca8e5a0f8d99b55be7483c98dac4fa96',
 OLD['cli']:'3b820c34abc46000759a688758fe0ac3addbd3691f8b078e6e619f63d860e1b2'}
NEW={'outer':'models/predictor_pgd_publication_outer_v1.py','client':'models/predictor_pgd_publication_client_v1.py',
 'cli':'scripts/run_predictor_pgd_publication_v1.py','builder':'models/predictor_pgd_publication_handoff_v1.py'}
CONTROL_KIND='pgd-publication-outer-control-v1';AUTH_KIND='pgd-publication-outer-university-exception-v1'
POLICY_SHA='9f1d10b01fc48e09711084be515f1193fa5140244e1d07eb1848b0fde5164920'
BOOT_SHA='763a9d1d0e609583e1356392991ad93a52b50309099baeb417751183d4847239'
encoded,digest=integration.encoded,integration.digest


def build_local(repository):
    root=Path(repository);old={n:literals.read(root/p,wanted=FROZEN[p]) for n,p in OLD.items()}
    control,sources,readers,manifest,meta=integration.assemble(root)
    integration.verify_readers(root,readers)
    if meta['policy_sha256']!=POLICY_SHA or digest(sources[integration.BOOT])!=BOOT_SHA:raise ValueError('Exact reviewed publication integration required.')
    changes={'d4a68b570d88fb5331707d5d1ca0ebd5872972d1b8dfb7d1f1802160fae9e1a7':POLICY_SHA,
      '9d865b43e4f841213963005d85c4cac4e261f62d726353fb78cd52ad48589b2c':BOOT_SHA,
      'pgd-host10614-outer-control-v1':CONTROL_KIND,'pgd-host10614-outer-university-exception-v1':AUTH_KIND,
      OLD['outer']:NEW['outer']}
    outer,trace=literals.constants(old['outer'],changes)
    if {ast.literal_eval(e['old_literal']) for e in trace}!=set(changes):raise ValueError('Exact new handoff literal bindings required.')
    previous=ast.literal_eval(helpers.assignment(old['outer'],'LOCAL_SOURCES').value)
    additions=(NEW['outer'],NEW['client'],NEW['cli'],NEW['builder'],'models/predictor_pgd_publication_integration_v1.py',
      'models/predictor_pgd_publication_scan_v1.py',integration.TEMPLATE,*integration.READERS.values(),integration.CONTROL,integration.PAYLOAD)
    local_sources=additions+previous
    if len(set(local_sources))!=len(local_sources) or len(local_sources)!=56:raise ValueError('Distinct full56local source/input pins required.')
    outer,value_trace=helpers.replace_value(outer,'LOCAL_SOURCES',local_sources)
    outer=helpers.replace_once(outer,b'from models import predictor_pgd_host10614_v1 as binding',b'from models import predictor_pgd_publication_handoff_v1 as binding')
    client=old['client']
    bridges=((b'predictor_pgd_host10614_outer_v1 as outer',b'predictor_pgd_publication_outer_v1 as outer'),
      (b'predictor_pgd_host10614_bootstrap_v1 as boot',b'predictor_pgd_publication_bootstrap_v1 as boot'),
      (b'predictor_pgd_host10614_collection_v1 as collection',b'predictor_pgd_publication_collection_v1 as collection'))
    for before,after in bridges:client=helpers.replace_once(client,before,after)
    cli=helpers.replace_once(old['cli'],b'from models.predictor_pgd_host10614_client_v1 import run',b'from models.predictor_pgd_publication_client_v1 import run')
    result={NEW['outer']:outer,NEW['client']:client,NEW['cli']:cli}
    for name,data in result.items():compile(data,'<UNEXECUTED-PUBLICATION-HANDOFF-'+name+'>','exec')
    metadata={'kind':'pgd-publication-handoff-local-derivation-v1','frozen_local_source_pins':FROZEN,'outer_literal_edits':trace,
      'local_source_assignment':value_trace,'local_source_files':56,'control_kind':CONTROL_KIND,'authorization_kind':AUTH_KIND,
      'integration_supervised_control_sha256':digest(control),'policy_sha256':POLICY_SHA,'bootstrap_sha256':BOOT_SHA,
      'source_files':69,'source_bytes':sum(map(len,sources.values())),'new_authorization_or_attempt_created':False,
      'target_work_or_SSH_started':False,'scientific_qualified':False,'training_admitted':False}
    return result,metadata


def prepare_package(repository,destination):
    from models import predictor_pgd_publication_outer_v1 as outer
    from models import predictor_pgd_split_source_bundle_v1 as source
    root,dest=Path(repository),Path(destination);parent=dest.parent.lstat()
    if (not root.is_absolute() or root.resolve(strict=True)!=root or not dest.is_absolute() or dest.parent.resolve(strict=True)!=dest.parent
            or os.path.lexists(dest) or not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode)!=0o700 or parent.st_uid!=os.getuid()):
        raise ValueError('Fresh canonical private owned local handoff required.')
    generated,derivation=build_local(root)
    for name,data in generated.items():literals.read(root/name,wanted=digest(data))
    control,sources,readers,manifest,integration_meta=integration.assemble(root);integration.verify_readers(root,readers)
    b=literals.parsed(control);p=literals.parsed(b['policy_text'])
    outer.verify_directory_policy_sources(root,source);outer.verify_refusal_policy_sources(root,source)
    pins={name:digest(literals.read(root/name)) for name in outer.LOCAL_SOURCES}
    c={'kind':CONTROL_KIND,'supervised_control_text':control.decode(),'supervised_control_sha256':digest(control),
      'outer_source_sha256':pins[NEW['outer']],'local_source_sha256':pins,'runtime_files':manifest['runtime_files'],
      'repository':str(root),'local_handoff':str(dest),'packet_path':outer.PACKET_PATH,'known_hosts_text':outer.KNOWN_HOSTS,
      'known_hosts_sha256':digest(outer.KNOWN_HOSTS.encode()),'expected_ssh_fingerprint':outer.FINGERPRINT,
      'financial_state_unchanged':True,'authorized_at_preparation':False,'target_helper_reviewed':False}
    raw=encoded(c);outer.validate_control(raw,pins[NEW['outer']])
    payload=encoded({n:base64.b64encode(v).decode() for n,v in sources.items()});outer.decode_sources(literals.parsed(payload),c,b)
    if len(payload)+len(raw)+65536>2*2**20:raise ValueError('Original source/control upload frame cap exceeded.')
    package={'kind':'pgd-publication-handoff-unarmed-package-v1','outer_control_sha256':digest(raw),
      'supervised_control_sha256':digest(control),'source_files':69,'source_bytes':sum(map(len,sources.values())),
      'source_envelope_bytes':len(payload),'local_source_files':56,'derivation':derivation,'limits':p['scope'],
      'authorization_created':False,'target_work_or_SSH_executed':False,'scientific_qualified':False,'training_admitted':False}
    dest.mkdir(mode=0o700)
    for name,data in (('CONTROL-CANDIDATE.json',raw),('SOURCE-PAYLOAD.json',payload),('known-hosts',outer.KNOWN_HOSTS.encode()),('PACKAGE.json',encoded(package))):
        outer.private_write(dest,name,data,lambda:None)
    return package

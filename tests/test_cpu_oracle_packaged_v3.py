"""Cold extracted-bundle checks; live resources and full-size proofs stay off."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import pytest

from models import cpu_oracle_capture_v3 as remote
from models import cpu_oracle_capture_client_v3 as client
from models.cpu_oracle_packet import build
from models.study_artifacts import read_record
from test_cpu_oracle_capture_v3 import control,authorization

ROOT=Path(__file__).resolve().parents[1]
PACKET=ROOT/'runs/pretrained/diagnostic/cpu-oracle-university-20261030/packet/vjepa-cpu-oracle.tar.gz'


def overlay(folder):
    c=control();remote.write_overlay(folder,client.sources(c),c)
    return c


def cli(folder,*,local=False,packet=None,sha=None):
    command=[sys.executable,'-I','-B',str(folder/'overlay/scripts/capture_cpu_reference_v3.py'),
        '--packet',str(packet or folder/'code'),'--scratch',str(folder),'--upstream',str(ROOT/'external/vjepa2'),
        '--launcher-started',str(time.time())]
    if local:command.append('--local-tiny-rehearsal')
    if sha:command+=['--control-sha256',sha]
    return command


def environment():
    return {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
        'PYTHONNOUSERSITE':'1','PYTHONPATH':''}


@pytest.fixture(scope='module')
def cold_tiny(tmp_path_factory,record_testsuite_property):
    root=tmp_path_factory.mktemp('v3-cold-extracted')
    manifest,spec=build(ROOT,root/'packet-build',tiny=True)
    folder=root/'run';folder.mkdir(mode=0o700)
    archive=root/'packet-build/vjepa-cpu-oracle.tar.gz'
    assert remote.old.digest(archive)==spec['sha256']
    packet=folder/'code';packet.mkdir(mode=0o700)
    with tarfile.open(archive) as source:
        members=source.getmembers()
        assert len(members)==len(spec['files']) and {m.name for m in members}==set(spec['files'])
        for member in members:
            name=remote.old.safe_name(member.name)
            assert member.isfile() and member.size==spec['files'][name]['bytes']
            with source.extractfile(member) as incoming:data=incoming.read()
            assert hashlib.sha256(data).hexdigest()==spec['files'][name]['sha256']
            target=packet/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    overlay(folder);neutral=root/'neutral';neutral.mkdir()
    result=subprocess.run(cli(folder,local=True),cwd=neutral,env=environment(),
        capture_output=True,text=True,timeout=150)
    (root/'cold.stdout').write_text(result.stdout);(root/'cold.stderr').write_text(result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    record_testsuite_property('cold_bundle_evidence_directory',str(root))
    return root,folder,manifest,result


def test_actual_extracted_cli_completes_four_fresh_original_tiny_workers(cold_tiny):
    _,folder,manifest,result=cold_tiny
    state=read_record(folder/'results/session-v3.json')
    assert state['complete'] and state['saved_value_checks_passed'] and state['local_tiny_rehearsal']
    assert [[r['threads'],r['replay']] for r in state['tasks']]==remote.old.TASKS
    assert not state['research_admitted'] and not state['original_refusals_waived']
    assert not (folder/'AUTHORIZATION.json').exists() and not (folder/'space-proof-v2.bin.partial').exists()
    for i in range(4):
        worker=read_record(folder/f'results/workers/{i}/worker.json')
        assert worker['capture_complete'] and worker['numerical_checks_passed'] and worker['tiny_fixture']
        assert worker['local_CPU_rehearsal'] and worker['context_before']['CUDA_initialized'] is False
        assert worker['context_after']['CUDA_initialized'] is False
    assert result.stdout.count('V3 fresh original CPU worker:')==4


def test_cold_cli_export_is_complete_and_all_member_hashes_verify(cold_tiny):
    _,folder,_,_=cold_tiny
    archive=folder/'results/evidence.tar.gz'
    manifest=json.loads((folder/'results/evidence.tar.gz.manifest.json').read_text())
    assert remote.old.digest(archive)==manifest['sha256'] and archive.stat().st_size==manifest['bytes']
    with tarfile.open(archive) as source:
        members=source.getmembers();assert len(members)==len(manifest['files'])
        for member in members:
            assert member.isfile() and member.size==manifest['files'][member.name]['bytes']
            import hashlib
            with source.extractfile(member) as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['files'][member.name]['sha256']
    assert 'results/summary-v3.json' in manifest['files'] and 'results/session-v3.json' in manifest['files']


@pytest.fixture
def full_bundle(tmp_path):
    c=overlay(tmp_path)
    remote.old.extract_packet(PACKET,c['archive_manifest'],tmp_path/'code')
    remote.old.write_json(tmp_path/'operator-control.json',c)
    sha=remote.old.digest(tmp_path/'operator-control.json')
    remote.old.write_json(tmp_path/'resource-authorization.json',authorization(sha))
    neutral=tmp_path/'neutral';neutral.mkdir()
    return tmp_path,c,sha,neutral


def import_harness():
    path=ROOT/'tests/test_cpu_oracle_overlay_import.py'
    spec=importlib.util.spec_from_file_location('frozen_overlay_regression',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.HARNESS.replace('cpu_oracle_capture_runtime_v2','cpu_oracle_capture_runtime_v3')


def test_full_source_bundle_imports_all_namespaces_before_blocked_dispatch(full_bundle):
    folder,_,sha,neutral=full_bundle
    harness=folder/'import-only.py';harness.write_text(import_harness())
    command=[sys.executable,'-I','-B',str(harness),str(folder/'overlay/scripts/capture_cpu_reference_v3.py'),
        str(folder/'code'),str(folder),sha]
    result=subprocess.run(command,cwd=neutral,env=environment(),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout.strip().splitlines()[-1])
    assert value['outcome']=='runtime_imported_dispatch_blocked' and value['worker_dispatch_blocked']
    assert value['oracle_calls']==value['subprocess_attempts']==0 and not value['CUDA_initialized']
    assert not (folder/'results').exists() and not (folder/'allocation-result-v2.json').exists()


@pytest.mark.parametrize('defect',['wrong_packet','symlink','overlay_tamper','packet_tamper','missing_package','old_grant','local_full'])
def test_cold_cli_refuses_changed_roots_sources_and_unapproved_modes(full_bundle,defect):
    folder,_,sha,neutral=full_bundle;packet=folder/'code';local=False
    if defect=='wrong_packet':packet=neutral
    elif defect=='symlink':
        alias=folder/'code-alias';alias.symlink_to(packet,target_is_directory=True);packet=alias
    elif defect in ('overlay_tamper','packet_tamper'):
        p=folder/('overlay/models/cpu_oracle_capture_runtime_v3.py' if defect=='overlay_tamper' else 'code/attacks/__init__.py')
        p.write_bytes(p.read_bytes()+b'\n# altered fixture\n')
    elif defect=='missing_package':(packet/'attacks/__init__.py').unlink()
    elif defect=='old_grant':
        old=ROOT/'runs/pretrained/diagnostic/cpu-oracle-capture-v2-20261104/linux10603/AUTHORIZATION.json'
        (folder/'resource-authorization.json').write_bytes(old.read_bytes())
    elif defect=='local_full':local=True;sha=None
    result=subprocess.run(cli(folder,local=local,packet=packet,sha=sha),cwd=neutral,env=environment(),
        capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and not (folder/'results').exists()
    if defect in ('wrong_packet','symlink'):assert 'original sibling package directories' in result.stderr
    elif defect=='old_grant':assert 'separate explicit v3 target resource exception' in result.stderr
    elif defect=='local_full':assert 'Explicit tiny rehearsal is local only' in result.stderr
    else:assert 'Packaged source or fixture changed' in result.stderr


def test_local_rehearsal_refuses_a_university_hostname(cold_tiny,tmp_path):
    _,original,_,_=cold_tiny
    folder=tmp_path/'run';folder.mkdir()
    shutil.copytree(original/'code',folder/'code');overlay(folder)
    command=cli(folder,local=True)
    script=("import platform,runpy,sys;platform.node=lambda:'linux10603';sys.argv="+repr(command[3:])+";runpy.run_path(sys.argv[0],run_name='__main__')")
    result=subprocess.run([sys.executable,'-I','-B','-c',script],cwd=tmp_path,env=environment(),
        capture_output=True,text=True,timeout=30)
    assert result.returncode==2 and 'Explicit tiny rehearsal is local only' in result.stderr
    assert not (folder/'results').exists()

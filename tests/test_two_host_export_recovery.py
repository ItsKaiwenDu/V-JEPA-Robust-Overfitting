"""CPU-only alias regression and read-only incident transport fixtures.

No fixture is a CUDA result; byte recovery cannot confer numerical acceptance.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('incident_copy',ROOT/'scripts/recover_two_host_exports.py')
copy_ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(copy_ops)
spec=importlib.util.spec_from_file_location('coordinator',ROOT/'models/two_host_panel.py')
ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
FROZEN=ROOT/'runs/pretrained/diagnostic/two-host-panel-20261026/extracted/two-host-execution-20261026'


def test_export_event_uses_one_canonical_root_for_a_home_alias(tmp_path):
    """Reproduce the actual /home versus /DCNFS failure using a real symlink."""
    root=tmp_path/'canonical-home';root.mkdir();alias=tmp_path/'home-alias';alias.symlink_to(root,target_is_directory=True)
    r=json.loads((FROZEN/'configs/matched_panel_two_host_v1.json').read_text())
    fixture=json.loads((ROOT/'tests/fixtures/two_host_bootstrap_metadata.json').read_text())
    reg=ops.Registry(alias,r,'1'*64,ops.import_metadata(fixture['journal']));cid='clip-03-bank-1'
    directory=ops.bound(alias,'cases/'+cid+'/attempt-01');directory.mkdir(parents=True)
    # Incomplete metadata-only export fixture. It deliberately has no numerical receipt.
    for name,body in [('case-claim.json',{'case_id':cid}),('case-request.json',{'CPU_fixture_only':True}),
                      ('case-envelope.json',{'complete':False,'CPU_fixture_only':True})]:ops.write_record(directory/name,body)
    reg.fail(cid,'linux10609','CPU fixture, no GPU work',0)
    (directory/'numeric-evidence.tar.gz').write_bytes(b'CPU export-path fixture; not numerical evidence')
    (directory/'numeric-evidence.tar.gz.manifest.json').write_text('{}\n')
    event=ops.export_revision_case(FROZEN,reg,directory)
    assert reg.root==root.resolve()
    assert event['archive_path']=='cases/'+cid+'/attempt-01/revision-evidence.tar.gz'
    assert ops.digest(reg.root/event['archive_path'])==event['archive_sha256']
    assert reg.status()['accepted_case_ids']==ops.CASE_ORDER[:7]


def make_fixture(home):
    required,optional=copy_ops.inventory(home)
    for name,path in required.items():
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('CPU transport fixture only.\n')
    for name,source in [('packet/IMPLEMENTATION.json',FROZEN/'IMPLEMENTATION.json'),
                        ('packet/revision.json',FROZEN/'configs/matched_panel_two_host_v1.json')]:
        required[name].write_bytes(source.read_bytes())
    for cid in copy_ops.CASES.values():
        archive=required['cases/'+cid+'/revision-evidence.tar.gz']
        with tarfile.open(archive,'w:gz'):pass
        Path(str(archive)+'.manifest.json').write_text(json.dumps({'bytes':archive.stat().st_size,'sha256':copy_ops.digest(archive)}))
    return required,optional


def transport_fixture(tmp_path):
    home=tmp_path/'home';files,optional=make_fixture(home);output=tmp_path/'download';output.mkdir()
    code="import importlib.util,sys;from pathlib import Path;s=importlib.util.spec_from_file_location('copy',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.stream(Path(sys.argv[2]),m.HOST)"
    with (output/'transport.tar.part').open('wb') as out,(output/'transport.stderr').open('wb') as err:
        p=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'scripts/recover_two_host_exports.py'),str(home)],stdout=out,stderr=err)
    assert p.returncode==0,(output/'transport.stderr').read_text()
    return home,files,optional,output


def test_both_existing_archives_copy_even_without_export_events_and_do_not_modify_remote_files(tmp_path):
    home,files,optional,output=transport_fixture(tmp_path)
    before={n:copy_ops.digest(p) for n,p in files.items()}
    result=copy_ops.receive(output)
    assert result['independent_numerical_acceptance'] is False and result['GPU_job_repeated'] is False
    assert set(result['absent_optional_files'])==set(optional)
    assert {n:copy_ops.digest(p) for n,p in files.items()}==before
    for cid in copy_ops.CASES.values():assert (output/'downloaded/cases'/cid/'revision-evidence.tar.gz').is_file()
    assert not any(p.exists() for p in optional.values())


@pytest.mark.parametrize('change',['wrong-host','missing-receipt','changed-sidecar','escaped-link'])
def test_collect_rejects_wrong_host_missing_or_escaped_and_mixed_evidence(tmp_path,change):
    files,_=make_fixture(tmp_path);host=copy_ops.HOST
    path=files['cases/clip-03-bank-1/receipt.json']
    if change=='wrong-host':host='linux10603'
    elif change=='missing-receipt':path.unlink()
    elif change=='changed-sidecar':files['cases/clip-03-bank-1/revision-evidence.tar.gz.manifest.json'].write_text('{"sha256":"other","bytes":0}')
    else:path.unlink();path.symlink_to(files['packet/IMPLEMENTATION.json'])
    with pytest.raises(ValueError):copy_ops.collect(tmp_path,host)


@pytest.mark.parametrize('change',['bytes','checksum','missing-completion','duplicate-completion'])
def test_corrupt_and_incomplete_copies_are_preserved_without_acceptance(tmp_path,change):
    _,_,_,output=transport_fixture(tmp_path);path=output/'transport.tar.part';stderr=output/'transport.stderr'
    if change=='bytes':path.write_bytes(b'corrupt')
    elif change=='checksum':
        text=stderr.read_text();record=json.loads(text.split('TWO_HOST_INCIDENT_BUNDLE ')[1]);record['sha256']='0'*64
        stderr.write_text('TWO_HOST_INCIDENT_BUNDLE '+json.dumps(record)+'\n')
    elif change=='missing-completion':stderr.write_text('connection closed\n')
    else:stderr.write_text(stderr.read_text()+stderr.read_text())
    before=path.read_bytes()
    with pytest.raises(ValueError):copy_ops.receive(output)
    assert path.read_bytes()==before and not (output/'downloaded').exists()


def test_download_never_overwrites_existing_local_evidence(tmp_path):
    _,_,_,output=transport_fixture(tmp_path);copy_ops.receive(output)
    (output/'transport.tar').rename(output/'transport.tar.part')
    with pytest.raises(FileExistsError):copy_ops.receive(output)


@pytest.mark.parametrize('name',['../escape','/absolute','a/../b','a\\b','./name','.'])
def test_archive_paths_are_regular_and_contained(name):
    with pytest.raises(ValueError):copy_ops.safe_name(name)

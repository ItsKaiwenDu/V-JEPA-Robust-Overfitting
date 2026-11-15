"""Saved-response and framing boundaries only; no scientific or remote work."""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import pytest
from models import private_collection_v1 as private

ROOT=Path(__file__).resolve().parents[1]
RAW=Path(os.environ['VJEPA_PRIVATE_COLLECTION_EVIDENCE'])
STAGE=ROOT/'runs/pretrained/diagnostic/predictor-missing-replay-handoff-v1-20261114'
SESSION=STAGE/'linux10603-ready/attended-nqg9xqaj'


def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def current_umask():
    value=os.umask(0o077);os.umask(value);return value


@contextmanager
def mask(value):
    previous=os.umask(value)
    try:yield
    finally:os.umask(previous)


def fixture_response(folder, *, fault=None):
    c={'kind':'private-collection-explicit-local-fixture-v1','execution_permitted':False}
    h=hashlib.sha256(private.frozen.old.json_bytes(c)).hexdigest()
    content=b'explicit local bytes only';name='reference/arrays/fixture.bin'
    member_name='../escape' if fault=='traversal' else name
    members={member_name:{'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'mode':0o600,'mtime_ns':1700000000123456789}}
    if fault=='bad-hash':members[member_name]['sha256']='00'*32
    if fault=='public-mode':members[member_name]['mode']=0o644
    if fault=='extent':members[member_name]['bytes']+=1
    first={'kind':c['kind'],'control_sha256':h,'phase':'scratch-created','scratch_directory':'/SIMULATED-LOCAL-NO-TARGET'}
    header={'kind':c['kind'],'control_sha256':h,'phase':'collection','model_restarted':False,'logical_bytes':sum(v['bytes'] for v in members.values()),'files':members}
    if fault=='wrong-control':first['control_sha256']='ff'*32
    if fault=='wrong-kind':header['kind']='different-format'
    if fault=='restarted':header['model_restarted']=True
    data=io.BytesIO()
    data.write(private.frozen.old.json_bytes(first));data.write(private.frozen.old.json_bytes(header))
    with tarfile.open(fileobj=data,mode='w|',format=tarfile.USTAR_FORMAT) as archive:
        m=tarfile.TarInfo(member_name);m.size=len(content);m.mode=0o600
        if fault=='symlink':m.type=tarfile.SYMTYPE;m.linkname='/OUTSIDE';m.size=0
        if fault=='hardlink':m.type=tarfile.LNKTYPE;m.linkname='/OUTSIDE';m.size=0
        archive.addfile(m,io.BytesIO(content) if m.isfile() else None)
        if fault=='duplicate':archive.addfile(m,io.BytesIO(content))
    blob=data.getvalue()
    if fault=='truncated':blob=blob[:-512]
    if fault=='extra':blob+=b'extra'
    p=folder/'fixture-response.bin';p.write_bytes(blob);p.chmod(0o600)
    return p,c,h,digest(p)


def test_original_decoder_code_object_and_globals_remain_unchanged(tmp_path):
    original=dict(private.frozen.unpack.__globals__)
    response,c,h,checksum=fixture_response(tmp_path)
    with mask(0):
        private.unpack(response,tmp_path/'saved',c,h,response_sha256=checksum)
        assert current_umask()==0
    assert private._CODE is private.frozen.unpack.__code__
    assert private.frozen.unpack.__globals__==original
    assert all(stat.S_IMODE(p.stat().st_mode)==0o700 for p in (tmp_path/'saved').rglob('*') if p.is_dir())


@pytest.mark.parametrize('umask',[0,0o022,0o027,0o077])
def test_all_new_parents_are_private_independent_of_caller_umask(tmp_path,umask):
    response,c,h,checksum=fixture_response(tmp_path)
    before=response.stat()
    with mask(umask):
        private.unpack(response,tmp_path/'saved',c,h,response_sha256=checksum)
        assert current_umask()==umask
    assert (response.stat().st_ino,response.stat().st_mtime_ns)==(before.st_ino,before.st_mtime_ns)
    for p in [tmp_path/'saved',*(tmp_path/'saved').rglob('*')]:
        assert not p.is_symlink()
        assert stat.S_IMODE(p.stat().st_mode)==(0o700 if p.is_dir() else 0o600)
    assert (tmp_path/'saved/reference/arrays/fixture.bin').stat().st_mtime_ns==1700000000123456789


@pytest.mark.parametrize('fault',['bad-hash','public-mode','extent','wrong-control','wrong-kind','restarted','symlink','hardlink','duplicate','traversal','truncated','extra'])
def test_frozen_refusals_and_umask_restoration_preserved(tmp_path,fault):
    response,c,h,checksum=fixture_response(tmp_path,fault=fault)
    before=response.read_bytes()
    with mask(0o022):
        with pytest.raises((ValueError,KeyError,tarfile.TarError)):
            private.unpack(response,tmp_path/'saved',c,h,response_sha256=checksum)
        assert current_umask()==0o022
    assert response.read_bytes()==before
    assert not (tmp_path/'escape').exists()
    assert private.frozen.unpack is private._UNPACK


@pytest.mark.parametrize('fault',['response-hash','control-hash','existing-destination','public-parent','source-link'])
def test_preconditions_refuse_without_overwriting_sources(tmp_path,fault):
    response,c,h,checksum=fixture_response(tmp_path);dest=tmp_path/'saved';before=response.read_bytes()
    if fault=='response-hash':checksum='00'*32
    elif fault=='control-hash':h='ff'*32
    elif fault=='existing-destination':dest.mkdir(mode=0o700);(dest/'KEEP').write_text('preserve')
    elif fault=='public-parent':tmp_path.chmod(0o755)
    else:os.link(response,tmp_path/'response-link')
    try:
        with pytest.raises((ValueError,FileExistsError)):
            private.unpack(response,dest,c,h,response_sha256=checksum)
        assert response.read_bytes()==before
        if fault=='existing-destination':assert (dest/'KEEP').read_text()=='preserve'
        else:assert not dest.exists()
    finally:tmp_path.chmod(0o700)


def test_actual_saved_response_decodes_to_directly_usable_private_capsule():
    response=SESSION/'response.bin';c=json.loads((SESSION/'saved/control.json').read_bytes())
    h='af2c495cb86ab78b48e20a700358f6964182f015939fa18c5f7566aa5cdf0842'
    report=json.loads((ROOT/'report/checks/2026-11-14-missing-candidate-replay-v1-target-review.json').read_bytes())
    before=response.stat();dest=RAW/'actual-private-collection-corrected'
    with mask(0o022):
        private.unpack(response,dest,c,h,response_sha256=report['response']['sha256'])
        assert current_umask()==0o022
    raw=SESSION/'saved';header=None
    with response.open('rb') as f:f.readline();header=json.loads(f.readline())
    assert {str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()}==set(header['files'])
    for name,item in header['files'].items():
        p=dest/name;assert digest(p)==item['sha256'] and p.stat().st_mtime_ns==item['mtime_ns']
        assert stat.S_IMODE(p.stat().st_mode)==item['mode'] and p.stat().st_nlink==1
    for p in [dest,*dest.rglob('*')]:
        if p.is_dir():assert stat.S_IMODE(p.stat().st_mode)==0o700
    assert stat.S_IMODE((raw/'reference').stat().st_mode)==0o755
    assert response.stat().st_ino==before.st_ino and response.stat().st_mtime_ns==before.st_mtime_ns
    # Cold value/lineage I/O only: import no Torch, NumPy or scientific worker.
    from models.predictor_replay_capsule_v1 import Capsule
    with Capsule(dest/'reference',manifest_sha256='8107d13d5c663102f6e5e85dc19a5f6dea52bb12054a57d5957a3d12720a67fd') as reader:
        assert set(reader.states)=={'same_replay_original','fresh_process_candidate_reference'}
    (RAW/'actual-collection-check.json').write_text(json.dumps({'passed':True,'regular_files':len(header['files']),'logical_bytes':header['logical_bytes'],'raw_source_unchanged':True,'all_reconstructed_directories_private':True,'capsule_cold_reader_accepts_directly':True,'new_model_GPU_SSH_or_scientific_work':False},sort_keys=True,indent=2)+'\n')



def test_original_0644_source_inside_private_session_is_read_only(tmp_path):
    response,c,h,checksum=fixture_response(tmp_path);response.chmod(0o644)
    before=response.stat()
    private.unpack(response,tmp_path/'saved',c,h,response_sha256=checksum)
    assert stat.S_IMODE(response.stat().st_mode)==0o644
    assert response.stat().st_mtime_ns==before.st_mtime_ns
    assert stat.S_IMODE((tmp_path/'saved/reference').stat().st_mode)==0o700


def test_public_source_parent_refuses_before_reconstruction(tmp_path):
    source=tmp_path/'public-source';source.mkdir(mode=0o755)
    response,c,h,checksum=fixture_response(source);response.chmod(0o644)
    with pytest.raises(ValueError,match='private collection directory'):
        private.unpack(response,tmp_path/'saved',c,h,response_sha256=checksum)
    assert not (tmp_path/'saved').exists()


def test_pinned_source_is_not_reopened_and_replacement_still_refuses(tmp_path,monkeypatch):
    response,c,h,checksum=fixture_response(tmp_path);parked=tmp_path/'original-response'
    original_open=private._PinnedSource.open
    def replace_name_after_hash(self,mode):
        response.rename(parked)
        response.write_bytes(b'REPLACEMENT IS NOT THE RECORDED RESPONSE');response.chmod(0o600)
        return original_open(self,mode)
    monkeypatch.setattr(private._PinnedSource,'open',replace_name_after_hash)
    with mask(0o022):
        with pytest.raises(ValueError,match='Response changed during local reconstruction'):
            private.unpack(response,tmp_path/'saved',c,h,response_sha256=checksum)
        assert current_umask()==0o022
    # The pinned original supplied the decoded data; the replaced pathname was
    # never reopened. The final identity check still refuses the reconstruction.
    assert (tmp_path/'saved/reference/arrays/fixture.bin').read_bytes()==b'explicit local bytes only'
    assert parked.is_file() and response.read_bytes().startswith(b'REPLACEMENT')

"""Frozen native input packaging: actual source coverage and hostile tar refusal."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts import prepare_native_sdpa_capsule_v1 as capsule

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'runs/pretrained/diagnostic/predictor-component-startup-v2-20261107/capsule-final'


def tiny_archive(tmp_path, fault=None):
    content=b'frozen-native-input'
    entry={'source':'unused','source_mode':0o644,'bytes':len(content),
           'sha256':hashlib.sha256(content).hexdigest(),'mode':0o600,'mtime_ns':1_000_000_000}
    name='input/native.npy'
    if fault=='absolute':name='/input/native.npy'
    if fault=='traversal':name='input/../native.npy'
    if fault=='noncanonical':name='input//native.npy'
    if fault=='backslash':name='input\\native.npy'
    files={name:dict(entry)}
    archive=tmp_path/'capsule.tar.gz'
    with tarfile.open(archive,'w:gz',format=tarfile.USTAR_FORMAT) as tar:
        names=[name,name] if fault=='duplicate' else [name]
        if fault=='unknown':names.append('input/unknown.npy')
        for path in names:
            member=tarfile.TarInfo(path);member.size=entry['bytes'];member.mode=entry['mode'];member.mtime=1
            if fault=='mode':member.mode=0o644
            if fault=='mtime':member.mtime=2
            if fault=='symlink':member.type=tarfile.SYMTYPE;member.linkname='elsewhere';member.size=0
            if fault=='size':member.size-=1
            data=b'X'*len(content) if fault=='content' else content
            tar.addfile(member,io.BytesIO(data))
    if fault=='missing':files['input/missing.npy']=dict(entry)
    if fault=='public-manifest':files[name]['mode']=0o644
    manifest={'bytes':archive.stat().st_size,'sha256':capsule.digest(archive),'files':files}
    if fault=='archive-hash':manifest['sha256']='0'*64
    if fault=='archive-size':manifest['bytes']+=1
    return archive,manifest


def test_private_copy_preserves_bytes_from_more_permissive_source(tmp_path):
    archive,manifest=tiny_archive(tmp_path)
    assert capsule.verify_capsule(archive,manifest)
    assert manifest['files']['input/native.npy']['source_mode']==0o644
    assert manifest['files']['input/native.npy']['mode']==0o600


@pytest.mark.parametrize('fault',('absolute','traversal','noncanonical','backslash','duplicate','unknown',
    'mode','mtime','symlink','size','content','missing','public-manifest','archive-hash','archive-size'))
def test_archive_identity_metadata_paths_and_contents_cannot_be_relabeled(tmp_path,fault):
    archive,manifest=tiny_archive(tmp_path,fault)
    with pytest.raises(ValueError):capsule.verify_capsule(archive,manifest)


def test_existing_capsule_is_preserved_before_any_input_work(tmp_path,monkeypatch):
    output=tmp_path/'existing';output.mkdir();sentinel=output/'old';sentinel.write_bytes(b'old')
    def forbidden(*a,**k):raise AssertionError('Existing output must refuse before reading inputs.')
    monkeypatch.setattr(capsule,'selected_inputs',forbidden)
    with pytest.raises(FileExistsError):capsule.build(ROOT,output)
    assert sentinel.read_bytes()==b'old'


def test_actual_reviewed_capsule_covers_native_inputs_and_excludes_training_pixels():
    manifest=json.loads((FINAL/'manifest.json').read_text())
    assert manifest['new_target_execution_permitted'] is False
    assert manifest['new_model_gradient_GPU_or_target_work'] is False
    assert capsule.verify_capsule(FINAL/'native-sdpa-capsule.tar.gz',manifest)
    files=manifest['files'];native={k:v for k,v in files.items() if k.startswith('results/capture/')}
    assert len(files)==118 and len(native)==18
    assert sum(k.startswith('code/') for k in files)==97
    assert sum(i['bytes'] for i in native.values())==47_273_481
    assert 'code/input/training-clip.pt' not in files
    assert 'code/input/prior-EOT8-derivatives.pt' not in files
    assert all(i['mode']==0o600 for i in files.values())
    selected=capsule.selected_inputs(ROOT)
    assert selected==files
    for item in files.values():
        source=ROOT/item['source'];s=source.stat()
        assert (s.st_size,s.st_mtime_ns,s.st_mode&0o777)==(item['bytes'],item['mtime_ns'],item['source_mode'])
        assert capsule.digest(source)==item['sha256']

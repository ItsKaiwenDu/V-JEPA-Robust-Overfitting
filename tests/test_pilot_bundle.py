import io
from pathlib import Path
import tarfile

import pytest

from models.pilot_bundle import digest, extract_verified, make_archive


def test_archives_are_reproducible_and_restore_exact_bytes(tmp_path):
    original = tmp_path / 'file'
    original.write_bytes(b'source\n' * 50)
    a = make_archive(tmp_path / 'a.tar.gz', {'nested/file': original})
    b = make_archive(tmp_path / 'b.tar.gz', {'nested/file': original})
    assert a == b
    extract_verified(tmp_path / 'a.tar.gz', a, tmp_path / 'restored')
    assert (tmp_path / 'restored/nested/file').read_bytes() == original.read_bytes()
    with pytest.raises(FileExistsError):
        extract_verified(tmp_path / 'a.tar.gz', a, tmp_path / 'restored')


@pytest.mark.parametrize('defect', ['hash', 'member_hash', 'extra', 'duplicate', 'traversal', 'absolute', 'symlink'])
def test_invalid_handoff_is_rejected_before_writing_outputs(tmp_path, defect):
    source = tmp_path / 'input'
    source.write_text('data')
    archive = tmp_path / 'a.tar.gz'
    spec = make_archive(archive, {'input': source})
    if defect == 'hash':
        spec['sha256'] = 'wrong'
    elif defect == 'member_hash':
        spec['files']['input']['sha256'] = 'wrong'
    else:
        with tarfile.open(archive, 'w:gz') as target:
            names = {'extra': ['input', 'extra'], 'duplicate': ['input', 'input'],
                     'traversal': ['../escaped'], 'absolute': ['/tmp/escaped'], 'symlink': ['input']}[defect]
            for name in names:
                info = tarfile.TarInfo(name)
                if defect == 'symlink':
                    info.type, info.linkname = tarfile.SYMTYPE, '/tmp/elsewhere'
                    target.addfile(info)
                else:
                    info.size = 4
                    target.addfile(info, io.BytesIO(b'data'))
        spec['sha256'] = digest(archive)
    with pytest.raises(ValueError):
        extract_verified(archive, spec, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()

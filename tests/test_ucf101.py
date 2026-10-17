import pytest

from datasets.ucf101 import Video, audit_subsets, read_official, write_manifests


@pytest.fixture
def official(tmp_path):
    root = tmp_path / 'official'
    root.mkdir()
    (root / 'classInd.txt').write_text('1 ActionA\n2 ActionB\n')
    train, test = [], []
    for label, name in enumerate(('ActionA', 'ActionB'), 1):
        for group in range(1, 8):
            for clip in range(1, 6):
                path = f'{name}/v_{name}_g{group:02d}_c{clip:02d}.avi'
                if group < 6:
                    train.append(f'{path} {label}')
                else:
                    test.append(path)
    (root / 'trainlist01.txt').write_text('\n'.join(train) + '\n')
    (root / 'testlist01.txt').write_text('\n'.join(test) + '\n')
    return root


def test_grouped_counts_replay_and_input_order_independence(official, tmp_path):
    first = write_manifests(official, tmp_path / 'first', expected_classes=2, train_count=10)
    assert first['counts'] == {'train': 20, 'validation': 10, 'test': 10}
    for name in ('trainlist01.txt', 'testlist01.txt'):
        path = official / name
        path.write_text('\n'.join(reversed(path.read_text().splitlines())) + '\n')
    second = write_manifests(official, tmp_path / 'second', expected_classes=2, train_count=10)
    assert first['files_sha256'] == second['files_sha256']
    assert first['manifest_sha256'] == second['manifest_sha256']


def test_group_leakage_is_rejected():
    a = Video('a', 'A/a.avi', 0, 'A_g01')
    b = Video('b', 'A/b.avi', 0, 'A_g01')
    with pytest.raises(ValueError, match='leakage'):
        audit_subsets({'train': [a], 'validation': [b], 'test': []})


def test_official_group_overlap_is_rejected(official):
    with (official / 'testlist01.txt').open('a') as handle:
        handle.write('ActionA/v_ActionA_g01_c01.avi\n')
    with pytest.raises(ValueError, match='overlap'):
        read_official(official)


def test_manifest_output_is_never_overwritten(official, tmp_path):
    destination = tmp_path / 'manifest'
    write_manifests(official, destination, expected_classes=2, train_count=10)
    before = (destination / 'metadata.json').read_bytes()
    with pytest.raises(FileExistsError):
        write_manifests(official, destination, expected_classes=2, train_count=10)
    assert (destination / 'metadata.json').read_bytes() == before


def test_bad_label_and_path_are_rejected(official):
    path = official / 'trainlist01.txt'
    path.write_text('ActionA/v_ActionA_g01_c01.avi 2\n')
    with pytest.raises(ValueError, match='label'):
        read_official(official)
    path.write_text('../v_ActionA_g01_c01.avi 1\n')
    with pytest.raises(ValueError):
        read_official(official)


def test_official_case_only_filename_alias_is_preserved(official):
    path = official / 'trainlist01.txt'
    path.write_text(path.read_text().replace('v_ActionA_', 'v_actiona_'))
    _, train, _ = read_official(official)
    assert train[0].relative_path.startswith('ActionA/v_actiona_')
    assert train[0].source_group_id.startswith('ActionA_g')


def test_insufficient_class_pool_fails_without_partial_output(official, tmp_path):
    output = tmp_path / 'bad'
    with pytest.raises(ValueError, match='insufficient'):
        write_manifests(official, output, expected_classes=2, train_count=100)
    assert not output.exists()

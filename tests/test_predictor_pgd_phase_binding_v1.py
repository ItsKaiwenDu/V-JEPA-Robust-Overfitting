"""New v5 client/phase integration with explicit fake budget and OS ports."""
import ast
import base64
import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from types import SimpleNamespace as NS

import pytest
from models import predictor_pgd_phase_binding_v1 as h
from models import predictor_pgd_workload_handoff_outer_v5 as outer
from models import predictor_pgd_workload_handoff_client_v5 as client
from models import predictor_pgd_workload_budget_v1 as budget
from models import predictor_pgd_workload_runtime_records_v1  # Metadata definitions only.

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'runs/pretrained/diagnostic/pgd-phase-binding-v1-20261124'
EVIDENCE = Path(os.environ['VJEPA_PHASE_BINDING_EVIDENCE'])


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ports = module('_new_phase_bound_integration', STAGE/'fake-integrated-client-v1.py')
support,worker,preparation=ports.support,ports.worker,ports.preparation
ORIGIN=ports.ORIGIN
put=ports.put

@pytest.fixture(scope='module', autouse=True)
def forbid_actual_work():
    yield from support.forbid_work.__wrapped__()


@pytest.fixture
def area(request):
    path = EVIDENCE/request.node.name
    path.mkdir(mode=0o700)
    put(path/'FAKE-ORIGIN.json', {'origin': ORIGIN, 'actual_user_grant': False,
        'actual_target_work': False, 'actual_budget_qualification': False})
    return path



def undo(raw, edits):
    for edit in reversed(edits):
        a = edit['start']
        b = a+len(edit['after'].encode())
        assert raw[a:b] == edit['after'].encode()
        raw = raw[:a]+edit['before'].encode()+raw[b:]
    return raw


def test_phase_binding_new_binding_inverse_ast_shutdown_cache_and_source_identity(area):
    generated, derivation, cfg, files = h.assemble(ROOT)
    sources = {n: base64.b64decode(v) for n, v in json.loads(files[cfg['pins']['payload']]).items()}
    bases = {h.OUTER: files[cfg['pins']['outer']], h.CLIENT: files[cfg['pins']['client']], h.CLI: files[cfg['pins']['cli']]}
    bases.update({path: sources['supervisor/'+name] for name, path in h.READERS.items()})
    for name, raw in generated.items():
        assert raw == (ROOT/name).read_bytes() and undo(raw, derivation['edits'][name]) == bases[name]
    before = ast.parse(bases[h.OUTER])
    after = ast.parse(generated[h.OUTER])
    definitions = lambda tree: [ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert definitions(before) == definitions(after)
    assert len(outer.LOCAL_SOURCES) == 24 and len(cfg['files']) == 23
    assert len(sources) == 56 and sum(map(len, sources.values())) == 936711
    assert h.digest(files[cfg['pins']['control']]) == 'd997bf9ed74c8ebf4504a5e6214a35e28365f85e9da3d7738a5b58f5edaa9f29'
    assert not any(hasattr(h, name) for name in ('prepare_unarmed', 'material', 'authorization'))
    put(area/'RESULT.json', {'origin': ORIGIN, 'six_inverses': True, 'all_outer_function_class_ASTs_exact': True,
        'local_pins': 24, 'input_pins': 23, 'new_real_package_API': False})


def test_phase_binding_real_new_local_preflight_checks_fake_grant_and_exact_shutdown_bytes(area):
    folder = support.package(area)
    checked = client.check_local(folder)
    assert len(checked[0]['local_source_sha256']) == 24 and len(checked[7]) == 56
    assert checked[6]['user_basis'] == ORIGIN and checked[6]['full_model_gradients'] == 120
    assert checked[-1] == (ROOT/outer.PACKET_PATH,)
    assert not (folder/'ATTEMPT.json').exists()
    assert 'cleanup_owned' in base64.b64decode(checked[7]['supervisor/workload_watch_v1.py']).decode()
    put(area/'RESULT.json', {'origin': ORIGIN, 'preflight': 'PASS', 'execution_or_budget_qualified': False})


@pytest.mark.parametrize('armed', [False, True])
def test_phase_binding_real_budget_gate_refuses_before_clock_attempt_or_network(area, monkeypatch, armed):
    folder = support.package(area) if armed else area/'missing'
    def forbidden(*args, **kw):
        pytest.fail('No time, timer, protocol or connection before budget qualification')
    monkeypatch.setattr(client, 'time', NS(monotonic=forbidden))
    monkeypatch.setattr(outer, 'Alarms', forbidden)
    monkeypatch.setattr(client, '_budget_unqualified_protocol', forbidden)
    with pytest.raises(ValueError, match='budget is unqualified'):
        client.run(folder)
    assert not (folder/'ATTEMPT.json').exists() and not (folder/'attended-session').exists()
    put(area/'RESULT.json', {'origin': ORIGIN, 'budget_gate_refused': True,
        'fake_grant_present': armed, 'clock_attempt_timer_or_network_created': False})


@pytest.mark.parametrize('armed', [False, True])
def test_phase_binding_isolated_new_CLI_refuses_or_dispatches_closed_new_client(area, armed):
    folder = support.package(area) if armed else area/'missing'
    events = []
    fake_sys = NS(path=[])
    original = builtins.__import__
    def importing(name, *args, **kw):
        if name == 'sys':
            return fake_sys
        if name == 'models.predictor_pgd_workload_handoff_client_v5':
            assert fake_sys.path == [str(ROOT)]
            events.append('v5')
            return NS(run=client.run)
        assert not name.startswith('models.')
        return original(name, *args, **kw)
    namespace = {'__name__': '_FAKE_ISOLATED_V5_CLI', '__file__': str(ROOT/h.CLI),
        '__builtins__': {**vars(builtins), '__import__': importing}}
    exec(compile((ROOT/h.CLI).read_bytes(), str(ROOT/h.CLI), 'exec'), namespace)
    if armed:
        with pytest.raises(ValueError, match='budget is unqualified'):
            namespace['main']([str(folder)])
        assert events == ['v5']
    else:
        with pytest.raises(SystemExit, match='Unarmed'):
            namespace['main']([str(folder)])
        assert not events and not fake_sys.path
    assert not (folder/'ATTEMPT.json').exists()


@pytest.mark.parametrize('version', ['v1', 'v2', 'v3'])
def test_phase_binding_all_consumed_real_packages_refuse_new_preflight_without_mutation(area, version):
    tail = 'linux10614-primary-final-unarmed' if version == 'v1' else 'linux10614-primary-'+version+'-final-unarmed'
    folder = ROOT/f'runs/pretrained/diagnostic/pgd-workload-handoff-{version}-20261123/handoff-parent'/tail
    before = (folder/'ATTEMPT.json').read_bytes()
    with pytest.raises(ValueError):
        client.check_local(folder)
    assert (folder/'ATTEMPT.json').read_bytes() == before and json.loads(before)['consumed'] is True
    put(area/'RESULT.json', {'origin': ORIGIN, 'old_consumed_package_preserved': version})


@pytest.fixture(scope='module')
def pin_baseline():
    parent = EVIDENCE/'preflight-baseline'
    parent.mkdir(mode=0o700)
    support.package(parent)
    project = parent/'project'
    project.mkdir(mode=0o700)
    for name in [*outer.LOCAL_SOURCES, outer.PACKET_PATH]:
        put(project/name, (ROOT/name).read_bytes())
    return parent


@pytest.mark.parametrize('fault', ['client', 'bootstrap', 'budget-module', 'phase-reader', 'payload', 'runtime-source', 'archive', 'grant-kind'])
def test_phase_binding_new_pins_source_budget_and_grant_drift_refuse_without_attempt(area, pin_baseline, fault):
    copied = area/'copied'
    shutil.copytree(pin_baseline, copied)
    folder = copied/'fake-package'
    project = copied/'project'
    control = json.loads((folder/'CONTROL-WORKLOAD.json').read_bytes())
    control.update(repository=str(project), local_handoff=str(folder))
    raw = h.enc(control)
    (folder/'CONTROL-WORKLOAD.json').write_bytes(raw)
    grant = outer.authorization(control, h.digest(raw), ORIGIN)
    if fault in ('client', 'bootstrap', 'budget-module', 'phase-reader'):
        name = {'client': h.CLIENT, 'bootstrap': h.READERS['bootstrap.py'],
            'budget-module': h.BUDGET, 'phase-reader': 'models/predictor_pgd_phase_binding_records_v1.py'}[fault]
        (project/name).write_bytes(b'EXPLICIT FAKE PIN DRIFT\n')
    elif fault == 'payload':
        (folder/'SOURCE-PAYLOAD.json').write_bytes(b'{}\n')
    elif fault == 'runtime-source':
        payload = json.loads((folder/'SOURCE-PAYLOAD.json').read_bytes())
        payload['supervisor/workload_watch_v1.py'] = base64.b64encode(b'FAKE SHUTDOWN DRIFT').decode()
        (folder/'SOURCE-PAYLOAD.json').write_bytes(h.enc(payload))
    elif fault == 'archive':
        (project/outer.PACKET_PATH).write_bytes(b'FAKE ARCHIVE DRIFT')
    else:
        grant['kind'] = 'pgd-candidate-outer-university-exception-v1'
    (folder/'AUTHORIZATION.json').write_bytes(h.enc(grant))
    with pytest.raises(ValueError):
        client.check_local(folder)
    assert not (folder/'ATTEMPT.json').exists() and not (folder/'attended-session').exists()


@pytest.fixture(scope='module')
def transcript():
    area = EVIDENCE/'integrated-primary'
    area.mkdir(mode=0o700)
    folder = support.package(area)
    fixture = support.ClientFixture(area, folder)
    before = ({k: v for k, v in sys.modules.items() if k.split('.')[0] in ('models', 'attacks', 'datasets')}, list(sys.meta_path))
    fixture.result = fixture.run()
    after = ({k: v for k, v in sys.modules.items() if k.split('.')[0] in ('models', 'attacks', 'datasets')}, list(sys.meta_path))
    assert before == after
    return fixture


def test_phase_binding_new_client_server_worker_parent_and_transport_complete_fake_flow(transcript):
    fixture = transcript
    assert fixture.result == fixture.server.status == 0 and not fixture.proc.terminated
    assert bytes(fixture.proc.stdin.data) == fixture.checked[8]+fixture.server.input
    session = fixture.folder/'attended-session'
    receipt = json.loads((session/'COLLECTION-RECEIPT.json').read_bytes())
    state = json.loads((session/'saved/session.json').read_bytes())
    assert receipt['transport_verified'] and receipt['SSH_exit_status'] == 0
    assert receipt['phase_metadata']['completed_worker_roles'] == ['original-unit','candidate-unit','candidate-replay']
    assert receipt['phase_metadata']['parent_measurements_complete'] and not receipt['phase_metadata']['performance_qualified']
    assert state['completed'] == ['proof', 'original-unit', 'candidate-unit', 'candidate-replay'] and state['complete'] is False
    assert not json.loads((session/'saved-transport/PARTIAL-RECEIPT.json').read_bytes())['transport_verified']
    for role in state['completed'][1:]:
        preparation_row = json.loads((fixture.area/('preparation-'+role+'.json')).read_bytes())
        assert preparation_row['events'].index('transfer') < preparation_row['events'].index('checkpoint') < preparation_row['events'].index('unit')
    assert ['FAKE-BUDGET-PORT-NOT-ACTUAL-QUALIFICATION'] in fixture.events
    put(fixture.area/'TRANSCRIPT-RESULT.json', {'origin': ORIGIN, 'result': 'PASS', 'files': len(receipt['files']),
        'wire_bytes': receipt['wire_bytes'], 'wire_sha256': receipt['wire_sha256'],
        'exact_client_outbound_matches_server_input': True, 'private_project_namespaces_unchanged': True,
        'actual_target_budget_or_scientific_execution_qualified': False})


def test_phase_binding_failed_summary_preserves_old_worker_marker_and_failed_parent(area):
    reused = os.environ.get('VJEPA_PHASE_REUSE_FAILED')
    if reused:
        work = Path(reused)
        fixture = NS(folder=work/'fake-package')
        assert json.loads((work/'SERVER-EVENTS.json').read_bytes())['fake_status'] == 1
        assert (work/'CLIENT-OUTBOUND.bin').read_bytes() == (ROOT/h.OUTER).read_bytes()+(work/'SERVER-INPUT.bin').read_bytes()
    else:
        work = area/'integrated-phase-publish'
        work.mkdir(mode=0o700)
        fixture = support.ClientFixture(work, support.package(work))
        assert fixture.run() == fixture.server.status == 1
        assert bytes(fixture.proc.stdin.data) == fixture.checked[8]+fixture.server.input
    session = fixture.folder/'attended-session'
    state = json.loads((session/'saved/session.json').read_bytes())
    assert state['phase'] == 'failed' and state['completed'] == ['proof']
    assert state['error_type'] == 'RuntimeError' and state['complete'] is False
    assert json.loads((work/'child-error-950001.json').read_bytes())['type'] == 'OSError'
    prefix = session/'saved/capture/results/workload/original-unit'
    assert json.loads((prefix/'complete.json').read_bytes())['complete'] is True
    assert not (prefix/'phase-summary.json').exists()
    receipt = json.loads((session/'COLLECTION-RECEIPT.json').read_bytes())
    assert receipt['transport_verified'] and receipt['SSH_exit_status'] == 1
    assert receipt['phase_metadata']['completed_worker_roles'] == []
    assert receipt['phase_metadata']['parent_measurements_complete'] is False
    assert not receipt['phase_metadata']['performance_qualified']
    put(area/'SAVED-TRANSCRIPT-RESULT.json', {'origin': ORIGIN, 'result': 'PASS',
        'reused_saved_failure_without_worker':bool(reused),
        'old_worker_complete_true_retained':True, 'parent_failed':True,
        'wire_bytes':receipt['wire_bytes'],'wire_sha256':receipt['wire_sha256'],
        'files':len(receipt['files']),'exact_client_outbound_matches_server_input':True,
        'actual_target_budget_or_scientific_execution_qualified':False})


@pytest.mark.parametrize('fault', ['missing-summary','owner','deadline','counts',
    'exclusion','missing-parent','outer-domain','journal-kind'])
def test_phase_binding_saved_phase_semantics_refuse_without_worker_repetition(area, transcript, fault):
    from models import predictor_pgd_phase_binding_records_v1 as records
    source = transcript.folder/'attended-session/saved'
    destination = area/'saved'
    shutil.copytree(source, destination)
    receipt = json.loads((transcript.folder/'attended-session/COLLECTION-RECEIPT.json').read_bytes())
    name = 'capture/results/workload/original-unit/phase-summary.json'
    if fault == 'missing-parent':
        del receipt['files']['phase-parent-summary.json']
    elif fault == 'missing-summary':
        del receipt['files'][name]
    else:
        if fault == 'outer-domain':
            name = 'phase-outer-setup.json'
        elif fault == 'journal-kind':
            name = 'capture/results/workload/original-unit/phase-events.jsonl'
        path = destination/name
        if fault == 'journal-kind':
            rows = [json.loads(line) for line in path.read_bytes().splitlines()]
            previous = '0'*64
            for row in rows:
                row['kind'] = 'EXPLICIT-FAKE-OTHER-KIND'
                row['previous_sha256'] = previous
                row.pop('row_sha256')
                row['row_sha256'] = h.digest(h.enc(row))
                previous = row['row_sha256']
            raw = b''.join(h.enc(row) for row in rows)
        else:
            row = json.loads(path.read_bytes())
            if fault == 'owner': row['pid'] += 1
            elif fault == 'deadline': row['deadline'] += 1
            elif fault == 'counts': row['completed_spans']['owned_context'] -= 1
            elif fault == 'exclusion': row['scope_excludes_own_summary_publication_close_and_return'] = False
            elif fault == 'outer-domain': row['clock_domain'] = 'unrelated-performance-clock'
            raw = h.enc(row)
        path.write_bytes(raw)
        receipt['files'][name].update(bytes=len(raw),sha256=h.digest(raw),mtime_ns=path.stat().st_mtime_ns)
    with pytest.raises(ValueError):
        records.validate_collection(destination, receipt, transcript.checked[0]['supervised_control_text'].encode(),
                                    window=NS(check=lambda:None))
    put(area/'RESULT.json', {'origin':ORIGIN,'fault':fault,'saved_metadata_refused':True,
        'new_worker_lifecycle':False,'scientific_or_performance_qualified':False})


@pytest.mark.parametrize('fault', ['hash','hardlink','symlink','mode','collection-deadline'])
def test_phase_binding_saved_tree_drift_refuses_before_phase_admission(area, transcript, fault):
    from models import predictor_pgd_phase_binding_records_v1 as records
    destination = area/'saved'
    shutil.copytree(transcript.folder/'attended-session/saved', destination)
    receipt = json.loads((transcript.folder/'attended-session/COLLECTION-RECEIPT.json').read_bytes())
    path = destination/'phase-parent-summary.json'
    window = NS(check=lambda:None)
    if fault == 'hash':
        raw = path.read_bytes().replace(b'"parent"',b'"parenz"')
        path.write_bytes(raw)
        os.utime(path,ns=(receipt['files']['phase-parent-summary.json']['mtime_ns'],)*2)
    elif fault == 'hardlink': os.link(path,area/'additional-link')
    elif fault == 'symlink':
        target = area/'ordinary-target'
        path.rename(target)
        path.symlink_to(target)
    elif fault == 'mode': path.chmod(0o644)
    else:
        def exhausted(): raise TimeoutError('EXPLICIT FAKE continuing collection deadline')
        window.check = exhausted
    with pytest.raises((ValueError, OSError)):
        records.validate_collection(destination,receipt,transcript.checked[0]['supervised_control_text'].encode(),window=window)
    put(area/'RESULT.json',{'origin':ORIGIN,'fault':fault,'anchored_byte_reader_refused':True,
        'new_worker_lifecycle':False,'actual_target_or_network_work':False})


@pytest.mark.parametrize('fault', ['prefix', 'work-clock', 'header-root', 'header-origin', 'partial', 'trailing'])
def test_phase_binding_saved_response_mutations_refuse_without_new_worker_lifecycle(area, transcript, fault):
    source = io.BytesIO(transcript.server.output)
    frames = [outer.frame(source, lambda: None) for _ in range(3)]
    codec = transcript.server.codec
    _, header = codec.WireReader(source, NS(check=lambda: None)).frame(codec.MAGIC, codec.HEADER_CAP)
    tail = source.read()
    if fault == 'prefix':
        frames[1]['scratch'] = frames[1]['scratch'].replace('workload-v1-', 'candidate-v1-')
    elif fault == 'work-clock':
        frames[1]['work_deadline'] += 1
    elif fault == 'header-root':
        header['base']['root_binding']['root'] += '-other'
    elif fault == 'header-origin':
        header['base']['connection_started'] += 1
        header['base']['connection_deadline'] += 1
    elif fault == 'partial':
        tail = tail[:200]
    else:
        tail += b'EXPLICIT-FAKE-TRAILING'
    response = io.BytesIO()
    for item in frames:
        outer.send_frame(response, item, lambda: None)
    raw = h.enc(header)
    response.write(codec.MAGIC+len(raw).to_bytes(4, 'big')+raw+tail)
    raw = response.getvalue()
    put(area/'MUTATED-SERVER-OUTPUT.bin', raw)
    fixture = support.ClientFixture(area, golden=transcript, response=raw)
    with pytest.raises((ValueError, EOFError)):
        fixture.run()
    session = fixture.folder/'attended-session'
    assert (session/'CLAIM-OBSERVED.json').exists() and (session/'CLIENT-REFUSAL.json').exists()
    assert not (session/'COLLECTION-RECEIPT.json').exists() and fixture.proc.terminated
    if fault in ('prefix', 'work-clock'):
        assert not (session/'ROOT-RECEIPT.json').exists()
    else:
        assert (session/'ROOT-RECEIPT.json').exists()
    if fault in ('partial', 'trailing'):
        assert not json.loads((session/'saved-transport/PARTIAL-RECEIPT.json').read_bytes())['transport_verified']
    put(area/'RESULT.json', {'origin': ORIGIN, 'result': 'PASS', 'fault': fault, 'no_new_worker_lifecycle': True})

"""Standalone NEW stdlib synthetic proof; never load original guard/science/tests.

Sources, corpus and independent oracles must be frozen before this runner starts.
No stopwatch. All workload clock/process/signal/thread/network ports are denied.
"""
import _thread
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'runs/pretrained/diagnostic/pgd-expected-rows-v1-20261124'
MODEL = ROOT/'models/predictor_pgd_expected_rows_v1.py'
BLOCKED = []
EXECUTED = []
ast.unparse(ast.parse('ready=1')); json.dumps({'ready': True})


def load(path):
    assert stat.S_ISREG(path.lstat().st_mode)
    return json.loads(path.read_bytes())


def sha(path):
    assert stat.S_ISREG(path.lstat().st_mode)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deny(name):
    def refused(*args, **kwargs):
        BLOCKED.append(name)
        raise RuntimeError('Forbidden workload port: '+name)
    return refused


def block(event, args):
    if event == 'exec':
        if args[0].co_filename == str(MODEL) and not EXECUTED:
            EXECUTED.append(str(MODEL)); return
        return deny(event)()
    if (event in ('subprocess.Popen', 'os.system', 'os.fork', 'os.posix_spawn', 'os.kill', 'os.killpg',
                  'os.pidfd_open', 'signal.pidfd_send_signal') or event.startswith('socket.') or
            (event == 'import' and args[0].split('.')[0] not in sys.stdlib_module_names)):
        return deny(event)()


class ProbeFailure(RuntimeError): pass
class ProbeAbort(BaseException): pass
class NameSubclass(str): pass


class Poison:
    def __iter__(self): raise AssertionError('Poison storage was iterated')
    def __bool__(self): raise AssertionError('Poison storage was truth-tested')
    def clear(self): raise AssertionError('Poison storage clear was called')


class RootSpy(list):
    def __init__(self, rows, events, fail=False):
        super().__init__(rows); self.events = events; self.fail = fail
    def __iter__(self):
        self.events.append(['root_iter'])
        for index in range(len(self)):
            if self.fail and index == 1:
                self.events.append(['root_error', index]); raise ProbeFailure('iteration-failure')
            self.events.append(['root_row', index]); yield self[index]
        self.events.append(['root_done'])


class RowSpy:
    def __init__(self, values, events, label): self.values = values; self.events = events; self.label = label
    def __iter__(self):
        self.events.append(['row_iter', self.label])
        for index, value in enumerate(self.values):
            self.events.append(['row_value', self.label, index]); yield value
        self.events.append(['row_done', self.label])


class Capture:
    def __init__(self, values, events, error=False):
        self.values = values; self.events = events; self.error = error; self.captured = None
    def __ne__(self, other):
        if self.error:
            self.events.append(['compare_error']); raise ProbeFailure('comparison-failure')
        self.events.append(['compare_capture']); self.captured = other
        return self.values != other


class CaptureList(list):
    def __init__(self, values, events): super().__init__(values); self.events = events; self.captured = None
    def __ne__(self, other):
        self.events.append(['compare_capture']); self.captured = other
        return list.__ne__(self, other)


def inputs(frame, events, retained_plain):
    supplied = frame['records']
    rows = [tuple(row) if type(row) is list else row for row in supplied] if supplied is not None else None
    observed = [tuple(row) for row in frame['observed']]
    recipe = frame['recipe']
    if recipe == 'plain':
        if rows is not None:
            retained_plain[:] = rows; rows = retained_plain
    elif recipe in ('outer-spy', 'root-error', 'root-generator'):
        rows = RootSpy(rows, events, recipe == 'root-error')
        if recipe == 'root-generator': rows = iter(rows)
    elif recipe == 'row-spy': rows[0] = RowSpy(rows[0], events, 0)
    elif recipe == 'name-subclass': rows[0] = (NameSubclass(rows[0][0]),)+rows[0][1:]
    elif recipe == 'observed-capture': observed = Capture(observed, events)
    elif recipe == 'observed-list-subclass': observed = CaptureList(observed, events)
    elif recipe == 'observed-error': observed = Capture(observed, events, True)
    else: raise AssertionError('Unknown frozen recipe: '+recipe)
    return rows, observed


def one_case(case, arm, baseline, owner_type):
    kinds = ('parameters', 'buffers', 'modules')
    owners = {kind: owner_type(kind) for kind in kinds} if arm == 'candidate' else {}
    plain = {kind: [] for kind in kinds}
    results = []; failures = []

    def compare(kind, frame, events):
        records, observed = inputs(frame, events, plain[kind]); index = 0
        owner = owners.get(kind)
        action = frame['action']
        if owner is not None:
            if action == 'dirty-list': owner._rows.extend([('untrusted-primitive', -1)])
            elif action == 'poison-storage': owner._rows = Poison()
            elif action == 'close': owner.close()
        def identity(token):
            nonlocal index
            slot = index; index += 1; events.append(['lookup', token])
            if frame['nested'] and slot == 0:
                events.append(['nested_begin'])
                inner_rows = [tuple(frame['nested_record'])]; inner_observed = [tuple(frame['nested_observed'])]
                def inner_identity(value): events.append(['lookup', value]); return 909
                value = owner.compare(inner_rows, inner_observed, inner_identity) if owner else baseline(kind, inner_rows, inner_observed, inner_identity)
                if owner is not None:
                    assert owner.diagnostic()['route'] == 'fallback' and owner.diagnostic()['busy']
                events.append(['nested_end', value])
            if frame['fail_at'] == slot:
                if frame['abort']: raise ProbeAbort('identity-abort')
                raise ProbeFailure('identity-failure')
            return frame['identity_returns'][slot]
        result = error = None
        try:
            result = owner.compare(records, observed, identity) if owner else baseline(kind, records, observed, identity)
        except BaseException as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        diagnostic = owner.diagnostic() if owner else None
        if owner is not None:
            if diagnostic['busy'] or diagnostic['retained_rows'] != 0:
                failures.append('candidate retained row references or active state')
            if diagnostic['route'] != frame['expected_route']:
                failures.append('route: '+str(diagnostic['route']))
            if type(observed) in (Capture, CaptureList) and observed.captured is owner._rows:
                failures.append('foreign comparison received reusable backing list')
        return {'result': result, 'error': error, 'diagnostic': diagnostic}

    if case['kind'] == 'driver':
        events = []
        def reached(kind):
            events.append(['enumerate', kind]); result = compare(kind, case['roles'][kind], events)
            assert result['error'] is None
            return result['result']
        def grad_scan():
            for index, value in enumerate(case['grad_values']):
                events.append(['grad_read', index])
                if value: return True
            return False
        result = reached('parameters') or reached('buffers') or grad_scan() or reached('modules')
        results.append({'result': result, 'error': None, 'events': events})
        if result is not case['expected_result'] or events != case['expected_events']:
            failures.append('driver independent result/trace mismatch')
    else:
        for frame in case['frames']:
            events = []; result = compare(case['kind'], frame, events); result['events'] = events
            expected_error = frame['expected_error']
            if expected_error is None:
                if result['error'] is not None or result['result'] is not frame['expected_result']:
                    failures.append('independent comparison result mismatch')
            elif (result['error'] is None or result['error']['type'] != expected_error['type'] or
                  expected_error['contains'] not in result['error']['message'] or result['result'] is not None):
                failures.append('independent primary error mismatch')
            if events != frame['expected_events']: failures.append('independent lookup/iteration/compare trace mismatch')
            results.append(result)
    for owner in owners.values():
        owner.close(); assert owner.diagnostic()['retained_rows'] == 0 and not owner.diagnostic()['busy']
    return {'frames': results, 'failures': failures, 'passed': not failures}


def invariant(check, baseline, owner_type):
    ident = check['id']; metadata = {}; events = []
    records = [('w', 11, True), ('bias', 12, False)]
    observed = [('w', 101, True), ('bias', 102, False)]
    owner = owner_type('parameters')
    def identity(token): events.append(['lookup', token]); return {11: 101, 12: 102}[token]
    if ident == 'late-reentrant-prefix':
        nested_routes = []
        def reentrant(token):
            events.append(['lookup', token])
            if token == 12:
                events.append(['nested_begin'])
                def inner(value): events.append(['lookup', value]); return 909
                result = owner.compare([('nested', 99, True)], [('nested', 909, True)], inner)
                nested_routes.append(owner.diagnostic()['route'])
                events.append(['nested_end', result])
            return {11: 101, 12: 102}[token]
        result = owner.compare(records, observed, reentrant)
        assert result is check['expected_result'] and events == check['expected_events'] and nested_routes == ['fallback']
        metadata = {'nested_fallback': True, 'outer_prefix_preserved': True, 'events': events}
    elif ident == 'fresh-row-tuples':
        backing = owner._rows; held = []
        def observing(token):
            events.append(['lookup', token])
            if token == 12: held.append(owner._rows[0])
            return {11: 101, 12: 102}[token]
        assert owner.compare(records, observed, observing) is False
        assert owner.compare(records, observed, observing) is False
        assert (owner._rows is backing) is check['expected_outer_list_same']
        assert (held[0] is not held[1]) is check['expected_tuple_rows_distinct']
        assert held[0] == held[1] == ('w', 101, True) and events == check['expected_events']
        metadata = {'outer_list_same': True, 'tuple_rows_distinct': True, 'events': events}
    elif ident == 'owner-isolation':
        other = owner_type('parameters'); assert owner._rows is not other._rows
        owner._rows.append(('dirty', -1, False)); assert other.diagnostic()['retained_rows'] == 0
        assert other.compare(records, observed, identity) is False
        other.close(); assert not owner.diagnostic()['closed']
        metadata = {'distinct_backing_lists_and_close_state': True}
    elif ident == 'diagnostic-copy':
        exposed = owner.diagnostic(); exposed.update({'retained_rows': 999, 'closed': True, 'busy': True})
        assert owner.diagnostic()['retained_rows'] == check['expected_retained_rows']
        assert not owner.diagnostic()['closed'] and not owner.diagnostic()['busy']
        assert owner.compare(records, observed, identity) is False
        metadata = {'diagnostic_mutation_does_not_change_owner': True}
    elif ident == 'no-retained-input-references':
        assert owner.compare(records, observed, identity) is False
        assert owner.diagnostic()['retained_rows'] == check['expected_retained_rows']
        assert all(value is not records and value is not observed and value is not identity for value in vars(owner).values())
        assert all(type(value) in (str, bool, list) for value in vars(owner).values())
        metadata = {'owner_keeps_no_input_observer_or_row_references_after_call': True, 'physical_memory_measured': False}
    elif ident == 'ordinary-interface-results':
        result = owner.compare(records, observed, identity)
        assert type(result).__name__ == check['expected_compare_type']
        assert owner.close() is check['expected_close_result']
        assert all(type(value) in (str, bool, int) for value in owner.diagnostic().values())
        metadata = {'ordinary_results_do_not_export_backing_list': True, 'Python_reflection_isolation_proven': False}
    else: raise AssertionError('Unknown frozen invariant: '+ident)
    owner.close(); assert owner.diagnostic()['retained_rows'] == 0
    return metadata


def main():
    plan = load(STAGE/'FROZEN-PLAN-v1.json')
    for name, want in plan['sources_and_oracles'].items(): assert sha(ROOT/name) == want
    assert plan['maximum_new_real_attempts'] == 0 and not plan['budget_gate_opened']
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(ast.parse(MODEL.read_text())))
    corpus = load(ROOT/'configs/predictor_pgd_expected_rows_cases_v1.json')['cases']
    invariants = load(STAGE/'INVARIANT-ORACLES-v1.json')['checks']
    assert [case['id'] for case in corpus] == plan['cases']
    assert [check['id'] for check in invariants] == plan['invariants']
    evidence = STAGE/'synthetic-results'; evidence.mkdir(mode=0o700)
    sys.addaudithook(block)
    for module, names in ((time, ('time', 'time_ns', 'monotonic', 'monotonic_ns', 'perf_counter', 'perf_counter_ns', 'process_time', 'process_time_ns', 'thread_time', 'thread_time_ns', 'sleep')),
                          (signal, ('alarm', 'setitimer', 'getitimer', 'signal', 'raise_signal', 'pthread_kill', 'pthread_sigmask', 'pidfd_send_signal')),
                          (os, ('getpid', 'getppid', 'fork', 'forkpty', 'posix_spawn', 'posix_spawnp', 'system', 'kill', 'killpg', 'pidfd_open')),
                          (socket, ('socket', 'create_connection', 'getaddrinfo')),
                          (subprocess, ('Popen', 'run', 'call', 'check_call', 'check_output')),
                          (_thread, ('start_new_thread',))):
        for name in names:
            if hasattr(module, name): setattr(module, name, deny(module.__name__+'.'+name))
    threading.Thread.start = deny('threading.Thread.start')
    scope = {'__name__': 'local_metadata_expected_rows_v1'}
    exec(compile(MODEL.read_bytes(), str(MODEL), 'exec'), scope)
    baseline = scope['baseline']; owner_type = scope['PrivateOuterList']; summaries = []
    for case in corpus:
        transcript = {'id': case['id'], 'kind': case['kind'], 'category': case['category']}
        for arm in ('baseline', 'candidate'):
            try: transcript[arm] = one_case(case, arm, baseline, owner_type)
            except BaseException as exc:
                transcript[arm] = {'passed': False, 'unexpected_error': {'type': type(exc).__name__, 'message': str(exc)}}
        passed = transcript['baseline']['passed'] and transcript['candidate']['passed']
        if passed:
            fields = lambda value: [{key: row[key] for key in ('result', 'error', 'events')} for row in value['frames']]
            passed = fields(transcript['baseline']) == fields(transcript['candidate'])
        transcript['result'] = 'PASS' if passed else 'FAIL'
        with (evidence/(case['id']+'.json')).open('x') as stream:
            stream.write(json.dumps(transcript, sort_keys=True, indent=2)+'\n')
        summaries.append({'id': case['id'], 'result': transcript['result']})
    for check in invariants:
        transcript = {'id': check['id'], 'category': 'privileged-synthetic-owner-invariant'}
        try: transcript['metadata'] = invariant(check, baseline, owner_type); transcript['result'] = 'PASS'
        except BaseException as exc:
            transcript['result'] = 'FAIL'; transcript['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        with (evidence/('invariant-'+check['id']+'.json')).open('x') as stream:
            stream.write(json.dumps(transcript, sort_keys=True, indent=2)+'\n')
        summaries.append({'id': 'invariant-'+check['id'], 'result': transcript['result']})
    assert not BLOCKED and EXECUTED == [str(MODEL)]
    for name, want in plan['sources_and_oracles'].items(): assert sha(ROOT/name) == want
    fixtures = {}
    for path in [evidence]+sorted(evidence.iterdir()):
        info = path.lstat(); assert info.st_dev == 16777229
        row = {'mode': stat.S_IMODE(info.st_mode), 'uid': info.st_uid, 'nlink': info.st_nlink, 'mtime_ns': info.st_mtime_ns}
        if path == evidence: row['directory'] = True
        else: row.update({'bytes': info.st_size, 'sha256': sha(path)})
        fixtures[str(path.relative_to(ROOT))] = row
    with (STAGE/'fixture-layout-preserved.json').open('x') as stream:
        stream.write(json.dumps(fixtures, sort_keys=True, indent=2)+'\n')
    result = {'result': 'PASS' if all(row['result'] == 'PASS' for row in summaries) else 'FAIL',
              'checks': summaries, 'new_cases': len(corpus), 'new_invariants': len(invariants),
              'new_checks_executed': len(summaries), 'new_fixture_entries': len(fixtures),
              'frozen_sources_unchanged_after_first_checks': True, 'blocked_port_attempts': BLOCKED,
              'only_new_synthetic_module_executed': True, 'no_stopwatch': True,
              'actual_target_science_thread_GPU_proof_network_or_training': False}
    with (STAGE/'FIRST-CHECKS-v1.json').open('x') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'checks'}, sort_keys=True))
    return 0 if result['result'] == 'PASS' else 1


if __name__ == '__main__': sys.exit(main())

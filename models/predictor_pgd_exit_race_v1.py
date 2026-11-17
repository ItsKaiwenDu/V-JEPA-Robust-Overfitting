"""Cold repair recipe for the exact consumed parent monitor, never a launcher.

Missing process metadata gets at most one bounded wait on the already-owned
Popen child. All existing acceptance, RSS, deadline and pidfd cleanup gates stay.
"""
import ast
import hashlib

FROZEN_SHA = 'ab23159993939318b6d28946a25fe3c4be0427ed6e9732e1530cb8246d6d9cdb'
OLD = """            except (FileNotFoundError,ProcessLookupError):
                status=child.poll()
                if status is not None:break
                raise
"""
NEW = """            except (FileNotFoundError,ProcessLookupError) as missing:
                status=child.poll()
                if status is None:
                    # An exiting owned child may lose proc metadata before poll
                    # exposes its status. Never retry sampling or guess success.
                    now=clock()
                    if type(now) not in (int,float) or not math.isfinite(now) or now<last or now>=phase_deadline or now>=work_deadline:
                        raise TimeoutError('Original parent deadline reached before owned exit confirmation.') from missing
                    last=now
                    try:status=child.wait(timeout=min(.1,phase_deadline-now,work_deadline-now))
                    except subprocess.TimeoutExpired:
                        missing.add_note('Owned child exit unconfirmed within the original bounded wait; refuse.')
                        raise missing
                    except Exception as waiting:
                        missing.add_note('Owned child exit wait also failed: '+type(waiting).__name__)
                        raise missing from waiting
                    now=clock()
                    if type(now) not in (int,float) or not math.isfinite(now) or now<last or now>=phase_deadline or now>=work_deadline:
                        raise TimeoutError('Original parent deadline reached after owned exit confirmation.') from missing
                    last=now
                if status is not None:break
                raise
"""


def build(frozen):
    if type(frozen) is not bytes or hashlib.sha256(frozen).hexdigest() != FROZEN_SHA:
        raise ValueError('Exact consumed monitor source required; never patch another revision.')
    text = frozen.decode()
    if text.count(OLD) != 1 or text.count('import stat\n') != 1:
        raise ValueError('Exact sampling-exit boundary required.')
    candidate = text.replace('import stat\n', 'import stat\nimport subprocess\n', 1).replace(OLD, NEW, 1).encode()
    compile(candidate, '<UNEXECUTED-OWNED-EXIT-MONITOR>', 'exec')
    inverse = candidate.decode().replace(NEW, OLD, 1).replace('import stat\nimport subprocess\n', 'import stat\n', 1).encode()
    if inverse != frozen:
        raise ValueError('Exact monitor byte inverse failed.')
    trees = [ast.parse(v) for v in (frozen, candidate)]
    defs = [{n.name: n for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))} for t in trees]
    if defs[0].keys() != defs[1].keys():
        raise ValueError('Monitor definition surface changed.')
    for name in defs[0].keys()-{'watch'}:
        if ast.dump(defs[0][name], include_attributes=False) != ast.dump(defs[1][name], include_attributes=False):
            raise ValueError('Process identity/handle helpers changed: '+name)
    return candidate, {'kind': 'pgd-owned-exit-monitor-candidate-v1', 'frozen_sha256': FROZEN_SHA,
                       'candidate_sha256': hashlib.sha256(candidate).hexdigest(), 'candidate_bytes': len(candidate),
                       'whole_module_exact_inverse_verified': True, 'changed_function': 'watch',
                       'maximum_confirmation_wait_seconds': 0.1, 'confirmation_uses_original_phase_and_work_deadlines': True,
                       'no_sampling_retry_or_exit_status_guess': True, 'original_RSS_sample_and_owned_task_exit_gates_unchanged': True,
                       'ProcReader_PidfdLease_and_cleanup_unchanged': True, 'authorized': False,
                       'scientific_qualified': False, 'training_admitted': False}

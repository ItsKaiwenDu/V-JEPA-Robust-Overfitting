"""Bounded owned cleanup definitions; callers must supply clock and OS ports.

These definitions authorize no launch. A closed descriptor never proves death.
The two-second total cleanup allowance comes from the existing work overhead.
"""
import math
import signal


def bounded_error(error):
    try:
        message = str(error)[:512]
    except BaseException:
        message = '<exception message unavailable>'
    return {'type': type(error).__name__[:96], 'message': message}


def error_chain(error):
    """Retain at most four explicit causes/contexts, without traversing cycles."""
    rows = []
    seen = set()
    current = error
    while current is not None and id(current) not in seen and len(rows) < 4:
        seen.add(id(current))
        row = bounded_error(current)
        notes = getattr(current, '__notes__', ())
        if type(notes) not in (list, tuple):
            notes = ()
        row['notes'] = [note[:256] for note in notes[:4] if type(note) is str]
        following = current.__cause__
        row['next_relation'] = 'cause' if following is not None else 'context'
        if following is None:
            following = current.__context__
        rows.append(row)
        current = following
    return {'exceptions': rows, 'truncated_or_cycle': current is not None}


def cleanup_owned(child, leases, root_lease, primary, *, work_deadline,
                  phase_deadline, clock, pause, last, samples, peak):
    """Keep held handles through TERM/KILL, bounded death confirmation and reap.

    Maximum total allowance is two seconds, clamped to the original hard work
    deadline. TERM gets at most .5s; KILL confirmation/reap uses the remainder.
    Unconfirmed handles remain held, and the caller must refuse clock handoff.
    """
    errors = []
    error_count = 0
    dead = {}
    status = None
    attempts = 0
    start = end = None
    closed_all = False

    def problem(error):
        nonlocal error_count
        error_count += 1
        if len(errors) < 8:
            errors.append(bounded_error(error))

    def moment():
        nonlocal last
        now = clock()
        if (type(now) not in (int, float) or not math.isfinite(now)
                or now < last):
            raise ValueError('Continuing cleanup clock is invalid; no reset.')
        last = now
        return now

    def observe():
        nonlocal status
        for key, lease in leases.items():
            try:
                if lease.closed:
                    raise ValueError('Unexpected closed owned handle; never reuse it.')
                value = lease.dead()
                if type(value) is not bool:
                    raise ValueError('Typed held-handle death observation required.')
                dead[key] = value
            except BaseException as error:
                dead[key] = None
                problem(error)
        try:
            value = child.poll()
            if value is not None and type(value) is not int:
                raise ValueError('Typed terminal child status required.')
            status = value
        except BaseException as error:
            status = None
            problem(error)

    def send(number):
        for key, lease in leases.items():
            if dead.get(key) is False:
                try:
                    if lease.closed:
                        raise ValueError('Closed handle cannot be signalled.')
                    lease.send(number)
                except BaseException as error:
                    problem(error)

    def all_dead():
        return len(dead) == len(leases) and all(value is True for value in dead.values())

    try:
        if (not leases or len(leases) > 64
                or leases.get((root_lease.pid, root_lease.birth)) is not root_lease
                or type(work_deadline) not in (int, float) or not math.isfinite(work_deadline)
                or type(phase_deadline) not in (int, float) or not math.isfinite(phase_deadline)):
            raise ValueError('Original bounded owned cleanup context required.')
        start = moment()
        end = min(work_deadline, start + 2.)
        term_end = min(end, start + .5)
        observe()
        send(signal.SIGTERM)
        while not all_dead() and attempts < 8:
            now = moment()
            if now >= term_end:
                break
            attempts += 1
            pause(min(.1, term_end - now))
            observe()
        observe()
        send(signal.SIGKILL)
        while attempts < 32:
            observe()
            if all_dead() and status is not None:
                break
            now = moment()
            if now >= end:
                break
            attempts += 1
            interval = min(.1, end - now)
            before = now
            try:
                # Popen.wait is a wait/reap operation, never a numeric PID signal.
                child.wait(timeout=interval)
            except BaseException as error:
                # TimeoutExpired is an observation; retain it without replacing
                # the primary failure. Other wait failures use the same bound.
                problem(error)
            now = moment()
            if now >= end:
                break
            # A failing fake/OS wait may return immediately. Avoid a busy loop.
            if now - before < interval:
                pause(min(interval - (now - before), max(0., end - now)))
        observe()
        moment()
        if all_dead() and status is not None:
            for lease in leases.values():
                try:
                    lease.close()
                except BaseException as error:
                    problem(error)
            closed_all = all(lease.closed for lease in leases.values())
    except BaseException as error:
        problem(error)
    confirmed = all_dead() and status is not None and closed_all
    record = {
        'kind': 'whole-pgd-owned-shutdown-observation-v1',
        'primary': bounded_error(primary) if primary is not None else None,
        'phase_deadline': phase_deadline, 'work_deadline': work_deadline,
        'cleanup_started': start, 'cleanup_deadline': end,
        'last_observed_clock': last, 'maximum_cleanup_seconds': 2.,
        'wait_iterations': attempts, 'exit_status': status,
        'all_observed_owned_dead': all_dead(), 'all_handles_closed': closed_all,
        'confirmed_idle': confirmed, 'errors': errors, 'error_count': error_count,
        'RSS_samples_before_failure': samples, 'peak_observed_RSS_bytes': peak,
        'owned_identities': [{'pid': key[0], 'birth': key[1],
                              'dead': dead.get(key), 'handle_closed': lease.closed}
                             for key, lease in list(leases.items())[:64]],
        'observation_authenticates_execution': False,
        'observed_tree_is_exhaustive': False, 'scientific_qualified': False,
        'training_admitted': False,
    }
    root_lease.shutdown_observation = record
    if primary is not None:
        primary.owned_shutdown = record
        if not confirmed or errors:
            primary.add_note('Owned shutdown recorded; confirmed idle=' + str(confirmed)
                             + '; cleanup errors=' + str(error_count))
        return record
    if not confirmed or errors:
        error = RuntimeError('Owned shutdown refused; confirmed idle=' + str(confirmed))
        error.owned_shutdown = record
        raise error
    return record

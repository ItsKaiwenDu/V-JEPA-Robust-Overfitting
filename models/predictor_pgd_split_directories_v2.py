"""Pure local setup-directory plan; no filesystem, process or scientific API.

This is a correction candidate, not a runner or an authorization. The consumed
v1 operator and all numerical/component source remain unchanged.
"""
from pathlib import PurePosixPath

SHELL={'.','capture','capture/results','capture/logs','cache','cache/cuda','tmp'}
WORKER_CLAIMS={
 'numerical':'capture/results/numerical',
 'boundary':'capture/results/boundary',
 'native':'capture/results/components/native',
 'forward-preserving':'capture/results/components/forward-preserving',
 **{'objective-'+str(i):'capture/results/numerical/objective-'+str(i) for i in range(8)}}


def plan(layout):
    """Select only source ancestors and supervisor-owned shell directories.

    The caller must first construct the original verified Layout and enforce the
    fresh root/source/host/authorization gates. Allowed result names are a
    validation catalogue; they do not transfer directory ownership to setup.
    """
    if (type(layout.assets) is not dict or len(layout.assets)!=172
            or type(layout.result) is not dict or len(layout.result)!=419
            or type(layout.directories) is not set):
        raise ValueError('Original full source/result catalogue required.')
    selected=set(SHELL)
    for name in layout.assets:
        path=PurePosixPath(name)
        if (not name or path.is_absolute() or str(path)!=name or any(p in ('.','..') for p in path.parts)
                or name.startswith('capture/')):
            raise ValueError('Only original source assets may contribute setup ancestors.')
        selected.update(str(parent) for parent in path.parents)
    if any(n.startswith('capture/results/') for n in selected):
        raise ValueError('Scientific result subtrees belong to their workers, never setup.')
    for name in selected:layout.directory(name)
    if not set(WORKER_CLAIMS.values())<=layout.directories:
        raise ValueError('Original worker-owned namespace changed.')
    return {'kind':'pgd-split-minimal-setup-directory-plan-v2',
        'setup_directories':sorted(selected,key=lambda n:(len(PurePosixPath(n).parts),n)),
        'worker_claim_directories':dict(WORKER_CLAIMS),
        'allowed_but_not_precreated':sorted(layout.directories-selected),
        'authorized':False,'target_helper_ready':False,'scientific_qualified':False,'training_admitted':False}

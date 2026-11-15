"""One fresh candidate objective with old references and a new result identity.

The frozen scientific body gains only a reference-binding state field. Reference
I/O is explicit and separate from the new result writer; no old header is edited.
"""
import ast
import hashlib
import io
from pathlib import Path
from types import FunctionType, SimpleNamespace
from models import predictor_forward_preserving_worker_v1 as frozen
from models import predictor_missing_replay_records_v1 as records
from models import predictor_replay_capsule_v1 as capsule

FROZEN_WORKER_SHA = '55d20ed67993b5fde7080393b92ec770b85ee40af46cf374e88d1551f56ccfb3'
ROLE = {'original': 'same_replay_original', 'replay': 'fresh_process_candidate_reference'}


def body_ast():
    path = Path(frozen.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_WORKER_SHA:
        raise ValueError('Frozen one-objective scientific worker changed.')
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    positions = [i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                 and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'state']
    if len(positions) != 1:
        raise ValueError('Exact frozen state construction required.')
    node.body.insert(positions[0]+1, ast.parse('state["reference_binding"] = args.reference_binding').body[0])
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


def bound_body(reader, sha):
    expected = reader.execution_binding(sha)
    if expected != records.binding(sha):
        raise ValueError('Capsule-to-new-result lineage differs.')
    tokens = {name: object() for name in ROLE}
    def source_load(token, *, control_sha256, index, live):
        if control_sha256 != sha or live is not True:
            raise ValueError('Reference request must name this new execution explicitly.')
        names = [name for name, value in tokens.items() if token is value]
        if len(names) != 1 or index != capsule.ROLES[ROLE[names[0]]]:
            raise ValueError('Wrong original/replay reference role; partials are forbidden.')
        # Old source identity is checked by the capsule's unchanged verifier, never
        # against the new execution ID supplied to this routing boundary.
        return reader.load_reference(ROLE[names[0]])
    def arrays_from(token, state):
        import numpy as np
        names = [name for name, value in tokens.items() if token is value]
        if len(names) != 1:
            raise ValueError('Only mapped complete reference arrays may be read.')
        role = ROLE[names[0]]
        if state != reader.load_reference(role):
            raise ValueError('Reference state changed before value comparison.')
        folder = reader._folders[role]
        return {name: np.load(io.BytesIO(reader._mapped_read(folder, item['file'], 16*2**20)), allow_pickle=False)
                for name, item in state['artifacts'].items()}
    view = SimpleNamespace(**{name: getattr(records, name) for name in
                             ('WORKER_KIND', 'atomic', 'LIMITS', 'sha_file', 'check_scope', 'write_final')})
    view.load = source_load
    namespace = dict(frozen.run.__globals__, records=view, arrays_from=arrays_from)
    exec(compile(body_ast(), str(Path(__file__)), 'exec'), namespace)
    return namespace['run'], tokens, expected


def run(args):
    if args.index != 3 or args.tiny is not False or args.device != 'cuda':
        raise ValueError('Only one new full-size CUDA candidate replay is allowed.')
    with capsule.Capsule(args.reference_capsule, manifest_sha256=records.MANIFEST_SHA) as reader:
        body, tokens, expected = bound_body(reader, args.control_sha256)
        values = dict(vars(args), reference=tokens['original'], replay_reference=tokens['replay'], reference_binding=expected)
        return body(SimpleNamespace(**values))

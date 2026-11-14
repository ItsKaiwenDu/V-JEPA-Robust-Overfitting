"""New scope/claim; frozen verified transport and tar framing, no standalone retrieval."""
import base64
import hashlib
import json
from pathlib import Path
from types import FunctionType
from models import predictor_eot8_record_client_v2 as transport
from models import predictor_forward_preserving_operator_v1 as op
old = op.old

check_local = FunctionType(transport.check_local.__code__, dict(transport.check_local.__globals__, op=op))
_unpack = FunctionType(transport.unpack.__code__, dict(transport.unpack.__globals__, op=op), argdefs=transport.unpack.__defaults__)


def payload(c, *, collect=None):
    if collect is not None:
        raise ValueError('No separate retrieval or restarted model under this claim.')
    root = Path(c['repository'])
    sources = {n: op.INIT if n == 'models/__init__.py' else (root/n).read_bytes() for n in op.REMOTE_NAMES}
    if any(hashlib.sha256(b).hexdigest() != c['remote_source_sha256'][n] for n, b in sources.items()) or sum(map(len, sources.values())) > 2*op.MiB:
        raise ValueError('Exact source/overlay extent changed.')
    order = [*op.FROZEN_NAMES, 'models/predictor_forward_preserving_records_v1.py', 'models/predictor_forward_preserving_operator_v1.py']
    body = "import base64,json,sys,types\npackage=types.ModuleType('models');package.__path__=[];sys.modules['models']=package\n"
    for n in order:
        name = n[:-3].replace('/', '.')
        data = base64.b64encode(sources[n]).decode()
        body += f"m=types.ModuleType({name!r});m.__file__={n!r};sys.modules[{name!r}]=m;setattr(package,{name.split('.')[-1]!r},m);exec(compile(base64.b64decode({data!r}),{n!r},'exec'),m.__dict__)\n"
    control = base64.b64encode(old.json_bytes(c)).decode()
    src = base64.b64encode(old.json_bytes({n: base64.b64encode(b).decode() for n, b in sources.items()})).decode()
    body += f"sys.exit(m.main({control!r},{src!r}))\n"
    if len(body.encode()) > 2*op.MiB:
        raise ValueError('Bounded stdin bootstrap exceeded.')
    return body.encode()


def unpack(response, destination, c, sha, *, collect=False):
    if collect:
        raise ValueError('No separate retrieval under this grant.')
    p = Path(response)
    if not p.is_file() or p.stat().st_size == 0:
        raise ValueError('No collection response; preserve the session and report it. Never rerun the model.')
    return _unpack(response, destination, c, sha, collect=False)


def run(folder, *, download_only=False, popen=transport.subprocess.Popen):
    # The frozen runner refuses unarmed/copied/consumed state before SSH.
    def authorize(a, sha):
        c = json.loads((Path(folder)/'CONTROL.json').read_text())
        return op.validate_authorization(a, sha, c)
    view = type('BoundOperator', (), dict(op.__dict__))
    view.validate_authorization = staticmethod(authorize)
    runner = FunctionType(transport.run.__code__, dict(transport.run.__globals__, op=view,
                          check_local=check_local, payload=payload, unpack=unpack), argdefs=transport.run.__defaults__)
    return runner(folder, download_only=download_only, popen=popen)

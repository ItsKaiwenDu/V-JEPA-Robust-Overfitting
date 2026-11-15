"""New unarmed client; original one-SSH code and verified private reconstruction."""
import base64
import hashlib
from pathlib import Path
from types import FunctionType
from models import predictor_eot8_record_client_v2 as transport
from models import predictor_pgd_target_operator_v1 as op
from models import private_collection_v1 as private

TRANSPORT_SHA=private.FROZEN_SHA
_check=FunctionType(transport.check_local.__code__,dict(transport.check_local.__globals__,op=op))


def check_local(folder):
    if op.old.digest(transport.__file__)!=TRANSPORT_SHA:raise ValueError('Frozen single-SSH transport changed.')
    return _check(folder)


def payload(c,*,collect=None):
    if collect is not None:raise ValueError('No separate retrieval or model restart.')
    root=Path(c['repository']);sources={n:op.VIRTUAL[n] if n in op.VIRTUAL else (root/n).read_bytes() for n in op.REMOTE_NAMES}
    if any(hashlib.sha256(b).hexdigest()!=c['remote_source_sha256'][n] for n,b in sources.items()) or sum(map(len,sources.values()))>2*op.MiB:
        raise ValueError('Exact overlay extent/source changed.')
    body="import base64,json,sys,types\npackage=types.ModuleType('models');package.__path__=[];sys.modules['models']=package\n"
    for n in op.BOOT_NAMES:
        name=n[:-3].replace('/','.');data=base64.b64encode(sources[n]).decode()
        body+=f"m=types.ModuleType({name!r});m.__file__={n!r};sys.modules[{name!r}]=m;setattr(package,{name.split('.')[-1]!r},m);exec(compile(base64.b64decode({data!r}),{n!r},'exec'),m.__dict__)\n"
    control=base64.b64encode(op.old.json_bytes(c)).decode();src=base64.b64encode(op.old.json_bytes({n:base64.b64encode(b).decode() for n,b in sources.items()})).decode()
    body+=f"sys.exit(m.main({control!r},{src!r}))\n"
    if len(body.encode())>2*op.MiB:raise ValueError('Bootstrap exceeds 2 MiB.')
    return body.encode()


def unpack(response,destination,c,sha,*,collect=False):
    if collect:raise ValueError('Only same-session collection is authorized.')
    return private.unpack(response,destination,c,sha,response_sha256=op.old.digest(response))


def run(folder,*,download_only=False,popen=transport.subprocess.Popen):
    # The old run checks AUTHORIZATION before source scans, claims, session or SSH.
    def auth(a,sha):
        c=__import__('json').loads((Path(folder)/'CONTROL.json').read_bytes());op.validate_authorization(a,sha,c)
    view=type('BoundPGDOperator',(),dict(op.__dict__));view.validate_authorization=staticmethod(auth)
    runner=FunctionType(transport.run.__code__,dict(transport.run.__globals__,op=view,check_local=check_local,payload=payload,unpack=unpack))
    return runner(folder,download_only=download_only,popen=popen)

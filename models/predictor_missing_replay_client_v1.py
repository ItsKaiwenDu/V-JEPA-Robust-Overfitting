"""One consumed claim and one SSH connection for work plus evidence collection."""
import ast
import base64
import hashlib
import json
from pathlib import Path
from types import FunctionType
from models import predictor_eot8_record_client_v2 as transport
from models import predictor_missing_replay_operator_v1 as op
old=op.old
TRANSPORT_SHA='d0d12ef34b731906b12ac964416eba3a48c1d8a2e80cd412bb3054f18c69aede'
_base_check=FunctionType(transport.check_local.__code__,dict(transport.check_local.__globals__,op=op))
_unpack=FunctionType(transport.unpack.__code__,dict(transport.unpack.__globals__,op=op),argdefs=transport.unpack.__defaults__)


def check_local(folder):
    c,sha=_base_check(folder)
    path=Path(c['repository'])/c['reference_archive_path']
    if path.is_symlink() or path.stat().st_size!=op.REFERENCE_BYTES or old.digest(path)!=op.REFERENCE_SHA:
        raise ValueError('Immutable reference archive changed.')
    return c,sha


def payload(c,*,collect=None):
    if collect is not None:raise ValueError('No separate retrieval or model restart under this claim.')
    root=Path(c['repository']);sources={n:op.INIT if n=='models/__init__.py' else (root/n).read_bytes() for n in op.REMOTE_NAMES}
    if any(hashlib.sha256(b).hexdigest()!=c['remote_source_sha256'][n] for n,b in sources.items()) or sum(map(len,sources.values()))>2*op.MiB:
        raise ValueError('Exact single-replay overlay extent changed.')
    body="import base64,json,sys,types\npackage=types.ModuleType('models');package.__path__=[];sys.modules['models']=package\n"
    for n in op.BOOT_NAMES:
        name=n[:-3].replace('/','.');data=base64.b64encode(sources[n]).decode()
        body+=f"m=types.ModuleType({name!r});m.__file__={n!r};sys.modules[{name!r}]=m;setattr(package,{name.split('.')[-1]!r},m);exec(compile(base64.b64decode({data!r}),{n!r},'exec'),m.__dict__)\n"
    control=base64.b64encode(old.json_bytes(c)).decode();src=base64.b64encode(old.json_bytes({n:base64.b64encode(b).decode() for n,b in sources.items()})).decode()
    body+=f"sys.exit(m.main({control!r},{src!r}))\n"
    if len(body.encode())>2*op.MiB:raise ValueError('Exact bootstrap exceeds 2 MiB.')
    return body.encode()


def unpack(response,destination,c,sha,*,collect=False):
    if collect:raise ValueError('Only original same-session collection is allowed.')
    if not Path(response).is_file() or Path(response).stat().st_size==0:
        raise ValueError('No collection response; preserve partial evidence and never rerun the model.')
    return _unpack(response,destination,c,sha,collect=False)


def transport_ast():
    path=Path(transport.__file__)
    if old.digest(path)!=TRANSPORT_SHA:raise ValueError('Frozen one-connection transport changed.')
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    count={'archives':0,'deadline':0}
    class Adapt(ast.NodeTransformer):
        def visit_With(self,n):
            text=ast.unparse(n.items[0].context_expr)
            if text=="(Path(c['repository']) / c['archive_path']).open('rb')":
                count['archives']+=1
                replacement=ast.parse("for archive_key in ('archive_path', 'reference_archive_path'):\n    with (Path(c['repository'])/c[archive_key]).open('rb') as incoming:\n        pass").body[0]
                replacement.body[0].body=n.body
                return ast.copy_location(replacement,n)
            return self.generic_visit(n)
        def visit_Constant(self,n):
            if n.value==3000:
                count['deadline']+=1
                return ast.copy_location(ast.Constant(value=1500),n)
            return n
    node=Adapt().visit(node)
    if count!={'archives':1,'deadline':1}:raise ValueError('Exact two-archive/25-minute transport adaptation required.')
    return ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))


def run(folder,*,download_only=False,popen=transport.subprocess.Popen):
    def authorize(a,sha):
        c=json.loads((Path(folder)/'CONTROL.json').read_text());op.validate_authorization(a,sha,c)
    view=type('BoundSingleReplayOperator',(),dict(op.__dict__))
    view.validate_authorization=staticmethod(authorize)
    namespace=dict(transport.run.__globals__,op=view,check_local=check_local,payload=payload,unpack=unpack)
    exec(compile(transport_ast(),str(Path(__file__)),'exec'),namespace)
    return namespace['run'](folder,download_only=download_only,popen=popen)

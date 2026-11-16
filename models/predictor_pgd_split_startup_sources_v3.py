"""Cold source builders for packet imports and bounded parent failure diagnostics.

No loader, worker, process, clock, proof or network operation executes on import.
Generated supervisors retain complete frozen prefixes; scientific bytes stay exact.
"""
import ast
import hashlib
from models import predictor_pgd_split_outer_sources_v1 as prior

BOOTSTRAP_SHA='4e977468a4e9fca4bda595983d0345bb6ed8c9f31ea1bcb7e8a0553b17a9b1f4'
PACKET_IMPORTS={
 'datasets':('code/datasets/__init__.py',{'bytes':67,'sha256':'d81c61d6627e4e28075736950978c3cca23843d93fb213454458bd32f5357cf1'},True),
 'datasets.prepared':('code/datasets/prepared.py',{'bytes':3784,'sha256':'531dcb4e4e69e3a59890e020940a6a8d0459d47100154730eabc6d267c5e2bb5'},False)}

BOOTSTRAP_EXTENSION='''
# BEGIN PACKET IMPORT EXTENSION V3: fixed packet files; no ambient project path.
_StartupBaseImporter=PinnedImporter
_StartupBaseRequireNamespace=require_namespace

class PinnedImporter(_StartupBaseImporter):
    def __init__(self,root,binding,c,policy):
        super().__init__(root,binding,c,policy)
        packet=parsed(policy['scratch_contract_text'])['packet']['files']
        self.packet_imports={}
        self.packet_import_policy='pgd-split-existing-packet-imports-v3'
        for name,(path,descriptor,package) in _STARTUP_PACKET_IMPORTS.items():
            if packet.get(path.removeprefix('code/'))!=descriptor or name in self.files:
                raise ValueError('Exact existing packet dependency descriptor required.')
            self.packet_imports[name]=(path,dict(descriptor),package)
        # Do not read packet-backed files before the original extraction phase.
    def _packet_source(self,name):
        path,descriptor,package=self.packet_imports[name]
        data=private_read(self.root,self.binding,path,descriptor['bytes'])
        if len(data)!=descriptor['bytes'] or digest(data)!=descriptor['sha256']:
            raise ValueError('Packet dependency hash/extent changed: '+name)
        return path,data,package
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]!='datasets':return super().find_spec(fullname,path,target)
        if fullname not in self.packet_imports:raise ModuleNotFoundError('Unpinned project import refused: '+fullname)
        if fullname in self.loaded or target is not None:raise ImportError('Project module reload refused: '+fullname)
        if fullname!='datasets':
            parent=sys.modules.get('datasets')
            _startup_check_module(self,'datasets',parent)
            expected=[str(self.root/'code/datasets')]
            if path!=expected:raise ImportError('Exact packet package search path required.')
        elif path is not None:raise ImportError('Top-level packet package cannot use an ambient path.')
        self.files[fullname]=self._packet_source(fullname)
        relative,data,package=self.files[fullname]
        return importlib.util.spec_from_loader(fullname,self,origin=str(self.root/relative),is_package=package)
    def exec_module(self,module):
        name=module.__name__
        if name in self.packet_imports:
            # Recheck held/named bytes at execution, after find_spec as well.
            current=self._packet_source(name)
            if name not in self.files or current!=self.files[name]:raise ImportError('Packet source changed after resolution.')
        return super().exec_module(module)
    def install(self):
        if any(n.split('.')[0]=='datasets' for n in sys.modules):raise ValueError('Fresh pinned datasets namespace required.')
        return super().install()

def _startup_check_module(finder,name,module):
    if module is None or name not in finder.files or name not in finder.loaded:
        raise ImportError('One already-loaded pinned project parent required: '+name)
    path,data,package=finder.files[name];expected=str(finder.root/path)
    if (getattr(module,'__file__',None)!=expected or getattr(module,'__loader__',None) is not finder
            or getattr(getattr(module,'__spec__',None),'loader',None) is not finder
            or getattr(getattr(module,'__spec__',None),'origin',None)!=expected
            or package and getattr(module,'__path__',None)!=[str((finder.root/path).parent)]):
        raise ImportError('Project module origin/loader/package path changed: '+name)

def require_namespace(root,binding,csha):
    finder=_StartupBaseRequireNamespace(root,binding,csha)
    # Bootstrap definitions exist in both the initial stdin and pinned module namespaces.
    # Bind exact policy descriptors, not Python class-object identity across copies.
    if (getattr(finder,'packet_import_policy',None)!='pgd-split-existing-packet-imports-v3'
            or getattr(finder,'packet_imports',None)!=_STARTUP_PACKET_IMPORTS):
        raise ValueError('Reviewed packet-aware importer required.')
    for name,module in sys.modules.items():
        if name.split('.')[0] in ('models','attacks','datasets'):_startup_check_module(finder,name,module)
    return finder
# END PACKET IMPORT EXTENSION V3
'''

COORDINATOR_EXTENSION='''
# BEGIN BOUNDED STARTUP ERROR DIAGNOSTICS V3; stop/ownership clocks unchanged.
def _startup_error_record(error):
    frames=[];tb=error.__traceback__
    for _ in range(8):
        if tb is None:break
        code=tb.tb_frame.f_code
        frames.append({'file':str(code.co_filename)[-256:],'function':str(code.co_name)[:96],'line':tb.tb_lineno})
        tb=tb.tb_next
    notes=[str(value)[:256] for value in getattr(error,'__notes__',())[:4]]
    record={'kind':'pgd-split-parent-error-diagnostic-v3','error_type':type(error).__name__[:96],
        'message':str(error)[:1024],'frames':frames,'frame_limit':8,'frames_truncated':tb is not None,'notes':notes,
        'diagnostic_is_not_child_success':True,'scientific_qualified':False,'training_admitted':False}
    if len(boot.encoded(record))>16*1024:
        record.update(message='Diagnostic text exceeded bound; type/first frame preserved.',frames=frames[:1],notes=[],frames_truncated=True)
    if len(boot.encoded(record))>16*1024:raise ValueError('Bounded failure diagnostic required.')
    return record

_StartupOuterDriver=OuterDriver
class OuterDriver(_StartupOuterDriver):
    def launch(self,role,plan):
        try:return super().launch(role,plan)
        except BaseException as error:
            # Observation only after the original owned cleanup; no signal/wait.
            status='unconfirmed'
            try:
                if self.outer_child is not None:status=self.outer_child.poll()
            except BaseException:pass
            error.add_note('Owned '+role+' child exit status: '+str(status)[:64]+
                '; original child log retained at capture/logs/'+role+'.log')
            raise

_StartupOuterCoordinator=OuterCoordinator
class OuterCoordinator(_StartupOuterCoordinator):
    def run_work(self):
        try:return super().run_work()
        except BaseException as error:
            if self.claimed and type(self.state) is dict:
                self.state['parent_error_diagnostic']=_startup_error_record(error)
                try:self.driver.save(self.state,replace=True)
                except BaseException as extra:error.add_note('Bounded parent diagnostic could not be saved: '+type(extra).__name__)
            raise
# END BOUNDED STARTUP ERROR DIAGNOSTICS V3
'''


def build_bootstrap(original):
    if type(original) is not bytes or hashlib.sha256(original).hexdigest()!=BOOTSTRAP_SHA:
        raise ValueError('Exact frozen bootstrap source required.')
    extension=ast.parse('_STARTUP_PACKET_IMPORTS='+repr(PACKET_IMPORTS)+'\n'+BOOTSTRAP_EXTENSION)
    data=original+b'\n'+ast.unparse(ast.fix_missing_locations(extension)).encode()+b'\n'
    compile(data,'<UNEXECUTED-PACKET-IMPORT-BOOTSTRAP-V3>','exec')
    return data


def build_coordinator(original):
    prefix,edits=prior.build(original)
    data=prefix+b'\n'+ast.unparse(ast.parse(COORDINATOR_EXTENSION)).encode()+b'\n'
    compile(data,'<UNEXECUTED-STARTUP-DIAGNOSTIC-COORDINATOR-V3>','exec')
    return data,dict(edits,parent_error_wrappers=2)


def active_import_nodes(tree):
    """Conservative source traversal excluding only literal if-False bodies.

    Includes function/deferred imports. Does not evaluate scientific expressions.
    """
    found=[];excluded=[];dynamic=[]
    def visit(node):
        if isinstance(node,ast.If) and isinstance(node.test,ast.Constant) and node.test.value is False:
            excluded.extend(n for n in ast.walk(node) if isinstance(n,(ast.Import,ast.ImportFrom)))
            for n in node.orelse:visit(n)
            return
        if isinstance(node,(ast.Import,ast.ImportFrom)):found.append(node)
        if isinstance(node,ast.Call) and ast.unparse(node.func) in ('__import__','importlib.import_module'):dynamic.append(node)
        for child in ast.iter_child_nodes(node):visit(child)
    visit(tree)
    return found,excluded,dynamic

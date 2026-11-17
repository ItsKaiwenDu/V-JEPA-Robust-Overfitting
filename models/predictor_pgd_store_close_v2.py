"""Cold repair recipe for the exact frozen byte store; no execution or target API.

Move the owned publication-fd close before recording its final metadata. Keep all
old guards and add checks of logical identity plus every other committed member.
The returned module is a candidate only, not a remote handoff or authorization.
"""
import ast
import hashlib

FROZEN_SHA='2e48a86724bf8d8e49981d7b4dd21875a5f31010e54a558488d16034fb93bb54'
OLD_CHECK="""        for name,want in self.written.items():
            if report['files'].get(name)!=want:raise ValueError('Previously committed result metadata changed: '+name)
"""
NEW_CHECK="""        for name,want in self.written.items():
            got=report['files'].get(name)
            if got!=want:
                error=ValueError('Previously committed result metadata changed: '+name)
                fields=sorted(k for k in ('bytes','allocated_bytes','identity','mtime_ns') if got is None or got.get(k)!=want.get(k))
                note={'fields':fields,'expected':want,'observed':got}
                error.add_note('Bounded result metadata delta: '+json.dumps(note,sort_keys=True,separators=(',',':'),allow_nan=False)[:2048])
                raise error
"""
OLD_COMMIT="""            os.fsync(parent_fd)
            self.written[name]=self.inventory.scan()['files'][name]
"""
NEW_COMMIT="""            os.fsync(parent_fd)
            # Retire the owned descriptor once, even if close itself raises.
            publication_fd,fd=fd,None
            os.close(publication_fd)
            if identity(private(parent,directory=True,device=self.inventory.device))[:4]!=identity(pst)[:4]:
                raise ValueError('Result parent changed after owned publication close.')
            named=os.stat(dest.name,dir_fd=parent_fd,follow_symlinks=False)
            if identity(named)!=identity(held) or named.st_size!=held.st_size or named.st_mtime_ns!=held.st_mtime_ns:
                raise ValueError('Published result identity/extent/mtime changed after owned close.')
            settled=self.inventory.scan()
            for key,want in self.written.items():
                if key!=name and settled['files'].get(key)!=want:
                    raise ValueError('Other committed result changed at publication close: '+key)
            committed=settled['files'].get(name)
            if (committed is None or committed['identity']!=identity(held)
                    or committed['bytes']!=held.st_size or committed['mtime_ns']!=held.st_mtime_ns):
                raise ValueError('Published result identity/extent/mtime changed during closed-file scan.')
            self.written[name]=committed
"""


def build(frozen):
    if type(frozen) is not bytes or hashlib.sha256(frozen).hexdigest()!=FROZEN_SHA:
        raise ValueError('Exact consumed byte-store source required; never patch an unknown revision.')
    text=frozen.decode()
    if text.count('import hashlib\n')!=1 or text.count(OLD_CHECK)!=1 or text.count(OLD_COMMIT)!=1:
        raise ValueError('Exact reviewed publication/check sites required.')
    candidate=text.replace('import hashlib\n','import hashlib\nimport json\n',1).replace(OLD_CHECK,NEW_CHECK,1).replace(OLD_COMMIT,NEW_COMMIT,1).encode()
    compile(candidate,'<UNEXECUTED-STORE-CLOSE-V2>','exec')
    restored=candidate.decode().replace(NEW_COMMIT,OLD_COMMIT,1).replace(NEW_CHECK,OLD_CHECK,1).replace('import hashlib\nimport json\n','import hashlib\n',1).encode()
    if restored!=frozen:raise ValueError('Whole-module exact inverse failed.')
    oldtree,newtree=ast.parse(frozen),ast.parse(candidate)
    oldstore=next(n for n in oldtree.body if isinstance(n,ast.ClassDef) and n.name=='Store')
    newstore=next(n for n in newtree.body if isinstance(n,ast.ClassDef) and n.name=='Store')
    unchanged=[]
    for node in oldtree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name!='Store':
            other=next(v for v in newtree.body if type(v) is type(node) and v.name==node.name)
            if ast.dump(node,include_attributes=False)!=ast.dump(other,include_attributes=False):raise ValueError('Unrelated store/inventory code changed.')
            unchanged.append(node.name)
    for node in oldstore.body:
        if isinstance(node,ast.FunctionDef) and node.name not in ('check','write_bytes'):
            other=next(v for v in newstore.body if isinstance(v,ast.FunctionDef) and v.name==node.name)
            if ast.dump(node,include_attributes=False)!=ast.dump(other,include_attributes=False):raise ValueError('Unrelated store operation changed.')
            unchanged.append('Store.'+node.name)
    return candidate,{'kind':'pgd-store-close-candidate-v2','frozen_sha256':FROZEN_SHA,'candidate_sha256':hashlib.sha256(candidate).hexdigest(),
        'candidate_bytes':len(candidate),'changed_methods':['Store.check','Store.write_bytes'],'unchanged_definitions':unchanged,
        'whole_module_exact_inverse_verified':True,'full_metadata_predicate_unchanged':True,'allocation_guard_removed':False,
        'descriptor_retired_before_commit_snapshot':True,'other_committed_members_rechecked':True,'logical_identity_checked_after_close_and_scan':True,
        'diagnostic_note_payload_character_cap':2048,'live_XFS_failure_field_still_unproved':True,
        'target_handoff_ready':False,'authorized':False,'scientific_qualified':False,'training_admitted':False}

"""Cold derivation of a source-supported metadata policy; no execution entry."""
import ast
import hashlib
import json
from pathlib import Path

INPUTS='configs/predictor_pgd_engine_materialization_inputs_v1.json'
INPUTS_SHA='395c4c496d40ca99d9ef3833aff8363b2a30b36903572159ac2c4c782ae0624a'
OLD='models/predictor_pgd_split_backend_v3.py'
FROZEN='2673d507c18b1b09a0b269d8740252b217a591b11def52bcf9f22cdca7c5c2c8'
NEW='models/predictor_pgd_split_backend_v4.py'
OLD_POLICY='pgd-pytorch28-node-generic-wrapper-typed-producer-metadata-v3'
POLICY='pgd-pytorch28-materialized-wrapper-typed-producer-metadata-v4'


def digest(raw):return hashlib.sha256(raw).hexdigest()

def inputs(repository):
    root=Path(repository);raw=(root/INPUTS).read_bytes()
    if digest(raw)!=INPUTS_SHA:raise ValueError('Exact local materialization source contract required.')
    contract=json.loads(raw)
    for name,d in contract['files'].items():
        value=(root/name).read_bytes()
        if len(value)!=d['bytes'] or digest(value)!=d['sha256']:raise ValueError('Pinned metadata/source input changed: '+name)
    return contract


def build(repository):
    root=Path(repository);inputs(root);original=(root/OLD).read_bytes()
    if digest(original)!=FROZEN:raise ValueError('Exact frozen v3 reader required.')
    before=('POLICY='+repr(OLD_POLICY)).encode();after=('POLICY='+repr(POLICY)).encode()
    if original.count(before)!=1:raise ValueError('One frozen policy literal required.')
    raw=original.replace(before,after,1);tree=ast.parse(raw);view=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_View');fn=next(n for n in view.body if isinstance(n,ast.FunctionDef) and n.name=='bind')
    lines=raw.splitlines(keepends=True);start=sum(map(len,lines[:fn.lineno-1]));end=sum(map(len,lines[:fn.end_lineno]));old=raw[start:end].decode();new=old
    doc_end=new.index('        node=self.rows[index]')
    new=new[:new.index('        """')]+'''        """Recorded default or adjacent-node materialization signatures.

        PyTorch2.8 collation may propagate node fields into its preceding
        FUNCTION wrapper. The reader only checks existing raw fields; it does
        not copy tags, observe native storage order or authenticate execution.
        All typed producer, parent, phase and kernel checks remain required.
        """
'''+new[doc_end:]
    changes=[]
    def edit(a,b):
        nonlocal new
        if new.count(a)!=1:raise ValueError('One exact association edit required.')
        new=new.replace(a,b,1);changes.append({'before':a,'after':b})
    edit("""        wrappers=[i for i in self.engines[(ENGINE+node['name'],0,-1)]
            if self.rows[i]['thread']==node['thread'] and self.rows[i]['device_index']==node['device_index']
            and self.rows[i]['start_ns']<=node['start_ns']<node['end_ns']<=self.rows[i]['end_ns']]
""","""        # Count every enclosing name/thread/process match, even malformed tags;
        # a second wrapper must not disappear by choosing a signature first.
        wrappers=sorted(i for i in self.ops
            if self.rows[i]['name']==ENGINE+node['name']
            and self.rows[i]['thread']==node['thread'] and self.rows[i]['device_index']==node['device_index']
            and self.rows[i]['start_ns']<=node['start_ns']<node['end_ns']<=self.rows[i]['end_ns'])
        tags=(engine['fwd_thread'],engine['sequence_nr']);successor=None
        if tags==(0,-1):
            signature='pytorch-2.8.0-generic-function-wrapper-defaults-v1'
        else:
            following=sorted(i for i in self.cpu if i>engine_index
                and self.rows[i]['thread']==engine['thread']
                and self.rows[i]['device_index']==engine['device_index'])
            successor=following[0] if following else None
            if tags!=key or successor!=index or not engine['start_ns']<node['start_ns']:
                raise ValueError('Source-shaped wrapper tags and recorded same-thread adjacency required.')
            signature='pytorch-2.8.0-adjacent-node-tag-materialization-v1'
""")
    edit("(engine['thread'],engine['fwd_thread'],engine['sequence_nr'])!=(node['thread'],0,-1)","engine['thread']!=node['thread']")
    edit("'engine_signature_policy':'pytorch-2.8.0-generic-function-wrapper-defaults-v1'","'engine_signature_policy':signature")
    edit("        self.anchors.append(anchor)","""        if successor is not None:
            anchor['recorded_next_same_thread_CPU_index']=successor
            anchor['native_thread_local_storage_order_observed']=False
        self.anchors.append(anchor)""")
    candidate=raw[:start]+new.encode()+raw[end:]
    inverse=candidate[:start]+old.encode()+candidate[start+len(new.encode()):]
    if inverse.replace(after,before,1)!=original:raise ValueError('Complete byte inverse required.')
    def normalize(t):
        for n in t.body:
            if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='POLICY' for x in n.targets):n.value=ast.Constant(OLD_POLICY)
            if isinstance(n,ast.ClassDef) and n.name=='_View':n.body=[f for f in n.body if not isinstance(f,ast.FunctionDef) or f.name!='bind']
        return ast.dump(t,include_attributes=False)
    if normalize(ast.parse(original))!=normalize(ast.parse(candidate)):raise ValueError('All other reader/projection/phase/kernel/IO/cap AST must remain exact.')
    compile(candidate,'<UNEXECUTED-MATERIALIZATION-METADATA-READER>','exec')
    return candidate,{'kind':'engine-materialization-v1-cold-derivation','frozen_source':OLD,'frozen_sha256':FROZEN,'candidate_source':NEW,'candidate_sha256':digest(candidate),'candidate_bytes':len(candidate),'policy_edit':{'before':before.decode(),'after':after.decode()},'method_edit':{'start':start,'old_end':end,'new_end':start+len(new.encode()),'before':old,'after':new},'association_edits':changes,'all_other_AST_exact':True,'complete_byte_inverse':True,'raw_rows_or_fields_rewritten':False,'target_integrated':False,'worker_completion_or_scientific_admission':False}

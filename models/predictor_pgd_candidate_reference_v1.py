"""Read-only native reference bytes and metadata; never decodes numerical values."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
from models import predictor_pgd_component_capsule_v1 as capsule

RETURNS=('output','dq','dk','dv')
SHAPE=(1,12,2048,32)
MAX_FILE=4*2**20
MAX_ARCHIVE=16*2**20
SPEC_BYTES=None
SPEC_SHA=None


def digest(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def require(value,message):
    if not value:raise ValueError(message)

def logical_npy(raw):
    require(type(raw) is bytes and 10<len(raw)<=MAX_FILE and raw[:8]==b'\x93NUMPY\x01\x00','Pinned native NPY v1 envelope required.')
    end=10+int.from_bytes(raw[8:10],'little')
    require(10<end<=4096 and end%16==0 and raw[end-1:end]==b'\n','Bounded complete NPY header required.')
    tree=ast.parse(raw[10:end].decode('ascii').strip(),mode='eval')
    require(isinstance(tree.body,ast.Dict) and len(tree.body.keys)==3,'Exact native NPY fields required.')
    keys=[ast.literal_eval(k) for k in tree.body.keys];require(set(keys)=={'descr','fortran_order','shape'} and len(set(keys))==3,'Unique native NPY fields required.')
    h=ast.literal_eval(tree)
    require(h['descr']=='<f4' and h['fortran_order'] is False and type(h['shape']) is tuple and h['shape']==SHAPE
        and all(type(n) is int for n in h['shape']) and len(raw)-end==3145728,'Original native FP32 logical geometry required.')
    return digest(raw[end:])


def describe_members(members,descriptors):
    require(type(members) is dict and set(members)==set(descriptors) and len(members)==18,'Exact native reference inventory required.')
    files={}
    for name,d in descriptors.items():
        raw=members[name];require(type(raw) is bytes and 0<len(raw)<=MAX_FILE and len(raw)==d['bytes'] and digest(raw)==d['sha256'],'Pinned native reference member changed: '+name)
        files[name]={'bytes':len(raw),'sha256':digest(raw)}
    j=lambda name:json.loads(members[name])
    w=j('native/worker.json');marker=j('native/complete.json');res=j('native/resource.json');parent=j('native/parent.json');own=j('native/ownership.json')
    boundary=j('source/boundary.json');manifest=j('source/capsule-manifest.json');pgd=j('reviews/pgd.json');target=j('reviews/component.json');values=j('reviews/values.json');metadata=j('reviews/metadata.json')
    require(w['kind']=='pgd-capsule-component-worker-v1' and w['role']=='native' and w['complete'] and marker['complete']
        and marker['binding']==w['binding']==res['binding']==own['binding'] and w['binding']['probe_index']==0,'Completed native worker/source identity required.')
    require(marker['worker']['sha256']==files['native/worker.json']['sha256']==res['worker_sha256']
        and marker['resource']['sha256']==files['native/resource.json']['sha256'],'Native resource/complete descriptors changed.')
    require(parent['pid']==res['pid']==w['pid']==own['pid'] and parent['exit_status']==0 and parent['parent_observation_passed']
        and own['route_complete'] and own['phase']=='closed','Accepted native route/parent result required.')
    require(0<parent['peak_observed_RSS_bytes']<=4*2**30 and 0<parent['RSS_samples']<=2400
        and 0<res['peak_allocator_bytes']<=7*2**30 and 0<=w['wall_seconds']<=res['wall_seconds']<=parent['wall_seconds']<=120,'Original native resource bounds required.')
    require(w['trace']['sha256']==files['native/events.json']['sha256'] and w['backend']['sha256']==files['native/backend.json']['sha256']
        and w['ownership']['sha256']==files['native/ownership.json']['sha256'],'Native trace/backend/ownership joins changed.')
    require(pgd['audit_result']=='PASS' and pgd['numerical_scoped_saved_passed'] and pgd['gradient_objectives_completed']==8 and pgd['same_input_pairs_passed']==4,'Accepted original PGD cohort required without rerunning it.')
    require(target['audit_result']=='PASS' and target['transport_verified'] and target['native_worker_metadata_complete'] and not target['candidate_complete'], 'Native acceptance and historical candidate failure must remain separate.')
    require(values['review_result']=='PASS' and values['native_exact_historical_boundary_returns'] and values['old_failed_worker_still_failed']
        and metadata['result']=='PASS' and metadata['native_metadata_result_unchanged_except_policy'],'Reviewed native values and v4 metadata required.')
    require(files['source/boundary.json']['sha256']==w['binding']['boundary_sha256']==manifest['boundary_sha256']
        and w['source']['capsule_manifest_sha256']==files['source/capsule-manifest.json']['sha256']
        and w['source']['source_control_sha256']==manifest['source_control_sha256']==boundary['control_sha256'],'Historical boundary/capsule lineage changed.')
    logical={}
    for name in RETURNS:
        d=files['native/'+name+'.npy'];require(w['outputs'][name]['sha256']==d['sha256'] and w['outputs'][name]['bytes']==d['bytes'],'Native return descriptor changed.')
        logical[name]=logical_npy(members['native/'+name+'.npy'])
        require(logical[name]==boundary['artifacts'][name]['source']['sha256']
            ==target['eight_component_return_files_preserved']['native'][name]['payload_sha256'],'Native logical values must exactly match the accepted boundary.')
    return {'kind':'pgd-candidate-readonly-native-reference-v1','files':files,'shape':list(SHAPE),'dtype':'<f4',
        'native_worker_sha256':files['native/worker.json']['sha256'],'native_resource_sha256':files['native/resource.json']['sha256'],
        'native_parent_sha256':files['native/parent.json']['sha256'],'native_control_sha256':w['binding']['control_sha256'],'native_pid':w['pid'],
        'source_control_sha256':boundary['control_sha256'],'boundary_sha256':manifest['boundary_sha256'],
        'capsule_manifest_sha256':files['source/capsule-manifest.json']['sha256'],'logical_sha256':logical,
        'accepted_PGD_review_sha256':files['reviews/pgd.json']['sha256'],'historical_candidate_stays_failed':True,
        'new_runtime_or_scientific_admission':False}


def archive(members,spec):
    payload={**members,'MANIFEST.json':encoded(spec)};out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w',format=tarfile.USTAR_FORMAT) as tf:
        for name,raw in sorted(payload.items()):
            info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o600;info.uid=info.gid=info.mtime=0;info.uname=info.gname='';tf.addfile(info,io.BytesIO(raw))
    raw=out.getvalue();require(len(raw)<=MAX_ARCHIVE,'Native reference archive exceeds 16 MiB.')
    return raw


def inspect_archive(raw,spec):
    require(type(raw) is bytes and 0<len(raw)<=MAX_ARCHIVE,'Bounded complete native reference archive required.')
    found={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tf:
        for entry in tf:
            require(entry.name in set(spec['files'])|{'MANIFEST.json'} and entry.name not in found
                and entry.isfile() and not entry.pax_headers and entry.mode==0o600 and entry.uid==entry.gid==entry.mtime==0
                and entry.uname==entry.gname=='','Exact ordinary deterministic native-reference member required.')
            bound=128*1024 if entry.name=='MANIFEST.json' else spec['files'][entry.name]['bytes']
            require(0<entry.size<=bound,'Bounded native-reference member required.');found[entry.name]=tf.extractfile(entry).read(bound+1)
    require(found.pop('MANIFEST.json',None)==encoded(spec) and set(found)==set(spec['files']),'Complete closed native reference required.')
    require(describe_members(found,spec['files'])==spec and archive(found,spec)==raw,'Exact native-reference bytes, headers and padding required.')
    return found


def specification():
    require(type(SPEC_BYTES) is bytes and digest(SPEC_BYTES)==SPEC_SHA,'Native reference source is unbound or changed.')
    return json.loads(SPEC_BYTES)


def expected_description(binding):
    spec=specification();require(type(binding) is dict and set(binding)=={'design_sha256','control_sha256','boundary_sha256','probe_index'}
        and type(binding['probe_index']) is int and binding['probe_index']==1 and binding['boundary_sha256']==spec['boundary_sha256']
        and type(binding['control_sha256']) is str and len(binding['control_sha256'])==64 and all(c in '0123456789abcdef' for c in binding['control_sha256'])
        and binding['control_sha256'] not in (spec['source_control_sha256'],spec['native_control_sha256']), 'Distinct new candidate binding and canonical native boundary required.')
    return {'kind':'pgd-candidate-native-reference-binding-v1','binding':dict(binding),'reference_manifest_sha256':SPEC_SHA,
        'native_worker_sha256':spec['native_worker_sha256'],'native_resource_sha256':spec['native_resource_sha256'],'native_parent_sha256':spec['native_parent_sha256'],
        'native_control_sha256':spec['native_control_sha256'],'source_control_sha256':spec['source_control_sha256'],
        'boundary_sha256':spec['boundary_sha256'],'capsule_manifest_sha256':spec['capsule_manifest_sha256'],'logical_sha256':spec['logical_sha256'],
        'historical_candidate_stays_failed':True,'worker_completion_or_scientific_admission':False}


def verify_description(value,binding):
    require(encoded(value)==encoded(expected_description(binding)),'Exact read-only native reference binding required.')
    return value


class Reference:
    def __init__(self,root):
        self.root=Path(root).absolute();self.spec=specification();self.used=False;self.pid=os.getpid();self.identity=self.directory(self.root);self.check()
    def directory(self,path):
        s=path.lstat();require(path.resolve(strict=True)==path and stat.S_ISDIR(s.st_mode) and s.st_uid==os.getuid()
            and stat.S_IMODE(s.st_mode)==0o700,'Private canonical native-reference directory required.')
        if hasattr(self,'identity'):require(s.st_dev==self.identity[0],'Native reference must stay on one device.')
        return s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode)
    def check(self):
        require(os.getpid()==self.pid and self.directory(self.root)==self.identity,'Native reference root/process changed.')
        allowed=set(self.spec['files'])|{'MANIFEST.json'};directories={'.'}
        for name in allowed:directories.update(str(p) for p in Path(name).parents)
        found=set();members={}
        for current,dirs,names in os.walk(self.root,followlinks=False):
            here=Path(current);require(str(here.relative_to(self.root)) in directories,'Unknown native-reference directory.');self.directory(here)
            for name in dirs:self.directory(here/name)
            for name in names:
                p=here/name;key=str(p.relative_to(self.root));require(key in allowed and key not in found,'Unknown native-reference file.')
                found.add(key);want=SPEC_SHA if key=='MANIFEST.json' else self.spec['files'][key]['sha256'];cap=128*1024 if key=='MANIFEST.json' else MAX_FILE
                data=capsule._read(p,cap,want)
                if key=='MANIFEST.json':require(data==SPEC_BYTES,'Exact native manifest required.')
                else:members[key]=data
        require(found==allowed and describe_members(members,self.spec['files'])==self.spec,'Complete immutable native reference required.')
        self.members=members
    def describe(self,binding):
        require(not self.used,'One native reference binding per fresh candidate namespace.');self.used=True;self.check()
        return expected_description(binding)
    def returns(self):
        require(self.used,'Native reference must bind before value checks.');self.check()
        return {n:self.members['native/'+n+'.npy'] for n in RETURNS}

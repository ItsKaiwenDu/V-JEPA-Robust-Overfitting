"""Whole-case journal, external first-case receipt and complete-panel analysis."""

from contextlib import contextmanager
import fcntl
import math
from pathlib import Path, PurePosixPath
import statistics

from models.checkpointing import CheckpointStore
from models.evaluation_store import json_hash
from models.matched_panel_case import FIRST_CASE, exact, execution_gate, verify_case
from models.matched_panel_plan import plan_digest
from models.pilot_bundle import digest
from models.study_artifacts import read_record, write_record

KIND = 'matched-panel-journal-v1'
BANKS = ('main','audit_0','audit_1')
CONTRASTS = {'B-A':('A','B'), 'D-C':('C','D'), 'C-A':('A','C'), 'D-B':('B','D')}


def safe_relative(root, name):
    path=PurePosixPath(name)
    root=Path(root).resolve()
    result=root/name
    if (path.is_absolute() or '..' in path.parts or str(path)!=name or '\\' in name
            or name=='.' or not result.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in [result,*list(result.parents)[:len(path.parts)-1]])):
        raise ValueError('Unsafe journal receipt path.')
    return result


def first_acceptance(plan, execution, receipt, archive_sha256):
    """Called by the offline export reviewer, never by the GPU queue."""
    if receipt['case_id']!=FIRST_CASE or receipt['plan_sha256']!=plan_digest(plan):
        raise ValueError('First-case acceptance cannot refer to another case or plan.')
    return {'kind':'matched-panel-first-case-acceptance-v1','plan_sha256':plan_digest(plan),
            'execution_sha256':json_hash(execution),'case_id':FIRST_CASE,
            'case_receipt_sha256':json_hash(receipt),'archive_sha256':archive_sha256,
            'reviewed_export':True,'research_admitted':False}


class PanelJournal:
    def __init__(self, root, *, plan, execution, allow_fixture=False):
        execution_gate(plan,execution,allow_fixture=allow_fixture)
        self.root=Path(root)
        self.plan,self.execution,self.allow_fixture=plan,execution,allow_fixture
        self.store=CheckpointStore(self.root)
        self.header={'kind':KIND,'plan_sha256':plan_digest(plan),'execution_sha256':json_hash(execution),
                     'case_order':[c['case_id'] for c in plan['cases']]}
        if not (self.root/'journal.json').exists():
            if any((self.root/'cases').glob('*')):
                raise ValueError('Unjournaled case evidence exists; preserve it and use a fresh panel root.')
            write_record(self.root/'journal.json',{**self.header,'completed':{},'first_acceptance':None,
                                                 'research_admitted':False})
        self.read()

    @contextmanager
    def lease(self):
        """One live case supervisor; process exit releases the NFS-compatible lock."""
        with (self.root/'.panel-running.lock').open('a+') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle,fcntl.LOCK_UN)

    def read(self):
        state=read_record(self.root/'journal.json')
        if (any(k not in state or not exact(state[k],v) for k,v in self.header.items())
                or state['research_admitted'] is not False
                or not set(state['completed'])<=set(self.header['case_order'])):
            raise ValueError('Panel journal plan/source identity differs.')
        if state['first_acceptance'] is not None:
            if 'clip-00-bank-0' not in state['completed']:
                raise ValueError('First-case acceptance lacks its committed receipt.')
            pointer=state['completed'][FIRST_CASE]
            path=safe_relative(self.root,pointer['directory'])/'receipt.json'
            receipt=read_record(path,expected_sha256=pointer['receipt_file_sha256'])
            expected=first_acceptance(self.plan,self.execution,receipt,pointer['archive_sha256'])
            if not exact(state['first_acceptance'],expected):
                raise ValueError('Saved first-case acceptance identity changed.')
        return state

    def verify_completed(self,case_id):
        state=self.read()
        pointer=state['completed'][case_id]
        attempt=safe_relative(self.root,pointer['directory'])
        receipt=read_record(attempt/'receipt.json',expected_sha256=pointer['receipt_file_sha256'])
        rebuilt=verify_case(attempt,plan=self.plan,case_id=case_id,execution=self.execution,
                            allow_fixture=self.allow_fixture)
        if not exact(receipt,rebuilt):
            raise ValueError('Completed case receipt differs from retained evidence.')
        archive=safe_relative(self.root,pointer['archive_path'])
        archive_manifest_path=safe_relative(self.root,pointer['archive_manifest_path'])
        for path,key in [(archive,'archive_sha256'),(archive_manifest_path,'archive_manifest_sha256')]:
            if digest(path)!=pointer[key]:
                raise ValueError('Completed case archive/manifest changed.')
        import json
        from models.matched_panel_io import verify_archive
        archive_manifest=json.loads(archive_manifest_path.read_text())
        verify_archive(archive,archive_manifest,pointer['archive_sha256'])
        return receipt

    def commit(self,case_id,attempt,*,archive=None):
        """Read back all artifacts before the atomic journal pointer commits."""
        if case_id not in self.header['case_order']:
            raise ValueError('Unknown panel case.')
        attempt=Path(attempt).resolve()
        try: relative=attempt.relative_to(self.root.resolve()).as_posix()
        except ValueError as error: raise ValueError('Case evidence must live under this panel root.') from error
        safe_relative(self.root,relative)
        receipt=verify_case(attempt,plan=self.plan,case_id=case_id,execution=self.execution,
                            allow_fixture=self.allow_fixture)
        saved=read_record(attempt/'receipt.json')
        if not exact(saved,receipt):
            raise ValueError('Case receipt is missing or differs.')
        archive=Path(archive) if archive is not None else attempt/'evidence.tar.gz'
        manifest_path=Path(str(archive)+'.manifest.json')
        try:
            archive_relative=archive.resolve().relative_to(self.root.resolve()).as_posix()
            manifest_relative=manifest_path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError as error:raise ValueError('Export must live under the panel root.') from error
        safe_relative(self.root,archive_relative);safe_relative(self.root,manifest_relative)
        pointer={'directory':relative,'receipt_file_sha256':digest(attempt/'receipt.json'),
                 'archive_path':archive_relative,'archive_manifest_path':manifest_relative,
                 'archive_sha256':digest(archive),'archive_manifest_sha256':digest(manifest_path)}
        import json
        from models.matched_panel_io import verify_archive
        verify_archive(archive,json.loads(manifest_path.read_text()),
                       pointer['archive_sha256'])
        with self.store._lock(exclusive=True):
            state=self.read()
            if case_id in state['completed']:
                if not exact(state['completed'][case_id],pointer):
                    raise ValueError('Refusing to replace an already committed case.')
                return self.verify_completed(case_id)
            completed=[c for c in self.header['case_order'] if c in state['completed']]
            if completed!=self.header['case_order'][:len(completed)] or case_id!=self.header['case_order'][len(completed)]:
                raise ValueError('Cases must commit in the predeclared order.')
            if case_id!=FIRST_CASE and state['first_acceptance'] is None:
                raise ValueError('Review the first export before any further case.')
            state['completed'][case_id]=pointer
            write_record(self.root/'journal.json',state,replace=True)
        return receipt

    def accept_first(self,acceptance):
        receipt=self.verify_completed(FIRST_CASE)
        state=self.read()
        expected=first_acceptance(self.plan,self.execution,receipt,state['completed'][FIRST_CASE]['archive_sha256'])
        if not exact(acceptance,expected):
            raise ValueError('First-case review belongs to different evidence, execution or plan.')
        with self.store._lock(exclusive=True):
            state=self.read()
            if state['first_acceptance'] is not None and not exact(state['first_acceptance'],acceptance):
                raise ValueError('Refusing to replace first-case acceptance.')
            state['first_acceptance']=acceptance
            write_record(self.root/'journal.json',state,replace=True)

    def progress(self):
        state=self.read()
        for case_id in state['completed']:
            self.verify_completed(case_id)
        if len(state['completed'])>1 and state['first_acceptance'] is None:
            raise ValueError('Later cases lack the required first-case review.')
        return {'kind':'matched-panel-progress-v1','complete':len(state['completed'])==len(self.header['case_order']),
                'completed_case_ids':[c for c in self.header['case_order'] if c in state['completed']],
                'required_cases':len(self.header['case_order']),'scientific_aggregate':None,
                'research_admitted':False}

    def aggregate(self):
        progress=self.progress()
        if not progress['complete']:
            return progress
        receipts={case_id:self.verify_completed(case_id) for case_id in self.header['case_order']}
        for a,b in zip(self.plan['cases'][::2],self.plan['cases'][1::2]):
            ca,cb=(receipts[c['case_id']]['contract'] for c in (a,b))
            for key in ('clip','runtime','weights','clean_sha256','targets_sha256','starts_sha256'):
                if not exact(ca[key],cb[key]):
                    raise ValueError('Replicate banks changed the clip, target, starts or runtime.')
            for name,value in ca['mask_tensors_sha256'].items():
                if not name.startswith('attack_') and value!=cb['mask_tensors_sha256'][name]:
                    raise ValueError('Scoring masks changed between paired attack banks.')
            if any(receipts[a['case_id']]['scores']['A'][bank]['clean']!=
                   receipts[b['case_id']]['scores']['A'][bank]['clean'] for bank in BANKS):
                raise ValueError('Replicate clean scoring baselines differ.')
        return {**progress,'plan_sha256':plan_digest(self.plan),'execution_sha256':json_hash(self.execution),
                'scientific_aggregate':aggregate_scores(self.plan,{c:r['scores'] for c,r in receipts.items()}),
                'case_receipt_sha256':{c:json_hash(r) for c,r in receipts.items()}}


def aggregate_scores(plan,scores):
    """Pure descriptive math; only callers with all verified cases may publish it."""
    expected={c['case_id'] for c in plan['cases']}
    if set(scores)!=expected or len(expected)!=20:
        raise ValueError('No scientific aggregate is allowed for an incomplete panel.')
    per_case={}
    for case in plan['cases']:
        values=scores[case['case_id']]
        if (set(values)!=set('ABCD') or any(set(row)!=set(BANKS) for row in values.values())
                or any(set(row)!= {'clean','adversarial'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<0
                   for v in row.values()) for arm in values.values() for row in arm.values())):
            raise ValueError('Invalid/missing independent scoring values.')
        if any(values[a][b]['clean']!=values['A'][b]['clean'] for a in 'ABCD' for b in BANKS):
            raise ValueError('An arm changed its clean scoring reference.')
        per_case[case['case_id']]={contrast:{bank:{
            'absolute':values[high][bank]['adversarial']-values[low][bank]['adversarial'],
            'relative_percent':None if values[low][bank]['adversarial']==0 else
                100*(values[high][bank]['adversarial']/values[low][bank]['adversarial']-1)}
            for bank in BANKS} for contrast,(low,high) in CONTRASTS.items()}
    per_clip=[]
    for i,clip in enumerate(plan['ordered_clips']):
        ids=[c['case_id'] for c in plan['cases'] if c['clip_index']==i]
        if len(ids)!=2: raise ValueError('Each clip needs both technical bank replicates.')
        contrasts={}
        for contrast in CONTRASTS:
            contrasts[contrast]={}
            for bank in BANKS:
                pair=[per_case[c][contrast][bank] for c in ids]
                contrasts[contrast][bank]={'absolute':math.fsum(p['absolute'] for p in pair)/2,
                    'relative_percent':None if any(p['relative_percent'] is None for p in pair) else
                        math.fsum(p['relative_percent'] for p in pair)/2}
        per_clip.append({'clip_index':i,'video_id':clip['video_id'],'contrasts':contrasts})
    summary={}
    for contrast in CONTRASTS:
        summary[contrast]={}
        for bank in BANKS:
            values=[r['contrasts'][contrast][bank]['absolute'] for r in per_clip]
            relative=[r['contrasts'][contrast][bank]['relative_percent'] for r in per_clip]
            summary[contrast][bank]={'mean':math.fsum(values)/10,'median':statistics.median(values),
                'minimum':min(values),'maximum':max(values),'positive_clips':sum(v>0 for v in values),
                'zero_clips':sum(v==0 for v in values),'negative_clips':sum(v<0 for v in values),
                'relative_percent_mean':None if any(v is None for v in relative) else math.fsum(relative)/10}
    return {'per_case':per_case,'per_clip':per_clip,'summary':summary,'analysis_unit':'clip',
            'clips':10,'technical_banks_per_clip':2,'score_banks_pooled':False,
            'automatic_primary_choice':False,'automatic_convergence_pass':False}

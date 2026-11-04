"""Offline saved-value reassessment; never replaces a v1 acceptance decision.

Identity, bounds, decisions and counts must agree. Error statistics are retained
under each auditor's context; they are not assumed identical across BLAS/thread
contexts. Every saved array is independently recomputed with the original bounds.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import platform

import numpy as np
import torch

from models.attention_fixture import BOOKKEEPING,INPUTS,OUTPUTS,load
from models.attention_saved_evidence import differences
from models.cpu_oracle_packet import verify
from models.cpu_oracle_probe import KIND as PRODUCER_KIND
from models.cuda_profile import tensor_signature
from models.oracle_trace import STAGES,audit_observation,load_arrays,safe_agreement
from models.pilot_bundle import digest,safe_name
from models.study_artifacts import read_record,write_record

KIND = 'cpu-oracle-offline-reassessment-v2'
METRICS = ('max_absolute_error','max_tolerance_ratio','relative_l2_error')
FIELDS = {'passed','failed_elements','shape','rtol','atol',*METRICS}
CONTROL_SHA = '8a42039c95a2592becfbef19f59afdaa5692b8e2125f08f6ede864f19bfe4a78'


def decision(entry):
    if not isinstance(entry,dict) or any(entry.get(k)!=v for k,v in BOOKKEEPING.items()) or type(entry.get('passed')) is not bool:
        raise ValueError('Invalid or changed numeric decision contract.')
    if set(entry)=={'passed','error','rtol','atol'}:
        if entry['passed'] or not isinstance(entry['error'],str) or not 0<len(entry['error'])<=8192:
            raise ValueError('Malformed nonfinite/error refusal.')
        return {'passed':False,'error':entry['error'],**BOOKKEEPING}
    if set(entry)!=FIELDS:
        raise ValueError('Missing or additional numeric receipt fields.')
    shape=entry['shape']
    if (not isinstance(shape,list) or not 1<=len(shape)<=4 or any(type(n) is not int or n<=0 for n in shape)
            or math.prod(shape)>4*2**20 or type(entry['failed_elements']) is not int
            or not 0<=entry['failed_elements']<=math.prod(shape)):
        raise ValueError('Invalid shape or failed-element count.')
    if any(type(entry[k]) not in (float,int) or not math.isfinite(entry[k]) or entry[k]<0 for k in METRICS):
        raise ValueError('Nonfinite or invalid diagnostic statistics.')
    if entry['passed']!=(entry['failed_elements']==0) or entry['passed']!=(entry['max_tolerance_ratio']<=1):
        raise ValueError('Numeric decision contradicts its recorded statistics.')
    return {k:entry[k] for k in ('passed','failed_elements','shape','rtol','atol')}


def compare_numeric_receipts(fresh,stored):
    a,b=decision(fresh),decision(stored)
    return {'decisions_match':a==b,
        'diagnostic_differences':[{ 'field':k,'fresh':fresh.get(k),'stored':stored.get(k)}
            for k in METRICS if fresh.get(k)!=stored.get(k)],
        'fresh_decision':a,'stored_decision':b}


def stage_decisions(report):
    required={'kind','capture_integrity_verified','converted_inputs_exact','chunks','numerical_checks_passed',
        'first_captured_mismatch','stage_comparisons','first_mismatch_establishes_root_cause','bounds'}
    if (set(report)!=required or report['kind']!='cpu-oracle-line-observation-v1'
            or report['capture_integrity_verified'] is not True or report['converted_inputs_exact'] is not True
            or report['bounds']!=BOOKKEEPING or report['first_mismatch_establishes_root_cause'] is not False
            or type(report['chunks']) is not int or report['chunks']!=len(report['stage_comparisons'])
            or not 0<report['chunks']<=64):
        raise ValueError('Malformed stage-audit domain, coverage or bounds.')
    result=[];first=None;keys=[]
    for row in report['stage_comparisons']:
        key=row['key']
        if (set(row)!={'key','comparisons'} or not isinstance(key,list) or len(key)!=4
                or any(type(n) is not int or n<0 for n in key) or key[3]<=key[2]
                or key[3]-key[2]>128 or set(row['comparisons'])!=set(STAGES)):
            raise ValueError('Invalid stage order, keys or comparison fields.')
        keys.append(tuple(key));items={n:decision(row['comparisons'][n]) for n in STAGES}
        result.append({'key':key,'comparisons':items})
        for name in STAGES:
            if first is None and not items[name]['passed']:
                first={'key':key,'stage':name,'comparison':items[name]}
    if len(keys)!=len(set(keys)) or keys!=sorted(keys):
        raise ValueError('Repeated or reordered stage keys.')
    if type(report['numerical_checks_passed']) is not bool or report['numerical_checks_passed']!=(first is None):
        raise ValueError('Stage summary contradicts numeric decisions.')
    stored_first=report['first_captured_mismatch']
    if stored_first is not None:
        if set(stored_first)!={'key','stage','comparison'}:
            raise ValueError('Malformed first-refusal receipt.')
        stored_first={**stored_first,'comparison':decision(stored_first['comparison'])}
    if stored_first!=first:
        raise ValueError('First refusal does not match the ordered stage decisions.')
    return result


def compare_stage_receipts(fresh,stored):
    a,b=stage_decisions(fresh),stage_decisions(stored)
    changed=differences(fresh,stored)
    diagnostic=[item for item in changed if item['path'].split('/')[-1] in METRICS]
    return {'decisions_match':a==b,'diagnostic_difference_count':len(diagnostic),
        'diagnostic_differences':diagnostic,'raw_reports_identical':not changed}


def audit_worker(folder,expected_worker_sha,values,fixture,*,tiny,handoff_sha):
    folder=Path(folder);w=read_record(folder/'worker.json',expected_sha256=expected_worker_sha)
    if (w['kind']!=PRODUCER_KIND or w['capture_complete'] is not True or w['CPU_only'] is not True
            or w['GPU_work_allowed'] is not False or w['tiny_fixture']!=tiny
            or w['fixture']!=fixture or w['handoff_sha256']!=handoff_sha or w['bounds']!=BOOKKEEPING or w['threads'] not in (1,4)
            or w['replay'] not in (0,1) or not 0<w['wall_seconds']<=90
            or type(w['phase_bitwise_equal']) is not bool or type(w['numerical_checks_passed']) is not bool
            or w['local_CPU_rehearsal']!=tiny or set(w['passes'])!={'before','observed','after'}
            or w['input_signatures']!={n:tensor_signature(values[n]) for n in INPUTS}):
        raise ValueError('Incomplete/changed original CPU capture or scope.')
    before,after=w['context_before'],w['context_after']
    for ctx in (before,after):
        if (ctx['CUDA_initialized'] is not False or ctx['threads']!=w['threads']
                or ctx['task_environment']['CUDA_VISIBLE_DEVICES']!=''):
            raise ValueError('Producer CPU/thread context changed.')
    for key in ('host','python','torch','numpy','threads','interop_threads','cpu_capability','task_environment','precision','affinity_read_only'):
        if before[key]!=after[key]:raise ValueError('Producer runtime changed during capture.')
    original=read_record(folder/'stage-comparisons.json',expected_sha256=w['stage_comparisons_sha256'])
    fresh=audit_observation(folder/'trace',w['trace_index_sha256'],{n:values[n].numpy() for n in INPUTS})
    compared=compare_stage_receipts(fresh,original)
    if {k:v for k,v in original.items() if k!='stage_comparisons'}!=w['trace_summary']:
        raise ValueError('Producer stage-summary receipt changed.')
    final={};final_comparisons={};original_final_pass=True;actual={}
    if set(w['checks'])!={'before','observed','after'}:
        raise ValueError('Missing producer final checks.')
    for phase in ('before','observed','after'):
        arrays=load_arrays(folder,w['passes'][phase]['artifact'],{n:list(values['q'].shape) for n in OUTPUTS})
        actual[phase]=arrays
        final[phase]={n:safe_agreement(arrays[n],values['oracle_'+n].numpy()) for n in OUTPUTS}
        if set(w['checks'][phase])!=set(OUTPUTS):raise ValueError('Final output/VJP keys changed.')
        final_comparisons[phase]={n:compare_numeric_receipts(final[phase][n],w['checks'][phase][n]) for n in OUTPUTS}
        original_final_pass=original_final_pass and all(decision(c)['passed'] for c in w['checks'][phase].values())
    if w['numerical_checks_passed']!=(original['numerical_checks_passed'] and original_final_pass):
        raise ValueError('Producer worker decision contradicts its saved checks.')
    exact=all(actual[p][n].tobytes()==actual['before'][n].tobytes() for p in ('observed','after') for n in OUTPUTS)
    if exact!=w['phase_bitwise_equal']:raise ValueError('Producer phase-equality label changed.')
    numeric=fresh['numerical_checks_passed'] and all(c['passed'] for phase in final.values() for c in phase.values())
    consistent=compared['decisions_match'] and all(c['decisions_match'] for phase in final_comparisons.values() for c in phase.values())
    return {'kind':KIND,'worker_sha256':expected_worker_sha,'capture_integrity_verified':True,
        'threads':w['threads'],'replay':w['replay'],'numerical_values_passed':numeric,
        'producer_decisions_match_reassessment':consistent,'saved_value_evidence_passed':numeric and consistent,
        'producer_context':before,'reassessment_context':{
            'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__),
            'architecture':platform.machine(),'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            'numpy_build':str(np.__config__.CONFIG)},
        'phase_bitwise_equal':exact,'fresh_stage_checks':fresh,'stage_receipt_comparison':compared,
        'fresh_final_checks':final,'final_receipt_comparisons':final_comparisons,
        'bounds':BOOKKEEPING,'original_parent_failure_waived':False,'research_admitted':False}


def verify_reviewed_bundle(root,review_file,review_sha):
    root=Path(root).resolve();review_file=Path(review_file)
    if digest(review_file)!=review_sha:raise ValueError('Independent review anchor changed.')
    review=json.loads(review_file.read_text())
    if (review['kind']!='cpu-oracle-scratch-target-independent-review-v1'
            or review['target_control_sha256']!=CONTROL_SHA or review['independent_review_complete'] is not True
            or review['original_full_session_passed'] is not False
            or review['storage']['setup_allocated_ceiling_met'] is not False):
        raise ValueError('Require the preserved rejected target review, not an acceptance override.')
    relative=Path(safe_name(review['local_session']));session=root/relative
    if not session.resolve().is_relative_to(root/'runs/pretrained/diagnostic'):
        raise ValueError('Unexpected source bundle location.')
    if digest(session/'response.bin')!=review['response_sha256']:
        raise ValueError('Reviewed response changed.')
    with (session/'response.bin').open('rb') as handle:
        lines=[handle.readline(2**20+1),handle.readline(2**20+1)]
        if any(len(line)>2**20 or not line.endswith(b'\n') for line in lines):
            raise ValueError('Invalid bounded original transport header.')
        first,header=map(json.loads,lines)
    if first['control_sha256']!=CONTROL_SHA or header['control_sha256']!=CONTROL_SHA:
        raise ValueError('Original source control differs.')
    saved=session/'saved';files=header['files']
    if saved.is_symlink() or not saved.resolve().is_relative_to(session.resolve()):
        raise ValueError('Saved bundle root is aliased outside its session.')
    actual=set()
    for current,dirs,names in os.walk(saved,followlinks=False):
        for name in dirs+names:
            if (Path(current)/name).is_symlink():raise ValueError('Symlink in saved bundle.')
        for name in names:actual.add(str((Path(current)/name).relative_to(saved)))
    if len(files)!=review['outer_regular_members_verified'] or actual!=set(files):
        raise ValueError('Original saved member coverage changed.')
    for name,item in files.items():
        path=saved/safe_name(name)
        if (not path.is_file() or path.stat().st_size!=item['bytes'] or path.stat().st_mtime_ns!=item['mtime_ns']
                or digest(path)!=item['sha256']):
            raise ValueError('Original saved source/array/receipt changed: '+name)
    manifest=json.loads((saved/'code/CPU-ORACLE-HANDOFF.json').read_text())
    sha=digest(saved/'code/CPU-ORACLE-HANDOFF.json')
    if sha!='952ee2e1a9e45914b36557d0d3e5700ca2c1c0efb10ded675a2e443723ddb4b5':
        raise ValueError('Original CPU handoff differs from the reviewed packet.')
    verify(saved/'code',sha)
    for name,expected in manifest['original_scientific_binding']['source_sha256'].items():
        if digest(root/name)!=expected:raise ValueError('Frozen scientific source changed.')
    for name in set(manifest['new_sources'])|{'models/attention_fixture.py','models/attention_saved_evidence.py'}:
        if digest(root/name)!=manifest['files'][name]['sha256']:raise ValueError('Original array reader/math dependency changed.')
    return review,saved,manifest,sha


def reassess(root,review_file,review_sha,output):
    review,saved,manifest,sha=verify_reviewed_bundle(root,review_file,review_sha)
    output=Path(output).resolve()
    old_tree=saved.parents[2]
    if output==old_tree or output.is_relative_to(old_tree) or old_tree.is_relative_to(output):
        raise ValueError('New reassessment must not write in or replace the captured tree.')
    output.mkdir(parents=True,exist_ok=False)
    original=read_record(saved/'results/session.json')
    if (original['capture_complete'] is not False or original['error']!='Stage comparisons do not reaggregate.'
            or [[r['threads'],r['replay']] for r in original['tasks']]!=[[4,0],[1,0],[4,1],[1,1]]):
        raise ValueError('Original failure/coverage changed.')
    values,_=load(saved/'code'/manifest['fixture']['folder'],manifest['fixture']['sha256'],case='encoder-1',tiny=False)
    workers=[]
    for i,row in enumerate(original['tasks']):
        result=audit_worker(saved/f'results/workers/{i}',row['worker_sha256'],values,manifest['fixture'],tiny=False,handoff_sha=sha)
        ctx=result['producer_context']
        if (ctx['host'],ctx['python'],ctx['torch'],ctx['numpy'])!=('linux10603','3.11.17','2.8.0+cu128','2.4.6'):
            raise ValueError('Producer runtime differs from reviewed capture.')
        write_record(output/f'worker-{i}.json',result)
        workers.append({'worker_sha256':result['worker_sha256'],'reassessment_sha256':digest(output/f'worker-{i}.json'),
            'threads':result['threads'],'replay':result['replay'],
            'saved_value_evidence_passed':result['saved_value_evidence_passed'],
            'diagnostic_difference_count':result['stage_receipt_comparison']['diagnostic_difference_count']})
    summary={'kind':KIND,'source_review_sha256':review_sha,'source_response_sha256':review['response_sha256'],
        'worker_reassessments':workers,'array_checks_passed':all(w['saved_value_evidence_passed'] for w in workers),
        'stage_comparisons':2048,'final_reference_comparisons':48,'bounds':BOOKKEEPING,
        'original_parent_accepted':False,'original_parent_refusal':original['error'],
        'original_setup_accepted':False,'original_refusals_waived':False,
        'launch_authorized':False,'control_host_ready':False,'research_admitted':False,
        'historical_discrepancy_resolved':False,'new_university_work_started':False,
        'sources':{name:digest(Path(root)/name) for name in ('models/cpu_oracle_reassessment_v2.py','models/oracle_trace.py','models/attention_fixture.py','models/attention_saved_evidence.py')},
        'scope':'Offline reassessment of authenticated saved arrays; no original protocol acceptance or launch authorization.'}
    write_record(output/'summary.json',summary)
    return summary


def validate_summary(summary,review_sha,response_sha):
    expected={'kind','source_review_sha256','source_response_sha256','worker_reassessments','array_checks_passed',
        'stage_comparisons','final_reference_comparisons','bounds','original_parent_accepted','original_parent_refusal',
        'original_setup_accepted','original_refusals_waived','launch_authorized','control_host_ready','research_admitted',
        'historical_discrepancy_resolved','new_university_work_started','sources','scope'}
    if set(summary)!=expected:raise ValueError('Reassessment summary schema changed.')
    for key in ('original_parent_accepted','original_setup_accepted','original_refusals_waived',
                'launch_authorized','control_host_ready','research_admitted','historical_discrepancy_resolved','new_university_work_started'):
        if summary.get(key) is not False:raise ValueError('Reassessment cannot change historical failures or authorize work.')
    if (summary['kind']!=KIND or summary['source_review_sha256']!=review_sha
            or summary['source_response_sha256']!=response_sha or summary['bounds']!=BOOKKEEPING
            or summary['original_parent_refusal']!='Stage comparisons do not reaggregate.'
            or type(summary['array_checks_passed']) is not bool
            or type(summary['stage_comparisons']) is not int or summary['stage_comparisons']!=2048
            or type(summary['final_reference_comparisons']) is not int or summary['final_reference_comparisons']!=48):
        raise ValueError('Reassessment source, scope, coverage or bounds changed.')
    expected_sources={'models/cpu_oracle_reassessment_v2.py','models/oracle_trace.py','models/attention_fixture.py','models/attention_saved_evidence.py'}
    if set(summary['sources'])!=expected_sources:raise ValueError('Reassessment dependency coverage changed.')
    rows=summary['worker_reassessments']
    if not isinstance(rows,list) or len(rows)!=4:raise ValueError('Four complete reassessment rows required.')
    for row,task in zip(rows,((4,0),(1,0),(4,1),(1,1))):
        if (set(row)!={'worker_sha256','reassessment_sha256','threads','replay','saved_value_evidence_passed','diagnostic_difference_count'}
                or (row['threads'],row['replay'])!=task or type(row['saved_value_evidence_passed']) is not bool
                or type(row['diagnostic_difference_count']) is not int or not 0<=row['diagnostic_difference_count']<=1536):
            raise ValueError('Reassessment row identity/decision changed.')
        for key in ('worker_sha256','reassessment_sha256'):
            if not isinstance(row[key],str) or len(row[key])!=64 or any(c not in '0123456789abcdef' for c in row[key]):
                raise ValueError('Invalid reassessment artifact SHA256.')
    if summary['array_checks_passed']!=all(row['saved_value_evidence_passed'] for row in rows):
        raise ValueError('Summary decision contradicts worker labels.')


def verify_reassessment(root,review_file,review_sha,output,summary_sha):
    """Recheck stored v2 decisions in a fresh local context, retaining metric drift."""
    review,saved,manifest,handoff_sha=verify_reviewed_bundle(root,review_file,review_sha)
    output=Path(output);summary=read_record(output/'summary.json',expected_sha256=summary_sha)
    validate_summary(summary,review_sha,review['response_sha256'])
    for name,expected in summary['sources'].items():
        if digest(Path(root)/safe_name(name))!=expected:raise ValueError('Reassessment code changed.')
    session=read_record(saved/'results/session.json')
    if len(summary['worker_reassessments'])!=len(session['tasks']) or len(session['tasks'])!=4:
        raise ValueError('Incomplete reassessment coverage.')
    values,_=load(saved/'code'/manifest['fixture']['folder'],manifest['fixture']['sha256'],case='encoder-1',tiny=False)
    rows=[]
    for i,(item,task) in enumerate(zip(summary['worker_reassessments'],session['tasks'])):
        stored=read_record(output/f'worker-{i}.json',expected_sha256=item['reassessment_sha256'])
        fresh=audit_worker(saved/f'results/workers/{i}',task['worker_sha256'],values,manifest['fixture'],tiny=False,handoff_sha=handoff_sha)
        if (stored['kind']!=KIND or item['worker_sha256']!=task['worker_sha256']
                or [stored['threads'],stored['replay']]!=[item['threads'],item['replay']]
                or [fresh['threads'],fresh['replay']]!=[item['threads'],item['replay']]
                or stored['worker_sha256']!=fresh['worker_sha256']
                or stored['producer_context']!=fresh['producer_context']
                or stored['bounds']!=BOOKKEEPING or stored['capture_integrity_verified'] is not True
                or stored['original_parent_failure_waived'] is not False or stored['research_admitted'] is not False):
            raise ValueError('Worker reassessment identity, domain or producer context changed.')
        for key in ('numerical_values_passed','producer_decisions_match_reassessment','saved_value_evidence_passed','phase_bitwise_equal'):
            if stored[key]!=fresh[key]:raise ValueError('Saved v2 numerical decisions no longer match independently checked arrays.')
        compared=compare_stage_receipts(fresh['fresh_stage_checks'],stored['fresh_stage_checks'])
        if not compared['decisions_match']:raise ValueError('V2 stage decisions changed.')
        for phase in ('before','observed','after'):
            for name in OUTPUTS:
                c=compare_numeric_receipts(fresh['fresh_final_checks'][phase][name],stored['fresh_final_checks'][phase][name])
                if not c['decisions_match']:raise ValueError('V2 fixed-final decisions changed.')
        if stored['saved_value_evidence_passed']!=item['saved_value_evidence_passed']:
            raise ValueError('Worker summary label contradicts its full reassessment.')
        rows.append({'threads':fresh['threads'],'replay':fresh['replay'],
            'saved_value_evidence_passed':fresh['saved_value_evidence_passed'],
            'diagnostic_difference_count':compared['diagnostic_difference_count']})
    if summary['array_checks_passed']!=all(row['saved_value_evidence_passed'] for row in rows):
        raise ValueError('Reassessment summary label changed.')
    return {'kind':KIND,'verified':True,'array_checks_passed':summary['array_checks_passed'],
        'worker_checks':rows,'original_parent_accepted':False,'original_setup_accepted':False,
        'launch_authorized':False,'research_admitted':False}

"""New stdlib synthetic metadata checks; no ledger/FileSink/worker lifecycle."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace as NS
import unittest

from models import predictor_pgd_phase_encoding_v1 as codec
from models import predictor_pgd_phase_encoding_build_v1 as build

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'runs/pretrained/diagnostic/pgd-phase-encoding-v1-20261124'
EVIDENCE=Path(os.environ['VJEPA_PHASE_ENCODING_EVIDENCE'])
ORIGIN='SYNTHETIC STDLIB METADATA ONLY; NO WORKER, USER GRANT OR SCIENTIFIC EXECUTION'


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def put(path,value):
    raw=value if type(value) is bytes else canonical(value)
    path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(),0o600)
        stream.write(raw)


def tagged(value):
    # Preserve nested non-string key types/tuple order for independent readback.
    if type(value) is dict:
        return {'type':'dict','items':[[tagged(k),tagged(v)] for k,v in value.items()]}
    if type(value) in (tuple,list):
        return {'type':type(value).__name__,'items':[tagged(v) for v in value]}
    if type(value) is float:
        return {'type':'float','hex':value.hex()}
    if type(value) is int:
        return {'type':'int','decimal':str(value)}
    if type(value) in (str,bool,type(None)):
        return {'type':type(value).__name__,'value':value}
    raise TypeError('Only plain synthetic JSON metadata has an evidence codec.')


def extracted(path):
    tree=ast.parse(path.read_bytes())
    ledger=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PhaseLedger')
    record=next(n for n in ledger.body if isinstance(n,ast.FunctionDef) and n.name=='record')
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)
           and n.name in ('enc','_phase_json_value_end','insert_row_sha256')]+[record]
    namespace={'__name__':'_SYNTHETIC_METADATA_RECORD_ONLY','json':json,'hashlib':hashlib,'math':math,'MAX_ROWS':384}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<ONLY-ENC-RECORD-AND-BYTE-HELPERS>','exec'),namespace)
    return namespace


class AppendSpy:
    """Records one requested metadata append; performs no journal or fsync work."""
    def __init__(self,error=None):
        self.attempts=[]
        self.error=error
    def append(self,raw):
        self.attempts.append(raw)
        if self.error is not None:
            raise self.error


def exercise(path,fields=None,*,rows=0,previous='0'*64,append_error=None,event='begin',ordinal=0):
    namespace=extracted(path)
    original=namespace['enc']
    encoded=[]
    snapshot=[]
    calls=[]
    def observing(row):
        calls.append('enc')
        raw=original(row)
        encoded.append(raw)
        if not snapshot:
            snapshot.append(dict(row))  # Private top-level copy before hash mutation.
        return raw
    namespace['enc']=observing
    sink=AppendSpy(append_error)
    self=NS(rows=rows,previous=previous,sink=sink,role='original-unit',owner=(424242,101),
        domain='process-monotonic/424242',control='a'*64,plan='b'*64)
    token=NS(ordinal=ordinal,parent=None,name='owned_context',index=ordinal,
        started=100.0,retrospective=False,deadline=820.0)
    error=None
    try:
        namespace['record'](self,event,token,101.0,**(fields or {}))
    except BaseException as caught:
        error=caught
    return NS(state=self,sink=sink,error=error,encoded=encoded,calls=calls,snapshot=snapshot)


BASE=STAGE/'source-candidate-v1/baseline.py'
CANDIDATE=STAGE/'source-candidate-v1/candidate.py'


class PhaseEncodingTests(unittest.TestCase):
    def setUp(self):
        self.area=EVIDENCE/self._testMethodName
        self.area.mkdir(mode=0o700)
        self.number=0
        put(self.area/'ORIGIN.json',{'origin':ORIGIN,'scientific_qualified':False,
            'performance_qualified':False,'training_admitted':False,'append_is_spy_only':True})

    def pair(self,fields=None,*,label='row',**kwargs):
        before=exercise(BASE,fields,**kwargs)
        after=exercise(CANDIDATE,fields,**kwargs)
        self.assertEqual(type(before.error),type(after.error))
        if before.error is not None:
            self.assertEqual(str(before.error),str(after.error))
        self.assertEqual(before.sink.attempts,after.sink.attempts)
        self.assertEqual((before.state.rows,before.state.previous),(after.state.rows,after.state.previous))
        if before.encoded:
            self.assertEqual(before.encoded[0],after.encoded[0])
        folder=self.area/f'{self.number:03}-{label}'
        self.number+=1
        if before.snapshot:
            row=before.snapshot[0]
            prehash=canonical(row)
            self.assertEqual(prehash,before.encoded[0])
            completed=dict(row,row_sha256=digest(prehash))
            final=canonical(completed)
            if before.sink.attempts:
                self.assertEqual(final,before.sink.attempts[0])
                put(folder/'EXPECTED-FINAL.bin',final)
                put(folder/'BASELINE-FINAL.bin',before.sink.attempts[0])
                put(folder/'CANDIDATE-FINAL.bin',after.sink.attempts[0])
            put(folder/'ROW-TYPED.json',tagged(row))
            put(folder/'BASELINE-PREHASH.bin',before.encoded[0])
            put(folder/'CANDIDATE-PREHASH.bin',after.encoded[0])
        put(folder/'RESULT.json',{'origin':ORIGIN,'equal':True,'append_spy_only':True,
            'baseline_enc_attempts':len(before.calls),'candidate_enc_attempts':len(after.calls),
            'append_attempts':len(before.sink.attempts),'rows_after':after.state.rows,
            'previous_after':after.state.previous,'error_type':None if after.error is None else type(after.error).__name__,
            'error_message':None if after.error is None else str(after.error),
            'actual_clock_timer_process_science_or_worker_work':False})
        return before,after

    def direct(self,row,label):
        prehash=canonical(row)
        value=digest(prehash)
        expected=canonical(dict(row,row_sha256=value))
        actual=codec.insert_row_sha256(prehash,value)
        self.assertEqual(actual,expected)
        folder=self.area/label
        put(folder/'ROW-TYPED.json',tagged(row))
        put(folder/'PREHASH.bin',prehash)
        put(folder/'EXPECTED-FINAL.bin',expected)
        put(folder/'CANDIDATE-FINAL.bin',actual)
        put(folder/'RESULT.json',{'origin':ORIGIN,'equal':True,'row_sha256':value,'producer_provenance':'unchanged enc'})

    def test_full_module_inverse_and_every_original_guard_body(self):
        baseline,candidate,raw_codec,result=build.assemble(ROOT)
        self.assertEqual(baseline,BASE.read_bytes())
        self.assertEqual(candidate,CANDIDATE.read_bytes())
        restored=candidate
        for edit in reversed(result['edits']):
            start=edit['start'];stop=start+len(edit['after'].encode())
            self.assertEqual(restored[start:stop],edit['after'].encode())
            restored=restored[:start]+edit['before'].encode()+restored[stop:]
        self.assertEqual(restored,baseline)
        self.assertEqual(result['execution_binding'],None)
        self.assertFalse(result['new_operational_graph_package_or_grant_emitted'])
        put(self.area/'RESULT.json',{'origin':ORIGIN,'full_module_inverse':True,
            'all_original_guard_timer_cleanup_and_FileSink_bytes_preserved':True,'edits':len(result['edits'])})

    def test_begin_and_completed_end_records(self):
        self.pair()
        self.pair({'completed':True,'end':101.0,'seconds':1.0,'error':None,
            'own_end_row_publication_included':False},event='end',label='end')

    def test_failed_end_error_and_exclusion_bytes(self):
        self.pair({'completed':False,'end':101.0,'seconds':1.0,
            'error':{'type':'SyntheticError','message':'synthetic failure'},
            'own_end_row_publication_included':False},event='end')

    def test_root_insertion_at_each_key_position(self):
        for i,row in enumerate([{}, {'a':1}, {'z':2}, {'a':1,'z':2},
            {'row_sha2560':0}, {'row_sha255':0,'row_sha257':1}, {'é':1,'Ω':2}, {'\x00':0,'😀':1}]):
            self.direct(row,f'position-{i}')

    def test_existing_hash_member_is_replaced_but_included_in_prehash(self):
        for i,old in enumerate([None,'old synthetic hash',{'nested':['row_sha256',0]},'x'*5000]):
            self.pair({'row_sha256':old},label=f'replacement-{i}')

    def test_role_like_and_hash_like_text_inside_nested_strings(self):
        self.pair({'payload':{'role':'\",\"row_sha256\":\"fake\",\"scientific_qualified\":true',
            'message':'before }, { after \\ \\" \n row_sha256',
            'nested':[{'row_sha256':{'role':['original-unit','candidate-replay']}}]}})

    def test_unicode_escaping_and_lone_surrogates(self):
        self.pair({'payload':['é','e\u0301','Ω','😀','\ud800','\udfff','\u2028',
            {'汉字':'\"\\\b\f\n\r\t','\ud800':'\udfff'}]})

    def test_all_ASCII_controls_are_retained(self):
        self.pair({'payload':''.join(chr(i) for i in range(32))+'\"\\/\x7f'})

    def test_nested_empty_containers_and_tuples(self):
        self.pair({'payload':[{},[],(),{'x':([],{},((),[{}]))}]})

    def test_nested_numeric_key_sorting_keeps_original_types(self):
        self.pair({'payload':{2:'two',10:'ten',-1:'minus'}})
        self.pair({'payload':{2.5:False,0.5:True}},label='float-keys')

    def test_signed_zero_subnormals_and_large_integer_bytes(self):
        self.pair({'payload':[-0.0,0.0,5e-324,1.7976931348623157e308,1e-20,2**256,-(2**256)]})

    def test_deep_valid_containers_without_recursive_decode(self):
        payload=0
        for i in range(128):
            payload=[payload] if i%2 else {'k':payload}
        self.pair({'payload':payload})

    def test_deterministic_metadata_matrix(self):
        texts=['plain','row_sha256','\",\"role\":\"fake','é','😀','\ud800','\\\n\t','{}[],:']
        scalars=[None,False,True,0,-0.0,5e-324,2**100,'x']
        for i in range(64):
            self.pair({'payload':{'label':texts[i%8],'value':scalars[(i//8)%8],
                'nested':[{texts[(i+3)%8]:[i,scalars[i%8]]}]}},label=f'matrix-{i}')

    def test_new_synthetic_hash_chain_matches_at_every_row(self):
        previous='0'*64
        for i in range(32):
            before,after=self.pair({'payload':{'synthetic_sequence':i}},rows=i,previous=previous,
                ordinal=i,event='begin' if i%2==0 else 'end',label=f'chain-{i}')
            self.assertIsNone(after.error)
            previous=after.state.previous

    def test_exact_4096_byte_row_is_accepted(self):
        initial=exercise(BASE,{'padding':''})
        size=4096-len(initial.sink.attempts[0])
        before,after=self.pair({'padding':'x'*size})
        self.assertIsNone(after.error)
        self.assertEqual(len(after.sink.attempts[0]),4096)

    def test_4097_byte_row_refuses_before_append_and_counter_commit(self):
        initial=exercise(BASE,{'padding':''})
        size=4097-len(initial.sink.attempts[0])
        before,after=self.pair({'padding':'x'*size})
        self.assertIsInstance(after.error,ValueError)
        self.assertEqual(after.sink.attempts,[])
        self.assertEqual(after.state.rows,0)

    def test_row_count_ceiling_refuses_before_encoding(self):
        before,after=self.pair(rows=384)
        self.assertIsInstance(after.error,ValueError)
        self.assertEqual(after.calls,[])
        self.assertEqual(after.sink.attempts,[])

    def test_append_spy_error_preserves_primary_and_uncommitted_counters(self):
        primary=OSError('Synthetic append spy refused; no real journal work')
        before,after=self.pair(append_error=primary)
        self.assertIs(before.error,primary)
        self.assertIs(after.error,primary)
        self.assertEqual(after.state.rows,0)
        self.assertEqual(after.state.previous,'0'*64)

    def test_success_uses_two_baseline_encodings_and_one_candidate_encoding(self):
        before,after=self.pair({'payload':'count serialization only, not elapsed time'})
        self.assertEqual(len(before.calls),2)
        self.assertEqual(len(after.calls),1)

    def test_nonfinite_values_keep_original_refusal_before_append(self):
        for i,value in enumerate([float('nan'),float('inf'),float('-inf')]):
            before,after=self.pair({'payload':{'nested':[value]}},label=f'nonfinite-{i}')
            self.assertIsInstance(after.error,ValueError)
            self.assertEqual(after.sink.attempts,[])

    def test_unsupported_values_keep_original_type_refusal(self):
        for i,value in enumerate([b'bytes',{1,2},object()]):
            before,after=self.pair({'payload':value},label=f'unsupported-{i}')
            self.assertIsInstance(after.error,TypeError)

    def test_mixed_nested_key_types_keep_original_sort_refusal(self):
        before,after=self.pair({'payload':{'string':1,2:'number'}})
        self.assertIsInstance(after.error,TypeError)

    def test_circular_container_keeps_original_refusal(self):
        value=[]
        value.append(value)
        before,after=self.pair({'payload':value})
        self.assertIsInstance(after.error,ValueError)

    def test_parser_refuses_truncated_malformed_and_trailing_inputs(self):
        malformed=[b'',b'[]\n',b'{',b'{"a":1}\nextra',b'{"a":1,}\n',b'{"a":[1,]}\n',
            b'{"a":{"k":}}\n',b'{"a":{"k":1,}}\n',b'{"a":[1}}\n',b'{"a":truefalse}\n',
            b'{"a":"unterminated}\n',b'{"a":NaN}\n',b'{"a":1e999}\n',b'{"a": 1}\n',
            b'{"a":[1 2]}\n',b'{"a":1} }\n',b'{"a":1}\r\n']
        for i,raw in enumerate(malformed):
            with self.assertRaises(ValueError):codec.insert_row_sha256(raw,'0'*64)
            put(self.area/f'{i:02}-REFUSED.bin',raw)

    def test_parser_refuses_duplicate_or_out_of_order_root_keys(self):
        for i,raw in enumerate([b'{"a":1,"a":2}\n',b'{"z":1,"a":2}\n',b'{1:2}\n']):
            with self.assertRaises(ValueError):codec.insert_row_sha256(raw,'0'*64)
            put(self.area/f'{i:02}-REFUSED.bin',raw)

    def test_parser_requires_immutable_bytes_and_exact_hex_digest(self):
        for value in (bytearray(b'{}\n'),memoryview(b'{}\n'),'{}\n',None):
            with self.assertRaises(ValueError):codec.insert_row_sha256(value,'0'*64)
        for value in ('0'*63,'0'*65,'G'*64,'A'*64,None,123):
            with self.assertRaises(ValueError):codec.insert_row_sha256(b'{}\n',value)
        put(self.area/'RESULT.json',{'origin':ORIGIN,'mutable_inputs_and_invalid_digests_refused':True})

    def test_source_pin_drift_refuses_without_any_emission(self):
        config=json.loads((ROOT/build.INPUTS).read_bytes())
        project=self.area/'private-source-copy'
        for name in [build.INPUTS,*config['files']]:
            put(project/name,(ROOT/name).read_bytes())
        (project/config['pins']['codec']).write_bytes(b'SYNTHETIC SOURCE PIN DRIFT\n')
        dest=self.area/'must-stay-absent'
        with self.assertRaisesRegex(ValueError,'Pinned'):
            build.emit_local(project,dest)
        self.assertFalse(dest.exists())
        put(self.area/'RESULT.json',{'origin':ORIGIN,'source_drift_refused_before_emission':True})

    def test_source_has_no_new_port_or_operational_API(self):
        tree=ast.parse((ROOT/'models/predictor_pgd_phase_encoding_v1.py').read_bytes())
        imports=[n.names[0].name for n in tree.body if isinstance(n,ast.Import)]
        self.assertEqual(imports,['json','math'])
        self.assertNotIn('FileSink',extracted(CANDIDATE))
        self.assertNotIn('PhaseLedger',extracted(CANDIDATE))
        self.assertIsNone(build.EXECUTION_BINDING)
        self.assertFalse(any(hasattr(build,name) for name in ('launch','material','authorization','prepare_unarmed')))
        put(self.area/'RESULT.json',{'origin':ORIGIN,'no_FileSink_ledger_lifecycle_or_workload_ports_executed':True,
            'candidate_source_only_not_runtime_adoption':True,'performance_or_budget_fit_qualified':False})

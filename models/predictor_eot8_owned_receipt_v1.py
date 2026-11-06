"""Stdlib-only check of saved graph/thread scope receipts; no tensor execution."""
from collections import Counter

PHASES=('EOT8-before','draw-before','draw-profiled','draw-after','EOT8-after')

def verify_ownership(worker):
    def require(condition,message):
        if not condition:raise ValueError('Graph ownership receipt: '+message)
    def positive(value):return type(value) is int and value>0
    require(worker.get('kind')=='predictor-clean-eot8-owned-worker-v1','worker identity')
    require(type(worker.get('local_tiny_analogue')) is bool,'domain')
    require(positive(worker.get('pid')),'worker PID')
    require(worker.get('mode') in ('original','predictor-math'),'mode')
    require(worker.get('ownership_refusals')==[],'refusal evidence missing or nonempty')
    scopes=worker.get('backward_scopes');rows=worker.get('call_rows')
    require(type(scopes) is list and len(scopes)==5,'five complete backward scopes required')
    require(type(rows) is list and 0<len(rows)<=4096,'bounded nonempty call rows required')
    counts={'encoder':2,'predictor':2} if worker['local_tiny_analogue'] else {'encoder':24,'predictor':12}
    owners={role+'.'+str(i) for role,n in counts.items() for i in range(n)}
    creator=scopes[0].get('creator_thread');require(positive(creator),'creator thread')
    graphs=set();cross_thread=0
    for i,(phase,scope) in enumerate(zip(PHASES,scopes)):
        require(type(scope) is dict,'scope record')
        require(type(scope.get('index')) is int and scope['index']==i and scope.get('phase')==phase,'ordered phase/index')
        require(scope.get('state')=='complete' and scope.get('hook_removed') is True,'scope incomplete or hook not removed')
        require(scope.get('pid')==worker['pid'] and scope.get('creator_thread')==creator,'scope process/creator')
        require(all(positive(scope.get(k)) for k in ('runner_thread','root_hook_thread','loss_id','input_id')),'scope identities')
        graph=scope.get('graph_id');require(type(graph) is int and graph>=0 and graph not in graphs,'unique observed GraphTask')
        graphs.add(graph);threads=scope.get('call_threads')
        require(type(threads) is list and 1<=len(threads)<=8 and all(positive(t) for t in threads) and len(set(threads))==len(threads),'bounded observed call threads')
        if scope['root_hook_thread']!=creator or any(t!=creator for t in threads):cross_thread+=1
    primary={phase:Counter() for phase in PHASES};back=Counter()
    for i,row in enumerate(rows):
        require(type(row) is dict and type(row.get('index')) is int and row['index']==i,'call row/index')
        phase=row.get('phase');require(phase in PHASES,'call phase')
        scope=scopes[PHASES.index(phase)]
        require(row.get('owner') in owners and row.get('owner')==str(row.get('role'))+'.'+str(row.get('block')),'owned block')
        require(row.get('pid')==worker['pid'] and row.get('creator_thread')==creator and positive(row.get('thread')),'call process/thread')
        require(row.get('entry_flags')==row.get('exit_flags') and type(row.get('entry_flags')) is dict,'backend context restoration')
        require(type(row.get('during_backward')) is bool,'call phase domain')
        if row['during_backward']:
            require(type(row.get('graph_id')) is int and row['graph_id']==scope['graph_id'] and type(row.get('backward_scope')) is int and row['backward_scope']==scope['index'],'backward exact graph/scope')
            require(row['thread'] in scope['call_threads'],'unobserved recomputation thread')
            require(row.get('outcome')=='returned' or row.get('outcome')=='interrupted' and row.get('error_type')=='_StopRecomputationError','backward call result')
            back[phase]+=1
        else:
            require(row.get('thread')==creator and type(row.get('graph_id')) is int and row['graph_id']==-1 and row.get('backward_scope') is None,'unowned forward thread/graph/scope')
            require(row.get('outcome')=='returned','forward call incomplete')
            primary[phase][row['owner']]+=1
    for phase in PHASES:
        require(primary[phase]==Counter({name:16 if phase.startswith('EOT8') else 2 for name in owners}),'eight draws/two families/block coverage')
        require(back[phase]>0,'actual recomputation unobserved')
    return {'kind':'predictor-eot8-graph-ownership-receipt-v1','scopes':5,'call_rows':len(rows),
            'owners':len(owners),'all_hooks_removed':True,'refusals':0,
            'observed_cross_creator_thread_scopes':cross_thread,
            'backend_proved_by_this_receipt':False,'timing_or_research_admission':False}

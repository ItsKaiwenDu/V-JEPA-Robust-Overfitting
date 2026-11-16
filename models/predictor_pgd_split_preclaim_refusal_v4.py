"""Cold-only correction of collection after a refused parent preparation.

Returns one function definition for later reviewed integration. It is not a new
runnable handoff, authorization, target connection or experiment.
"""
import ast
import copy
import hashlib

OUTER_V3_SHA='8c21d936e5d311c7ddc0dffa4bc91c01a7f5c1c0e8731efa5fc33c58df57ac46'
BRANCH='''
if not owner.claimed or owner.driver.guard is None:
    if work_error is not None:
        raise work_error
    raise RuntimeError('Parent session was not initialized; no collection permitted.')
'''


def build(original):
    if type(original) is not bytes or hashlib.sha256(original).hexdigest()!=OUTER_V3_SHA:
        raise ValueError('Exact consumed v3 source required; never patch it in place.')
    tree=ast.parse(original)
    function=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='remote_session'))
    block=next(n for n in function.body if isinstance(n,ast.Try))
    indexes=[i for i,n in enumerate(block.body) if isinstance(n,ast.Expr) and ast.unparse(n)=='owner.driver.assert_idle()']
    if len(indexes)!=1:raise ValueError('One original owned-idle barrier required.')
    index=indexes[0]+1
    if not ast.unparse(block.body[index]).startswith('alarms.arm(min(origin + 2400,'):
        raise ValueError('Original finalization boundary changed.')
    block.body[index:index]=ast.parse(BRANCH).body
    result=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    data=ast.unparse(result).encode()+b'\n'
    compile(data,'<UNEXECUTED-PRECLAIM-REFUSAL-FUNCTION-V4>','exec')
    return data,{'uninitialized_parent_collection_barrier':1,'authorized':False,'target_helper_ready':False,
        'scientific_qualified':False,'training_admitted':False}

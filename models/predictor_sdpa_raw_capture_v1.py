"""Bounded raw extraction only; semantic validation belongs after publication.

Reads an already completed profiler via the same frozen APIs. No torch import,
profiler start, model, derivative, device, process or network operation. Serializable
invalid metadata is preserved exactly, never repaired or accepted here. Values that
cannot fit the bounded JSON transport refuse with exact record/field context.
"""
import math
from models import predictor_sdpa_launch_evidence_v1 as frozen
KIND=frozen.KIND
API=frozen.API
MAX_BYTES=frozen.MAX_BYTES
MAX_EVENTS=frozen.MAX_EVENTS
MAX_FIELD_CHARS=16384
MAX_INTEGER_BITS=256


def scalar(value,where):
    kind=type(value)
    if value is None or kind is bool:return value
    if kind is int:
        if value.bit_length()<=MAX_INTEGER_BITS:return value
        raise ValueError(f'Raw capture {where}: integer exceeds {MAX_INTEGER_BITS} bits; no coercion/truncation.')
    if kind is float:
        if math.isfinite(value):return value
        raise ValueError(f'Raw capture {where}: nonfinite float {value!r}; no JSON substitution.')
    if kind is str:
        if len(value)<=MAX_FIELD_CHARS:return value
        raise ValueError(f'Raw capture {where}: string length {len(value)} exceeds {MAX_FIELD_CHARS}; no truncation.')
    raise ValueError(f'Raw capture {where}: unsupported scalar type {kind.__name__[:128]}; no conversion.')


def call(owner,method,where):
    function=getattr(owner,method,None)
    if not callable(function):raise ValueError(f'Raw capture {where}: missing API {method}; no fallback.')
    try:value=function()
    except Exception as e:raise ValueError(f'Raw capture {where}: API {method} raised {type(e).__name__[:128]}.') from e
    return scalar(value,where)


def capture(profile,*,torch_version):
    result=getattr(getattr(profile,'profiler',None),'kineto_results',None)
    if result is None:raise ValueError('Completed raw Kineto result required; no processed-event fallback.')
    return capture_results(result,torch_version=torch_version)


def capture_results(result,*,torch_version):
    function=getattr(result,'events',None)
    if not callable(function):raise ValueError('Raw capture events: missing API events; no fallback.')
    try:source=function()
    except Exception as e:raise ValueError(f'Raw capture events: API events raised {type(e).__name__[:128]}.') from e
    if not isinstance(source,(list,tuple)) or not 0<len(source)<=MAX_EVENTS:
        raise ValueError('Complete bounded raw event sequence required; never truncate.')
    trace={'kind':KIND,'source_api':API,'torch_version':scalar(torch_version,'torch_version'),
        'trace_start_ns':call(result,'trace_start_ns','trace_start_ns'),'records':[]}
    total=len(frozen.encoded(trace))
    for index,event in enumerate(source):
        row={'index':index}
        for field,method in frozen.METHODS.items():row[field]=call(event,method,f'record {index} field {field}')
        enum=getattr(event,'device_type',None)
        if not callable(enum):raise ValueError(f'Raw capture record {index} field device: missing API device_type.')
        try:device=str(enum())
        except Exception as e:raise ValueError(f'Raw capture record {index} field device: API device_type raised {type(e).__name__[:128]}.') from e
        if device not in ('DeviceType.CPU','DeviceType.CUDA'):
            raise ValueError(f'Raw capture record {index} field device: unsupported device_type {device[:128]!r}; no fallback.')
        row['device']={'DeviceType.CPU':'cpu','DeviceType.CUDA':'cuda'}[device]
        # Account for JSON row bytes without its newline, and the list comma.
        total+=len(frozen.encoded(row))-1+(index>0)
        if total>MAX_BYTES:raise ValueError('Raw capture envelope exceeds existing 8 MiB ceiling; no truncation.')
        trace['records'].append(row)
    data=frozen.encoded(trace)
    if len(data)!=total or len(data)>MAX_BYTES:raise ValueError('Exact bounded raw envelope accounting required.')
    return trace

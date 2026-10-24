#!/usr/bin/env bash
# Source-only handoff; existing university Python; no GPU, downloads or installs.
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
py=${VJEPA_CPU_FIXTURE_PYTHON:-"$HOME/.venvs/vjepa/bin/python"}
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

verify_sources() {
  "$py" - "$base" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
manifest=json.loads((root/'CPU-FIXTURE-MANIFEST.json').read_text())
assert manifest['cpu_only'] and not manifest['research_admitted']
for name,expected in manifest['files_sha256'].items():
    path=root/name
    if (path.is_symlink() or not path.resolve().is_relative_to(root)
            or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected):
        raise SystemExit('Handoff checksum mismatch: '+name)
print('Verified CPU handoff files:',len(manifest['files_sha256']))
PY
}

finish() {
  status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$run/exit_status.txt"
  "$py" - "$base" "$run" "$status" <<'PY'
import json,pathlib,sys,xml.etree.ElementTree as ET
base,run=map(pathlib.Path,sys.argv[1:3]);status=int(sys.argv[3])
sys.path.insert(0,str(base))
from models.pilot_bundle import make_archive
suite=run/'tests.xml'
stats=None
if suite.exists():
    try:
        rows=list(ET.parse(suite).getroot().iter('testsuite'))
        stats={k:rows[0].attrib[k] for k in ('tests','failures','errors','skipped','time')}
    except (ET.ParseError,IndexError,KeyError):
        stats={'complete':False,'error':'JUnit output is partial or invalid.'}
result={'schema_version':1,'cpu_only':True,'diagnostic_only':True,
        'research_admitted':False,'gpu_benchmark_performed':False,
        'exit_status':status,'passed':status==0,'tests':stats}
(run/'session.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
files={p.name:p for p in run.iterdir() if p.is_file() and p.suffix in ('.json','.xml','.log','.txt')}
files['CPU-FIXTURE-MANIFEST.json']=base/'CPU-FIXTURE-MANIFEST.json'
temp=run/'pytest'
selected=[temp/'native-workers0/results',temp/'precision-workers0/results']
selected += [temp/f'test_fresh_process_training_re{i}' for i in range(4)]
for directory in selected:
    if directory.is_dir() and not directory.is_symlink():
        for path in directory.rglob('*'):
            if path.is_file() and not path.is_symlink() and path.suffix in ('.json','.log','.pt'):
                if path.stat().st_size>16*2**20:raise ValueError('Unexpected large fixture artifact.')
                files['fixtures/'+str(path.relative_to(temp))]=path
if sum(p.stat().st_size for p in files.values())>128*2**20:
    raise ValueError('CPU evidence exceeds the declared small-fixture export limit; preserve run directory.')
manifest=make_archive(run/'evidence.tar.gz',files)
(run/'evidence.tar.gz.manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print('CPU evidence export:',manifest['bytes'],'bytes; SHA256',manifest['sha256'])
PY
  date -u
  echo "CPU fixture check exit status: $status"
  exit "$status"
}

[[ $(uname -s) == Linux ]] || { echo 'This launcher is for the existing university Linux environment.' >&2; exit 2; }
[[ -x "$py" ]] || { echo "Missing existing environment: $py" >&2; exit 1; }
cd "$base"
if [[ ${1:-} == --worker ]]; then
  [[ $# == 2 ]] || exit 2
  run=$2
  trap finish EXIT
  verify_sources
  hostname
  date -u
  df -PT "$run"
  "$py" - "$run/runtime.json" <<'PY'
import fcntl,json,os,pathlib,platform,sys,torch
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert str(torch.__version__)=='2.8.0+cu128','Use the existing verified university PyTorch 2.8.0+cu128 environment.'
assert not torch.cuda.is_initialized()
# Keep the probe on the actual run filesystem, including a write-only control.
run=pathlib.Path(sys.argv[1]).parent
lock_file=run/'checkpoint-lock-probe.lock'
original=b'checkpoint lock probe; preserve contents\n'
lock_file.write_bytes(original)
checks={}
for access in ('a','a+'):
    for kind,operation in [('shared',fcntl.LOCK_SH),('exclusive',fcntl.LOCK_EX)]:
        with lock_file.open(access) as handle:
            try:
                fcntl.flock(handle,operation|fcntl.LOCK_NB)
            except OSError as error:
                checks[access+' '+kind]={'acquired':False,'errno':error.errno,'error':str(error)}
            else:
                checks[access+' '+kind]={'acquired':True}
                fcntl.flock(handle,fcntl.LOCK_UN)
probe={'schema_version':1,'filesystem_path':str(run.resolve()),'checks':checks,
       'contents_unchanged':lock_file.read_bytes()==original}
(run/'lock-probe.json').write_text(json.dumps(probe,indent=2,sort_keys=True)+'\n')
assert probe['contents_unchanged'] and all(checks['a+ '+kind]['acquired'] for kind in ('shared','exclusive')), 'Checkpoint read/write locks failed on the actual run filesystem.'
print('Checkpoint lock access probe:',json.dumps(probe,sort_keys=True))
import pytest,numpy,timm,einops,yaml
pathlib.Path(sys.argv[1]).write_text(json.dumps({'python':platform.python_version(),'torch':str(torch.__version__),
    'numpy':numpy.__version__,'pytest':pytest.__version__,'architecture':platform.machine(),
    'cpu_only':True,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
    'torch_build':torch.__config__.show()},indent=2,sort_keys=True)+'\n')
PY
  timeout --signal=TERM --kill-after=10s 1500 "$py" -m pytest -q -rs -p no:cacheprovider \
    --basetemp "$run/pytest" --junitxml="$run/tests.xml" \
    tests/test_fixture_state.py tests/test_native_diagnostic.py tests/test_precision_patch.py \
    tests/test_shared_execution.py tests/test_measured_budget.py \
    tests/test_checkpoint_locking.py tests/test_training.py tests/test_evaluation_checkpoint.py
  verify_sources
  "$py" - "$run/tests.xml" "$base/CPU-FIXTURE-MANIFEST.json" <<'PY'
import json,sys,xml.etree.ElementTree as ET
manifest=json.load(open(sys.argv[2]))
tree=ET.parse(sys.argv[1]).getroot();suites=list(tree.iter('testsuite'))
assert len(suites)==1
stats=suites[0].attrib
assert int(stats['tests'])==manifest['expected_tests'] and stats['errors']==stats['failures']=='0'
skips=[c.attrib['classname']+'::'+c.attrib['name'] for c in tree.iter('testcase') if c.find('skipped') is not None]
assert not skips and stats['skipped']=='0','Every selected CPU repair check must run.'
print('PASS: all selected CPU fixture/recovery/provenance checks ran, with zero skips.')
PY
  exit
fi
[[ $# == 0 ]] || { echo 'Usage: bash scripts/university_cpu_fixtures.sh' >&2; exit 2; }
for tool in git timeout nohup; do command -v "$tool" >/dev/null; done
verify_sources
if [[ ! -e "$base/external/vjepa2" ]]; then
  git clone --quiet "$base/upstream.bundle" "$base/external/vjepa2"
  git -C "$base/external/vjepa2" checkout --quiet --detach 204698b45b3712590f06245fbfba32d3be539812
fi
[[ $(git -C "$base/external/vjepa2" rev-parse HEAD) == 204698b45b3712590f06245fbfba32d3be539812 ]] || exit 1
mkdir -p "$base/runs"
run=$(mktemp -d "$base/runs/cpu-fixtures.XXXXXX")
nohup timeout --signal=TERM --kill-after=15s 1680 bash "$base/scripts/university_cpu_fixtures.sh" \
  --worker "$run" > "$run/launcher.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run/launcher.pid"
echo "Started detached CPU-only fixture check on $(hostname)."
echo "Run directory: $run"
echo "Watch: tail -f '$run/launcher.log'"
echo "Result: $run/session.json"
echo "Export: $run/evidence.tar.gz"
echo 'No GPU is exposed to this job. SSH can be closed while it runs.'

"""Idempotently install one verified shared-home packet; no job starts here."""
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    expected=sys.argv[1]
    if socket.gethostname() not in ('linux10609','linux10603'):raise ValueError('Wrong university host.')
    body=sys.stdin.buffer.read(1024*1024+1)
    if len(body)>1024*1024 or hashlib.sha256(body).hexdigest()!=expected:raise ValueError('Upload changed.')
    payload=json.loads(body)
    if payload['kind']!='matched-panel-two-host-source-upload-v1':raise ValueError('Wrong source upload.')
    archive=base64.b64decode(payload['archive_base64'],validate=True);manifest=payload['manifest']
    if len(archive)!=manifest['bytes'] or hashlib.sha256(archive).hexdigest()!=manifest['sha256']:raise ValueError('Archive changed.')
    transfer=Path.home()/'Desktop/VJEPA-Robust-Overfitting/matched-panel-transfer-20261024'
    extractor=transfer/'verify_extract.py'
    if digest(extractor)!='efe7ae73e955eca1660e66920f1cc78ccad7ef42ede71933dddbce661008bda1':raise ValueError('Extractor changed.')
    os.umask(0o077);target=transfer/'two-host-execution-20261026'
    try:target.mkdir(exist_ok=False);owner=True
    except FileExistsError:owner=False
    if owner:
        (target/'operator.tar.gz').write_bytes(archive)
        (target/'operator.manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
        subprocess.run([sys.executable,'-B',str(extractor),'--archive',str(target/'operator.tar.gz'),
            '--manifest',str(target/'operator.manifest.json'),'--destination',str(target/'unpacked')],check=True,timeout=30)
        with (target/'.READY.tmp').open('x') as stream:
            stream.write(manifest['sha256']+'\n');stream.flush();os.fsync(stream.fileno())
        os.replace(target/'.READY.tmp',target/'READY')
    else:
        for _ in range(300):
            if (target/'READY').is_file():break
            time.sleep(.1)
        else:raise ValueError('An incomplete shared installation exists; preserve it for inspection.')
    if target.is_symlink() or (target/'READY').read_text().strip()!=manifest['sha256']:
        raise ValueError('Existing installation identity differs.')
    if digest(target/'operator.tar.gz')!=manifest['sha256']:raise ValueError('Stored upload changed.')
    for name,entry in manifest['files'].items():
        relative=Path(name);path=target/'unpacked'/relative
        if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
            or not path.resolve().is_relative_to((target/'unpacked').resolve())
            or path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']):
            raise ValueError('Shared packet file changed: '+name)
    print('Shared two-host packet installed/verified. No GPU job started.',flush=True)

if __name__=='__main__':main()

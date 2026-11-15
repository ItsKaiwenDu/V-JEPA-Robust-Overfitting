"""Private fake-proc files and fake handles ONLY. No process or signal is real."""
import copy
import os
from pathlib import Path
import signal
from types import SimpleNamespace as NS
import pytest
from models import predictor_pgd_split_watch_v1 as watch


def process(root,pid,parent,*,birth=71,session=901,group=None,rss=1024,children=(),extra_thread=None):
    folder=root/str(pid);folder.mkdir(parents=True,exist_ok=True)
    fields=['0']*25;fields[0]='S';fields[1]=str(parent);fields[2]=str(session if group is None else group);fields[3]=str(session);fields[19]=str(birth)
    (folder/'stat').write_text(str(pid)+' (fake ) thread) '+' '.join(fields)+'\n')
    (folder/'smaps_rollup').write_text('0-0 ---p 0000 00:00 0 [rollup]\nRss: '+str(rss)+' kB\nPss: 0 kB\n')
    task=folder/'task'/str(pid);task.mkdir(parents=True,exist_ok=True);(task/'children').write_text(' '.join(map(str,children)))
    if extra_thread:
        task=folder/'task'/str(extra_thread[0]);task.mkdir();(task/'children').write_text(' '.join(map(str,extra_thread[1])))
    return folder


def test_proc_reader_uses_all_threads_recurses_descendants_and_sums_resident_bytes(tmp_path):
    process(tmp_path,901,os.getpid(),rss=1024,extra_thread=(905,[902]))
    process(tmp_path,902,901,birth=72,rss=2048,children=[903],session=902)
    process(tmp_path,903,902,birth=73,rss=4096,session=902)
    reader=watch.ProcReader(tmp_path);value=reader.sample(901,71)
    assert set(value['processes'])=={901,902,903} and value['rss_bytes']==(1024+2048+4096)*1024
    assert value['processes'][902]['session_pid']==902


@pytest.mark.parametrize('fault',['birth','pid','rss-missing','rss-unit','rss-duplicate','rss-negative','foreign-parent','self-cycle','oversized'])
def test_proc_reader_refuses_inconsistent_or_missing_memory_observations(tmp_path,fault):
    folder=process(tmp_path,901,os.getpid());reader=watch.ProcReader(tmp_path)
    if fault=='birth':
        with pytest.raises(ValueError,match='birth'):reader.sample(901,72)
        return
    if fault=='pid':(folder/'stat').write_text((folder/'stat').read_text().replace('901 (','999 ('))
    elif fault=='rss-missing':(folder/'smaps_rollup').write_text('Pss: 10 kB\n')
    elif fault=='rss-unit':(folder/'smaps_rollup').write_text('Rss: 10 MB\n')
    elif fault=='rss-duplicate':(folder/'smaps_rollup').write_text('Rss: 10 kB\nRss: 10 kB\n')
    elif fault=='rss-negative':(folder/'smaps_rollup').write_text('Rss: -10 kB\n')
    elif fault=='foreign-parent':
        (folder/'task/901/children').write_text('902');process(tmp_path,902,777,birth=72)
    elif fault=='self-cycle':(folder/'task/901/children').write_text('901')
    else:(folder/'stat').write_text('X'*16385)
    with pytest.raises(ValueError):reader.sample(901,71)


def test_normal_process_state_changes_do_not_change_its_birth_identity(tmp_path,monkeypatch):
    process(tmp_path,901,os.getpid());reader=watch.ProcReader(tmp_path);original=reader.identity;calls=[]
    def changing(pid):
        value=original(pid);value['state']='R' if not calls else 'S';calls.append(pid);return value
    monkeypatch.setattr(reader,'identity',changing)
    assert reader.sample(901,71)['rss_bytes']==1024*1024


def test_process_owner_mismatch_refuses_before_reading_other_user_memory(tmp_path,monkeypatch):
    process(tmp_path,901,os.getpid());uid=os.getuid();monkeypatch.setattr(watch.os,'getuid',lambda:uid+1)
    with pytest.raises(ValueError,match='same-user'):watch.ProcReader(tmp_path).identity(901)


class FakeHandleAPI:
    def __init__(self):self.opened=[];self.closed=[];self.sent=[];self.alive={}
    def open(self,pid):
        handle=('FAKE-PIDFD',pid,len(self.opened));self.opened.append(handle);self.alive[handle]=True;return handle
    def dead(self,handle):return not self.alive[handle]
    def send(self,handle,sig):self.sent.append((handle,sig));self.alive[handle]=False
    def close(self,handle):self.closed.append(handle)


def test_pid_handle_stays_bound_to_owned_birth_and_never_sends_numeric_pid(tmp_path):
    process(tmp_path,901,os.getpid());reader=watch.ProcReader(tmp_path);api=FakeHandleAPI()
    lease=watch.PidfdLease(901,71,os.getpid(),901,reader,api=api)
    lease.send(signal.SIGTERM);lease.close()
    assert api.sent==[(api.opened[0],signal.SIGTERM)] and api.closed==api.opened
    assert api.sent[0][0][0]=='FAKE-PIDFD'
    with pytest.raises(ValueError,match='handle'):lease.send(signal.SIGKILL)


def test_pid_reuse_during_handle_open_refuses_and_closes_without_a_signal(tmp_path):
    process(tmp_path,901,os.getpid());reader=watch.ProcReader(tmp_path);api=FakeHandleAPI();opener=api.open
    def changed(pid):
        handle=opener(pid);process(tmp_path,901,os.getpid(),birth=72);return handle
    api.open=changed
    with pytest.raises(ValueError,match='birth'):watch.PidfdLease(901,71,os.getpid(),901,reader,api=api)
    assert not api.sent and api.closed==api.opened


def test_foreign_parent_cannot_be_leased_or_signaled(tmp_path):
    process(tmp_path,901,777);reader=watch.ProcReader(tmp_path);api=FakeHandleAPI()
    with pytest.raises(ValueError,match='parent'):watch.PidfdLease(901,71,os.getpid(),901,reader,api=api)
    assert not api.sent and not api.opened


class FakeClock:
    def __init__(self):self.value=10.
    def __call__(self):return self.value
    def pause(self,seconds):assert 0<=seconds<=.1;self.value+=seconds


class Lease:
    def __init__(self,world,pid,birth,parent,session):
        self.world=world;self.pid=pid;self.birth=birth;self.parent=parent;self.session=session;self.closed=False
        self.alive=True;world.opened.append(self)
    def dead(self):assert not self.closed;return not self.alive
    def send(self,sig):
        assert not self.closed and sig in (signal.SIGTERM,signal.SIGKILL)
        self.world.sent.append((self.pid,self.birth,sig));self.alive=False
    def close(self):assert not self.closed;self.closed=True


class World:
    def __init__(self,fault=None):
        self.fault=fault;self.clock=FakeClock();self.opened=[];self.sent=[];self.gates=0;self.samples=0;self.waits=0
        self.root=Lease(self,901,71,os.getpid(),901);self.code=None;self.pid=901
    def poll(self):
        if self.fault!='timeout' and self.clock.value>=10.29:
            self.code=1 if self.fault=='exit' else 0;self.root.alive=False
            for lease in self.opened:
                if self.fault!='survivor':lease.alive=False
        return self.code
    def wait(self,timeout):assert 0<=timeout<=.1;self.waits+=1;return self.code
    def gate(self):
        self.gates+=1
        if self.fault=='gate' and self.gates==2:raise RuntimeError('SIMULATED storage/GPU gate rejection')
    def sample(self,pid,birth):
        assert (pid,birth)==(901,71);self.samples+=1
        rows={901:{'pid':901,'start_ticks':71,'parent_pid':os.getpid(),'session_pid':901,'rss_bytes':512*2**20},
              902:{'pid':902,'start_ticks':72,'parent_pid':901,'session_pid':902,'rss_bytes':64*2**20}}
        if self.fault=='rss':rows[901]['rss_bytes']=4*2**30
        if self.fault=='rss-type':rows[901]['rss_bytes']=True
        if self.fault=='foreign':rows[902]['parent_pid']=777
        if self.fault=='root-reuse':rows[901]['start_ticks']=88
        if self.fault=='missing-root':rows.pop(901)
        if self.fault=='clock':self.clock.value-=1
        return {'processes':rows,'rss_bytes':sum(r['rss_bytes'] for r in rows.values())}
    def lease(self,pid,birth,parent,session):return Lease(self,pid,birth,parent,session)
    def observe(self,**overrides):
        options={'sample':self.sample,'lease_factory':self.lease,'gate':self.gate,'phase_deadline':10.5,
            'work_deadline':11.,'launch_started':10.,'clock':self.clock,'pause':self.clock.pause}
        options.update(overrides)
        return watch.watch(self,self.root,**options)


def test_parent_supervises_complete_lifecycle_and_releases_every_fake_process_handle(tmp_path):
    w=World();receipt=w.observe()
    assert receipt['parent_observation_passed'] and receipt['RSS_samples']>=2 and receipt['observed_process_identities']==2
    assert receipt['peak_observed_RSS_bytes']==(512+64)*2**20 and receipt['wall_seconds']<.5
    assert not receipt['sampling_is_instantaneous_kernel_or_address_space_quota'] and not receipt['proc_child_enumeration_is_exhaustive']
    assert not receipt['scientific_qualified'] and not w.sent and all(h.closed for h in w.opened) and w.waits==1
    (tmp_path/'FAKE-PARENT-OBSERVATION.json').write_text(__import__('json').dumps({'origin':'FAKE-PROC-HANDLE-SIMULATION','receipt':receipt})+'\n')


@pytest.mark.parametrize('fault',['timeout','rss','rss-type','foreign','root-reuse','missing-root','clock','gate','exit','survivor'])
def test_parent_failure_never_retries_or_signals_an_unowned_identity(fault):
    w=World(fault)
    with pytest.raises((ValueError,RuntimeError,TimeoutError,MemoryError)) as caught:w.observe()
    needles={'timeout':'deadline','rss':'RSS exceeds','rss-type':'Typed observed','foreign':'Unowned','root-reuse':'root missing',
        'missing-root':'root missing','clock':'deadline','gate':'SIMULATED storage/GPU','exit':'child failed','survivor':'surviving'}
    assert needles[fault] in str(caught.value),str(caught.value)
    assert all(h.closed for h in w.opened) and all((pid,birth) in ((901,71),(902,72)) for pid,birth,sig in w.sent)
    assert not any(birth==88 or pid==777 for pid,birth,sig in w.sent)
    if fault=='foreign':assert all(h.pid!=902 for h in w.opened)
    assert w.samples<8 and w.gates<10


@pytest.mark.parametrize('change',[{'phase_deadline':130.1},{'phase_deadline':9.},{'work_deadline':10.2}])
def test_invalid_or_extended_deadline_still_closes_and_stops_the_owned_launch(change):
    w=World('timeout')
    with pytest.raises(ValueError,match='deadlines'):w.observe(**change)
    assert w.sent and all(pid==901 and birth==71 for pid,birth,sig in w.sent) and w.root.closed

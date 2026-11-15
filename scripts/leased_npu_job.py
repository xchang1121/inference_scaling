"""Linux host supervisor: dynamically lease one idle NPU for one bounded batch.

Only containers carrying this study's owner/task/job labels may be stopped.
Host/device locks span startup, execution and verified cleanup, even if the SSH
client disconnects. A job is immutable and never automatically rerun on failure.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import time

TASK = 'online-injection-tuning-20261115'
OWNER = 'changxiangyu'
ROOT = Path('/data/disk/changxiangyu/agentic_scaling')


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, timeout=45, **kwargs)


def inspect(name):
    return json.loads(command(['docker', 'inspect', name], stderr=subprocess.PIPE))[0]


def inventory():
    text = command(['/usr/local/bin/npu-smi', 'info'])
    healthy = {int(x) for x in re.findall(r'\|\s*(\d+)\s+910\w*\s*\|\s*OK\s*\|', text)}
    free = {int(x) for x in re.findall(r'No running processes found in NPU (\d+)', text)}
    reserved = set()
    for name in command(['docker', 'ps', '--format', '{{.Names}}']).split():
        try:
            item = inspect(name)
        except subprocess.CalledProcessError as error:
            # A listed ephemeral container can disappear before inspect. Only
            # this explicit absence is safe to skip; other failures stay closed.
            if error.returncode == 1 and 'No such object:' in (error.stderr or ''):
                continue
            raise
        devices = {int(d['PathOnHost'].removeprefix('/dev/davinci'))
                   for d in item['HostConfig']['Devices'] or []
                   if re.fullmatch(r'/dev/davinci\d+', d['PathOnHost'])}
        if 0 < len(devices) < 8:
            reserved.update(devices)
    return sorted((free & healthy) - reserved), text


def acquire(lockdir, spec):
    lockdir.mkdir(exist_ok=True, parents=True)
    with (lockdir/'allocation.lock').open('a') as allocation:
        fcntl.flock(allocation, fcntl.LOCK_EX)
        memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        if (memory['MemFree'] < spec.get('minimum_host_free_kib', 32*1024*1024)
                or memory['MemAvailable'] < spec.get('minimum_host_available_kib', 64*1024*1024)):
            return None
        available, before = inventory()
        for device in available:
            lock = (lockdir/f'npu-{device}.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            # Recheck after acquiring the persistent lease, before container start.
            try:
                still_available = device in inventory()[0]
            except BaseException:
                lock.close()
                raise
            if not still_available:
                lock.close()
                continue
            return device, lock, before
    return None


def stop_owned(name, job_id):
    info = inspect(name)
    expected = {'inference_scaling.owner': OWNER, 'inference_scaling.task': TASK,
                'inference_scaling.job': job_id}
    if any(info['Config']['Labels'].get(k) != v for k, v in expected.items()):
        raise RuntimeError('refusing cleanup of a container with different ownership')
    if info['State']['Running']:
        try:
            command(['docker', 'stop', '--time', '5', name])
        except (subprocess.SubprocessError, OSError):
            command(['docker', 'kill', name])
    info = inspect(name)
    if info['State']['Running']:
        raise RuntimeError('owned container is still running after cleanup')
    return info['State']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    job_id = spec['job_id']
    if not re.fullmatch(r'[a-z0-9-]{1,60}', job_id):
        raise ValueError('unsafe job id')
    out = ROOT/'results'/TASK/job_id
    out.mkdir(exist_ok=True, parents=True)
    status_path = out/'status.json'
    state = {'job_id': job_id, 'queued_at': time.time(), 'phase': 'queued', 'pid': os.getpid()}

    def save(**changes):
        state.update(changes)
        temporary = out/'status.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(status_path)

    def interrupted(signum, frame):
        raise RuntimeError(f'job interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (out/'job.lock').open('a') as job_lock:
        fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists():
            raise RuntimeError('immutable job already started; inspect it, never relaunch')
        save()
        lease = None
        try:
            while lease is None:
                lease = acquire(ROOT/'deployment/ascend/npu-leases', spec)
                if lease is None:
                    if time.time()-state['queued_at'] > spec.get('queue_timeout', 300):
                        save(phase='no_capacity', finished_at=time.time())
                        return
                    time.sleep(15)
        except BaseException as error:
            save(phase='allocation_failed', finished_at=time.time(), error=repr(error),
                 leased_seconds=0, lease_released=True, container_created=False)
            raise
        device, lock, before = lease
        name = f'agentic-scaling-{OWNER}-tune-{job_id}'
        created = False
        result = 1
        try:
            save(phase='leased', device=device, leased_at=time.time(), container=name)
            (out/'npu-before.txt').write_text(before)
            old = inspect(spec['template'])
            if not spec['template'].startswith(f'agentic-scaling-{OWNER}-'):
                raise ValueError('only own deployment templates may be reused')
            expected_image = spec.get('image')
            if expected_image and old['Image'] != expected_image:
                raise RuntimeError('environment image mismatch')
            uid, gid = map(int, old['Config']['User'].split(':'))
            os.chown(out, uid, gid)
            cmd = ['docker', 'create', '--init', '--name', name, '--user', old['Config']['User'],
                   '--group-add', '1001', '--shm-size', '34359738368',
                   '--label', f'inference_scaling.owner={OWNER}',
                   '--label', f'inference_scaling.task={TASK}',
                   '--label', f'inference_scaling.job={job_id}',
                   '-e', 'OMP_NUM_THREADS=4', '-e', 'VLLM_WORKER_MULTIPROC_METHOD=spawn',
                   '-e', f'TUNING_NPU_DEVICE={device}']
            for dp in [f'/dev/davinci{device}', '/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc']:
                cmd += ['--device', dp+':'+dp]
            for mount in old['Mounts']:
                src = str(out) if mount['Destination'] == '/results' else mount['Source']
                cmd += ['--mount', 'type=bind,source='+src+',target='+mount['Destination']
                        +('' if mount['RW'] else ',readonly')]
            cmd += [old['Image'], 'sleep', 'infinity']
            command(cmd)
            created = True
            command(['docker', 'start', name])
            setup = 'from pathlib import Path;import os\nfor n in ["/ascend","/.config"]:\n p=Path(n);p.mkdir(exist_ok=True);os.chown(p,'+str(uid)+','+str(gid)+')'
            command(['docker', 'exec', '--user', '0:0', name, 'python3', '-c', setup])
            save(phase='running', started_at=time.time(), image=old['Image'])
            with (out/'run.log').open('wb') as log:
                proc = subprocess.run(['docker', 'exec', name, *spec['command']], stdout=log,
                                      stderr=subprocess.STDOUT, timeout=spec.get('run_timeout', 3600))
                result = proc.returncode
        except BaseException as error:
            save(error=repr(error))
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            save(phase='releasing', exit_code=result, execution_finished_at=time.time())
            if created:
                save(container_state=stop_owned(name, job_id))
            available, after = inventory()
            (out/'npu-after.txt').write_text(after)
            save(phase='complete' if result == 0 else 'failed', finished_at=time.time(),
                 device_free_after=device in available, lease_released=True)
            lock.close()


if __name__ == '__main__':
    main()

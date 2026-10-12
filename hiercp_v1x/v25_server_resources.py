"""Read-only live PBS lease guard; it never signals jobs or changes services."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time


class AllocationLease:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.bytes = self.path.read_bytes()
        self.request = json.loads(self.bytes)
        value = self.request
        if (value.get('format') != 'v25_owned_GPU_allocation_v1'
                or value.get('UID') != os.getuid() or value.get('owner') != 'aicompetition06'
                or value.get('node') != 'ece-a6gpu6' or type(value.get('gpu')) is not int
                or not re.fullmatch(r'\d+\.ECE-util1',value.get('job_id',''))
                or not isinstance(value.get('GPU_UUID'),str) or not value['GPU_UUID'].startswith('GPU-')
                or len(set(value.get('CPU_affinity',[]))) < 4
                or type(value.get('expires_at_unix')) not in (int,float)):
            raise ValueError('Genuine own GPU/PBS/CPU/expiry allocation proof required')
        self.last_verified = None

    def verify(self, *, force=False):
        now = time.time()
        value = self.request
        if self.path.read_bytes() != self.bytes:
            raise ValueError('The admitted PBS allocation request changed')
        if value['expires_at_unix']-now < 180:
            raise RuntimeError('PBS lease has less than180seconds; preserve exact checkpoint and resume under next valid lease')
        if not set(os.sched_getaffinity(0)) <= set(value['CPU_affinity']):
            raise ValueError('Process CPU affinity exceeds actual PBS allocation')
        if not force and self.last_verified is not None and now-self.last_verified < 30:
            return
        job = subprocess.run(['/opt/pbs/bin/qstat','-f',value['job_id']],capture_output=True,text=True,timeout=20)
        body = job.stdout
        if (job.returncode or not re.search(r'job_state\s*=\s*R\b',body)
                or not re.search(r'Job_Owner\s*=\s*'+re.escape(value['owner'])+'@',body)
                or value['node'] not in body or not re.search(r'Resource_List\.ngpus\s*=\s*1\b',body)):
            raise RuntimeError('Own running single-GPU PBS reservation not verified: '+job.stderr.strip())
        gpu = subprocess.run(['nvidia-smi','--id='+str(value['gpu']),'--query-gpu=uuid',
            '--format=csv,noheader'],capture_output=True,text=True,timeout=15)
        if gpu.returncode or gpu.stdout.strip() != value['GPU_UUID']:
            raise ValueError('Physical GPU and allocated CUDA UUID differ')
        self.last_verified = now

    def install_scorer_guard(self, scorer):
        import types
        self.verify(force=True)
        original = scorer._encode
        def encode(instance, cpu, *, training):
            self.verify()
            return original(cpu,training=training)
        scorer._encode = types.MethodType(encode,scorer)
        return lambda: setattr(scorer,'_encode',original)

    def install_preparation_guard(self, factory):
        original = factory.V25NativeInputs._rss
        lease = self
        def checked(instance, *args, **kwargs):
            lease.verify()
            return original(instance, *args, **kwargs)
        factory.V25NativeInputs._rss = checked


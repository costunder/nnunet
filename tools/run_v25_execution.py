"""Resource and input-lifetime adapters around the unchanged frozen V25 CLI."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parallel_prepare = '--parallel-prepare' in argv
    shared_prepare = '--shared-preparation' in argv
    independent_prepare = '--independent-prepare' in argv
    if parallel_prepare:
        argv.remove('--parallel-prepare')
    if shared_prepare:
        argv.remove('--shared-preparation')
    if independent_prepare:
        argv.remove('--independent-prepare')
    if shared_prepare and independent_prepare:
        raise ValueError('Common shared preparation and independent arm-prefix preparation are distinct stages')
    from tools import run_v25 as frozen
    from hiercp_v1x import v25_server_resources as original_resources
    from hiercp_v1x.v25_direct_resources import DIRECT_FORMAT, DirectAllocationLease
    args = frozen.parse(argv)
    original_lease = original_resources.AllocationLease
    def resource_lease(path):
        declaration = json.loads(Path(path).read_text())
        if declaration.get('format') == DIRECT_FORMAT:
            guard = DirectAllocationLease(path)
            if args.mode != 'prepare':
                guard.select_cuda()
            return guard
        return original_lease(path)
    original_resources.AllocationLease = resource_lease
    adapter_files = ['tools/run_v25_execution.py', 'hiercp_v1x/v25_direct_resources.py']
    from hiercp_v1x.v25_upper_memory_runtime import install_upper_memory_runtime
    execution_files=list(adapter_files)
    if parallel_prepare:execution_files.append('hiercp_v1x/v25_parallel_preparation.py')
    if shared_prepare or independent_prepare:execution_files.append('hiercp_v1x/v25_independent_preparation.py')
    memory_runtime = install_upper_memory_runtime(runtime_files=(ROOT/name for name in execution_files))
    adapter_files.append('hiercp_v1x/v25_upper_memory_runtime.py')
    if parallel_prepare:
        if args.mode != 'prepare' or args.arm != 'A':
            raise ValueError('One common full production preparation belongs to arm A')
        manifest = Path(os.environ['V25_PARALLEL_EXECUTION_MANIFEST']).resolve(strict=True)
        from hiercp_v1x import v25_preparation
        from hiercp_v1x.v25_parallel_preparation import prepare_local_parallel
        def prepare(instance):
            return prepare_local_parallel(instance, manifest)
        v25_preparation.AcceleratedNativeInputs.prepare_local = prepare
        adapter_files.append('hiercp_v1x/v25_parallel_preparation.py')
    if shared_prepare or independent_prepare:
        if args.mode != 'prepare' or (shared_prepare and args.arm != 'A'):
            raise ValueError('Shared/prefix adapters require production prepare; common stage belongs to arm A')
        from hiercp_v1x import v25_independent_preparation as independent
        if independent_prepare:
            independent.install_prefix_lock(args,frozen)
        adapter_files.append('hiercp_v1x/v25_independent_preparation.py')
    args.output.mkdir(parents=True, exist_ok=True)
    folder = args.output / 'execution'
    folder.mkdir(exist_ok=True)
    receipt = dict(format='v25_resource_only_execution_adapter_v1', arm=args.arm, mode=args.mode,
        resource_request=str(args.allocation_request),
        resource_request_sha256=hashlib.sha256(args.allocation_request.read_bytes()).hexdigest(),
        adapter_source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in adapter_files},
        frozen_entry_point=str(frozen.__file__),
        frozen_entry_point_sha256=hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest(),
        parallel_common_preparation=parallel_prepare, scientific_configuration_unchanged=True,
        shared_common_preparation_only=shared_prepare,independent_arm_prefix_preparation=independent_prepare,
        upper_memory_runtime=memory_runtime,
        existing_cache_preserved=True, debug=False, actual_PID=os.getpid(), created_at=time.time())
    frozen.publish(folder / ('launcher_' + str(time.time_ns()) + '.json'), receipt)
    if shared_prepare:
        # The live direct lease verifies the child's exact six-core set in its
        # constructor. Set the allowed arm affinity before constructing it;
        # prepare_shared subsequently repeats the full production validation.
        import psutil
        if args.cpu_affinity is not None:
            process=psutil.Process()
            if not set(args.cpu_affinity)<=set(process.cpu_affinity()):
                raise ValueError('Shared preparation CPU affinity exceeds actual allowed CPUs')
            process.cpu_affinity(args.cpu_affinity)
        return independent.prepare_shared(frozen,args,resource_lease(args.allocation_request))
    return frozen.main(argv)


if __name__ == '__main__':
    main()

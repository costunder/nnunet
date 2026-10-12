"""Resource and input-lifetime adapters around the unchanged frozen V25 CLI."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@contextmanager
def prefix_geometry_runtime(args, frozen, *, common_only=False):
    """Bind exact reusable CPU geometry before frozen CUDA initialization.

    The frozen CLI checks CUDA memory before build_runtime creates inputs.
    For those modes we create the same admitted inputs earlier and supply that
    instance to its unchanged build_runtime. Every later active-U cache miss
    therefore uses the already bound exact CPU transform, without redoing CT
    decoding or EDT. The initial common full128 builder remains unchanged.
    """
    if common_only:
        yield
        return
    from hiercp_v1x import v25_factory as factory, v25_preparation as preparation
    from hiercp_v1x import v25_prefix_geometry_reuse as reuse
    original_install = preparation.install_runtime
    original_constructor = factory.V25NativeInputs
    original_build = factory.build_runtime
    instances = []
    selected_constructor = None
    preconstructed = None
    execution_paths = tuple(ROOT/name for name in (
        'tools/run_v25_execution.py', 'hiercp_v1x/v25_prefix_geometry_reuse.py',
        'hiercp_v1x/v24_upper_reuse.py', 'hiercp_v1x/v25_terminal_descendant_resources.py'))

    def install(factory_module=factory):
        nonlocal selected_constructor, preconstructed
        if factory_module is not factory:
            raise ValueError('Exact original V25 factory module required for prefix execution')
        if selected_constructor is not None:
            raise RuntimeError('Prefix inputs runtime may only be installed once per stage')
        receipt = original_install(factory_module)
        base = factory_module.V25NativeInputs
        if base is not preparation.AcceleratedNativeInputs:
            raise ValueError('Prefix execution requires the admitted accelerated inputs class')

        class PrefixExecutionInputs(base):
            def __init__(inputs, *arguments, **keywords):
                try:
                    super().__init__(*arguments, **keywords)
                    adapter = reuse.install(inputs)
                    # Frozen admission replaces its SHA map, so retain every
                    # earlier runtime admission when pinning these sidecars.
                    paths = tuple(dict.fromkeys((*inputs._runtime_file_sha256, *execution_paths)))
                    inputs._admit_runtime_files(paths)
                    inputs.guard_source()
                    frozen.publish(args.output/'execution'/('prefix_geometry_binding_'+str(time.time_ns())+'.json'),
                        dict(format='v25_pre_CUDA_exact_prefix_geometry_binding_v1',
                            arm=args.arm, mode=args.mode, adapter_contract=adapter.contract,
                            runtime_files_sha256=dict(inputs._runtime_file_sha256),
                            bound_before_CUDA_initialization=True,
                            model_optimizer_graph_population_unchanged=True,
                            full128_cache_preserved=True, debug=False, actual_PID=os.getpid()))
                except BaseException:
                    # Constructor admission can fail before the original raw
                    # cache exists. Close complete cache owners only; partial
                    # source-admission failures retain no large CT workspace.
                    if all(hasattr(inputs,name) for name in
                            ('_lock','_raw_cache','_raw_bytes','_donors','_regions')):
                        inputs.close()
                    raise
                instances.append(inputs)

        selected_constructor = PrefixExecutionInputs
        factory_module.V25NativeInputs = selected_constructor
        if args.mode in ('calibrate', 'train', 'score-cp'):
            configuration = frozen.configuration(args.config, args.arm)
            preconstructed = selected_constructor(args.native_experiment, args.inventory,
                args.input_cache, configuration['runtime'])
        # Preserve the original CPU runtime proof used by existing saved
        # preparations and training identity. Execution bindings are separate.
        return receipt

    def build(*arguments, inputs=None, **keywords):
        if inputs is None and args.mode in ('calibrate', 'train', 'score-cp'):
            if preconstructed is None:
                raise RuntimeError('Exact prefix inputs were not bound before CUDA initialization')
            inputs = preconstructed
        return original_build(*arguments, inputs=inputs, **keywords)

    preparation.install_runtime = install
    factory.build_runtime = build
    try:
        yield
    finally:
        try:
            for inputs in reversed(instances):
                inputs.close()
        finally:
            preparation.install_runtime = original_install
            factory.build_runtime = original_build
            factory.V25NativeInputs = original_constructor


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
    from hiercp_v1x.v25_terminal_descendant_resources import install_runtime as install_terminal_descendants
    terminal_runtime = install_terminal_descendants()
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
    adapter_files = ['tools/run_v25_execution.py', 'hiercp_v1x/v25_direct_resources.py',
        'hiercp_v1x/v25_terminal_descendant_resources.py']
    if not shared_prepare:
        adapter_files.extend(('hiercp_v1x/v25_prefix_geometry_reuse.py', 'hiercp_v1x/v24_upper_reuse.py'))
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
        terminal_descendant_runtime=terminal_runtime,
        exact_full128_prefix_reuse=not shared_prepare,
        existing_cache_preserved=True, debug=False, actual_PID=os.getpid(), created_at=time.time())
    frozen.publish(folder / ('launcher_' + str(time.time_ns()) + '.json'), receipt)
    try:
        if shared_prepare:
            # The live direct lease verifies the child's exact six-core set in
            # its constructor. Apply it before creating the preparation lease.
            import psutil
            if args.cpu_affinity is not None:
                process=psutil.Process()
                if not set(args.cpu_affinity)<=set(process.cpu_affinity()):
                    raise ValueError('Shared preparation CPU affinity exceeds actual allowed CPUs')
                process.cpu_affinity(args.cpu_affinity)
            return independent.prepare_shared(frozen,args,resource_lease(args.allocation_request))
        with prefix_geometry_runtime(args, frozen, common_only=shared_prepare):
            return frozen.main(argv)
    finally:
        original_resources.AllocationLease = original_lease


if __name__ == '__main__':
    main()

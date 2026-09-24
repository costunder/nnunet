"""Full-case preparation with measured concurrency and live resource receipts."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import math
from pathlib import Path
from hiercp.preparation_runtime import Measurement, snapshot
from .contracts import write_new


def run_jobs(tasks, function, commit, workers, report_path, *, memory_per_job=0,
             benchmark_function=None, benchmark_tasks=None):
    tasks = list(tasks)
    if not tasks or (workers != 'auto' and (type(workers) is not int or workers < 1)):
        raise ValueError('Nonempty jobs and auto or positive integer workers required')
    if memory_per_job <= 0:
        raise ValueError('Full-case memory bound required before first admission')
    report_path = Path(report_path)
    report = dict(expected=len(tasks), completed=0, attempted=0, waves=[],
                  no_case_or_graph_subset=True, declared_full_input_bound=memory_per_job,
                  selection='Same complete-case workload at every measured concurrency; no output publication during probes',
                  status='running')
    bound = int(memory_per_job)
    best_width, best_rate = 1, -1.

    def safe_width():
        state = snapshot()
        safe = min(state['cpu_capacity'], math.floor(.5*state['available_memory_bytes']/bound))
        if safe < 1:
            raise MemoryError('Full case cannot fit RAM reserve; no data or model reduction')
        return safe

    def execute(batch, width, calibration):
        error = None
        measurement = Measurement()
        admission = dict(workers=width, tasks=len(batch), calibration=calibration,
                         per_job_memory_bound=bound, resources=snapshot())
        write_new(report_path.with_name(report_path.stem+f'.admission{len(report["waves"])+1}.json'), admission)
        print({'stage':'preparation_admission', 'workers':width, 'tasks':len(batch),
               'per_job_memory_bound':bound, 'calibration':calibration}, flush=True)
        with measurement:
            with ThreadPoolExecutor(max_workers=width, thread_name_prefix='v222-case') as pool:
                remaining = iter(batch)
                pending = set()
                def submit():
                    task = next(remaining, None)
                    if task is not None:
                        pending.add(pool.submit(benchmark_function if calibration else function, task))
                        if not calibration: report['attempted'] += 1
                for _ in range(width): submit()
                while pending:
                    ready, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in ready:
                        try:
                            value = future.result()
                            if not calibration:
                                commit(value); report['completed'] += 1
                            del value
                        except Exception as exc:
                            if error is None: error = exc
                        if error is None: submit()
                if error is not None: raise error
        wave = dict(measurement.report, workers=width, tasks=len(batch), calibration=calibration,
                    cases_per_second=len(batch)/measurement.report['elapsed_seconds'])
        report['waves'].append(wave)
        write_new(report_path.with_name(report_path.stem+f'.wave{len(report["waves"])}.json'), wave)
        print({'stage':'preparation_resources', 'workers':width, 'completed':report['completed'],
               'expected':len(tasks), 'cases_per_second':wave['cases_per_second'],
               'peak_rss_bytes':wave['sampled_peak_rss_bytes'], 'calibration':calibration}, flush=True)
        return wave

    try:
        safe = min(safe_width(),len(tasks))
        if workers == 'auto':
            if benchmark_function is None:
                raise ValueError('Auto workers require a side-effect-free same-workload benchmark function')
            if benchmark_tasks is None:
                # Fixed, spread-out full cases. This is resource calibration only;
                # every production task below is still executed exactly once.
                benchmark_tasks = [tasks[i*(len(tasks)-1)//max(1,safe-1)] for i in range(safe)]
            benchmark_tasks = list(benchmark_tasks)
            if not benchmark_tasks or any(task not in tasks for task in benchmark_tasks):
                raise ValueError('Benchmark must use declared full production cases')
            report['benchmark_tasks'] = benchmark_tasks
            widths=[1]
            limit=min(safe,len(benchmark_tasks))
            while widths[-1]<limit:widths.append(min(widths[-1]*2,limit))
            # Warm the same workload first; do not favor the first/last width
            # because a compressed file entered the OS cache during calibration.
            print({'stage':'preparation_benchmark_warmup','workers':limit,
                   'full_cases':benchmark_tasks},flush=True)
            with ThreadPoolExecutor(max_workers=limit) as pool:
                for value in pool.map(benchmark_function,benchmark_tasks): del value
            for width in widths:
                if width>safe_width():
                    raise MemoryError('RAM availability changed during worker calibration')
                wave=execute(benchmark_tasks,width,True)
                bound=max(bound,math.ceil((wave['sampled_peak_rss_bytes']-wave['before']['rss_bytes'])/width))
                if wave['cases_per_second']>best_rate:
                    best_width,best_rate=width,wave['cases_per_second']
            width=min(best_width,safe_width(),len(tasks))
            report['measured_best_workers']=best_width
        else:
            if workers>safe:raise MemoryError('Requested workers exceed full-case RAM/CPU admission')
            width=workers
        execute(tasks,width,False)
        report.update(status='complete', selected_workers=width,
                      per_job_memory_bound=bound)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['pending'] = len(tasks)-report['attempted']
        report['resources_after'] = snapshot()
        write_new(report_path, report)
    return report

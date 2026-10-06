"""Full-case preparation with measured concurrency and live resource receipts."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import math
from pathlib import Path
from hiercp.preparation_runtime import Measurement, snapshot
from .contracts import write_new


def run_jobs(tasks, function, commit, workers, report_path, *, memory_per_job=0):
    tasks = list(tasks)
    if not tasks or (workers != 'auto' and (type(workers) is not int or workers < 1)):
        raise ValueError('Nonempty jobs and auto or positive integer workers required')
    if memory_per_job <= 0:
        raise ValueError('Full-case memory bound required before first admission')
    report_path = Path(report_path)
    report = dict(expected=len(tasks), completed=0, attempted=0, waves=[],
                  no_case_or_graph_subset=True, declared_full_input_bound=memory_per_job,
                  selection='Measured full-case throughput; different-case waves are not an optimality proof',
                  status='running')
    bound = int(memory_per_job)
    best_width, best_rate, offset = 1, -1., 0

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
                        pending.add(pool.submit(function, task)); report['attempted'] += 1
                for _ in range(width): submit()
                while pending:
                    ready, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in ready:
                        try:
                            commit(future.result()); report['completed'] += 1
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
        width = 1 if workers == 'auto' else workers
        while offset < len(tasks):
            safe = safe_width()
            if workers != 'auto' and width > safe:
                raise MemoryError('Requested workers exceed full-case RAM/CPU admission')
            width = min(width, safe, len(tasks)-offset)
            calibration = workers == 'auto'
            size = width if calibration else len(tasks)-offset
            wave = execute(tasks[offset:offset+size], width, calibration)
            offset += size
            bound = max(bound, wave['sampled_peak_rss_bytes']-wave['before']['rss_bytes'])
            if not calibration: break
            improved = wave['cases_per_second'] > best_rate
            if improved: best_width, best_rate = width, wave['cases_per_second']
            next_width = min(width*2, safe_width()) if offset < len(tasks) else width
            if improved and next_width > width:
                width = next_width
                continue
            if offset < len(tasks):
                width = min(best_width, safe_width(), len(tasks)-offset)
                execute(tasks[offset:], width, False)
                offset = len(tasks)
            break
        report.update(status='complete', selected_workers=best_width if workers == 'auto' else workers,
                      per_job_memory_bound=bound)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['pending'] = len(tasks)-report['attempted']
        report['resources_after'] = snapshot()
        write_new(report_path, report)
    return report

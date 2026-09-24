"""Measured full-job admission and a continuously refilled CPU work queue."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import math
import threading
from contextlib import contextmanager
from hiercp.preparation_runtime import Measurement,snapshot
from .contracts import write_new

_condition=threading.Condition()
_active_cores=0

@contextmanager
def cpu_admission(cores,capacity):
    global _active_cores
    with _condition:
        while _active_cores+cores>capacity:_condition.wait()
        _active_cores+=cores
    try:yield
    finally:
        with _condition:
            _active_cores-=cores;_condition.notify_all()

def run_jobs(tasks,function,commit,workers,report_path,*,memory_per_job=0):
    tasks=list(tasks)
    if not tasks:raise ValueError('Empty preparation job list')
    capacity=snapshot()['cpu_capacity']
    # Serialize calibration samples across concurrent CP requests; otherwise
    # their CPU/RSS deltas would be attributed to the wrong job.
    with cpu_admission(capacity,capacity),Measurement() as initial:
        commit(function(tasks[0]))
    measured=max(1,int(memory_per_job),initial.report['sampled_peak_rss_bytes']-initial.report['before']['rss_bytes'])
    state=snapshot()
    safe=max(0,math.floor(.5*state['available_memory_bytes']/measured))
    cores_per_job=max(1.,initial.report['average_cpu_cores'])
    cpu_width=max(1,math.floor(state['cpu_capacity']/cores_per_job))
    width=min(cpu_width,safe,len(tasks)) if workers=='auto' else int(workers)
    if width<1 or width>safe:raise MemoryError('Full preparation jobs do not fit measured RAM headroom')
    completed=1
    weight=min(capacity,math.ceil(cores_per_job))
    def admitted(task):
        with cpu_admission(weight,capacity):return function(task)
    parallel=None
    try:
        with Measurement() as parallel:
            with ThreadPoolExecutor(max_workers=width,thread_name_prefix='v21-cpu') as executor:
                remaining=iter(tasks[1:]); pending=set()
                def submit():
                    task=next(remaining,None)
                    if task is not None:pending.add(executor.submit(admitted,task))
                for _ in range(width):submit()
                while pending:
                    ready,_=wait(pending,return_when=FIRST_COMPLETED)
                    for future in ready:
                        pending.remove(future); commit(future.result()); completed+=1; submit()
    finally:
        report={'completed':completed,'expected':len(tasks),'selected_workers':width,
                'selection':'one full-job RAM/CPU measurement; 50% available RAM reserve, bounded by measured cores per job',
                'first_job':initial.report,'parallel_jobs':parallel.report if parallel is not None else None,
                'per_job_memory_bound':measured,'declared_full_input_bound':memory_per_job,
                'continuous_refill':True,'no_case_or_graph_subset':True}
        write_new(report_path,report)
    return report

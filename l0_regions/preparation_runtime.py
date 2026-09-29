"""Bounded CPU lookahead; CUDA preparation always stays on the caller thread."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def prefetched(requests, load):
    """Preserve every request and its order, retaining at most one next result.

    A worker exception propagates through result(). Closing the iterator cancels
    work that has not started and joins the active CPU load without killing it.
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='region-lookahead')
    pending = []
    def timed(request):
        start = perf_counter()
        return load(request), perf_counter() - start
    def batches():
        iterator = iter(requests)
        first = next(iterator, None)
        if first is None:
            return
        pending.append(pool.submit(timed, first))
        request = first
        while pending:
            start = perf_counter()
            value, load_seconds = pending.pop().result()
            wait_seconds = perf_counter() - start
            following = next(iterator, None)
            if following is not None:
                pending.append(pool.submit(timed, following))
            yield request, value, load_seconds, wait_seconds
            request = following
    iterator = batches()
    try:
        yield iterator
    finally:
        iterator.close()
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)

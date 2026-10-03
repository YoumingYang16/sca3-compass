"""Bound result backlog to worker count; statistical tasks remain unchanged."""
from concurrent.futures import wait, FIRST_COMPLETED


def bounded_results(executor, jobs, *, max_pending):
    """Yield (key, result), submit replacement only AFTER consumer resumes.

    jobs is an iterable of (key, function, args, kwargs). This bounds both
    submitted futures and completed but unconsumed results. Worker count is
    an execution setting, never a case/repetition/seed selection rule.
    """
    if isinstance(max_pending,bool) or not isinstance(max_pending,int) or max_pending<1:
        raise ValueError('Positive integer max_pending required')
    iterator=iter(jobs)
    pending={}
    def submit_one():
        try:key,function,args,kwargs=next(iterator)
        except StopIteration:return False
        future=executor.submit(function,*args,**kwargs)
        pending[future]=key
        return True
    for _ in range(max_pending):
        if not submit_one():break
    try:
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            while done:
                future=done.pop()
                key=pending.pop(future)
                result=future.result()
                del future
                yield key,result
                del result
                submit_one()
    finally:
        for future in pending:future.cancel()

from concurrent.futures import Future,ThreadPoolExecutor
import pytest
from sca3_compass.robustness_dispatch import bounded_results


class Immediate:
    def __init__(self):self.submissions=0
    def submit(self,function,*args,**kwargs):
        self.submissions+=1
        future=Future()
        try:future.set_result(function(*args,**kwargs))
        except BaseException as error:future.set_exception(error)
        return future


@pytest.mark.parametrize('limit',[1,2,4])
def test_no_submission_ahead_of_consumer_budget(limit):
    executor=Immediate()
    tasks=((i,lambda value:value*value,(i,),{}) for i in range(20))
    output=bounded_results(executor,tasks,max_pending=limit)
    values={}
    for key,value in output:
        assert executor.submissions<=len(values)+limit
        values[key]=value
    assert values=={i:i*i for i in range(20)}


def test_actual_thread_executor_same_tasks_once():
    with ThreadPoolExecutor(3) as executor:
        result=dict(bounded_results(executor,((i,pow,(i,3),{}) for i in range(30)),max_pending=3))
    assert result=={i:i**3 for i in range(30)}


def test_failure_propagates_without_submitting_all_remaining():
    executor=Immediate()
    def failure():raise ArithmeticError('retained')
    with pytest.raises(ArithmeticError,match='retained'):
        list(bounded_results(executor,((i,failure,(),{}) for i in range(20)),max_pending=2))
    assert executor.submissions==2


@pytest.mark.parametrize('limit',[0,-1,True,1.5,None])
def test_invalid_limit(limit):
    with pytest.raises(ValueError):list(bounded_results(Immediate(),[],max_pending=limit))

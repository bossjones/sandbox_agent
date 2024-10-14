"""
Active Object Pattern

Overview of the Active Object Pattern
The Active Object pattern decouples method execution from method invocation for objects that each reside in their own thread of control. The goal is to introduce concurrency by using asynchronous method invocation and a scheduler for handling requests.

The pattern consists of six elements:
1. Proxy: It provides an interface towards clients with methods that encapsulate the creation of a Request instance and its enqueuing in the ActivationList.
2. MethodRequest: It represents each method to be invoked on the Servant.
3. ActivationList: It contains MethodRequest objects that the Scheduler executes.
4. Scheduler: It removes MethodRequest objects from the ActivationList and invokes the corresponding operation on the Servant.
5. Servant: It defines the context for the method execution.
6. Future: It provides a placeholder for the result of a MethodRequest.

Implementing the Active Object Pattern in Python
Python does not have a built-in mechanism to directly support the Active Object pattern. Nevertheless, using concurrent.futures module and implementing a simple task queue, we can realize a form of this pattern:
"""

from __future__ import annotations

import concurrent.futures
import queue
import time


class ActiveObject:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.queue = queue.Queue()

    def add_task(self, task, *args, **kwargs):
        future = self.executor.submit(task, *args, **kwargs)
        self.queue.put(future)
        return future

    def get_result(self, future):
        return future.result()


# Client code:
ao = ActiveObject()


def long_running_task(n):
    time.sleep(n)
    return "Task finished"


future = ao.add_task(long_running_task, 5)
# Continue doing other stuff...
result = ao.get_result(future)
print(result)

"""
In this example, ActiveObject represents an active object that executes tasks concurrently. add_task is used to add a new task to be executed, and get_result is used to retrieve the result of a task.

Remember, the Active Object pattern can increase the complexity of a system. Therefore, it should be used only when necessary, such as when the benefits of decoupling method invocation from execution and introducing concurrency outweigh the added complexity.
"""

"""
Guarded Suspension Pattern

Overview of the Guarded Suspension Pattern
The Guarded Suspension pattern is a design pattern that manages operations that must be suspended until a particular condition is met. In essence, the pattern protects access to an object's method with a guard condition. It's often used in conditions where we can't proceed with the computation until we have certain prerequisites satisfied.

Implementing the Guarded Suspension Pattern in Python
In Python, the threading module's Condition objects are a great way to implement the Guarded Suspension pattern. Below, you'll find a simple example where a server thread waits for a client thread to set a data_ready condition before it processes data.
"""

from __future__ import annotations

import threading


# Shared condition
condition = threading.Condition()
# Shared data
data = []
data_ready = False


def server_thread():
    global data
    global data_ready
    with condition:
        while not data_ready:
            print("Server: Waiting for data...")
            condition.wait()
        print(f"Server: Received data {data}")
        data_ready = False


def client_thread():
    global data
    global data_ready
    with condition:
        print("Client: Sending data")
        data = ["Data 1", "Data 2", "Data 3"]
        data_ready = True
        condition.notify()


# Start the server thread
threading.Thread(target=server_thread).start()

# Start the client thread
threading.Thread(target=client_thread).start()

"""
In this example, the server thread and the client thread share data and data_ready. The server thread waits for data_ready to be True before it processes data. The client thread sets data and data_ready, then notifies the server thread to start processing.

The Guarded Suspension pattern is a valuable pattern to ensure that processing is suspended until necessary conditions are met. However, always consider potential deadlocks and other synchronization issues when designing your solution.
"""

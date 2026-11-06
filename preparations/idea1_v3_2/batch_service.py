"""Trusted localhost dynamic aggregation, one native model inference owner."""
import pickle
import queue
import socket
import struct
import threading
import time
import traceback

from preparations.idea1_v3_1.server import receive


class Aggregator:
    def __init__(self, policy, protocol, maximum=1, wait_ms=30):
        self.policy, self.protocol = policy, protocol
        self.maximum, self.wait_ms = maximum, wait_ms
        self.jobs = queue.Queue()
        self.actual_batches = []

    def submit(self, request):
        event, result = threading.Event(), {}
        self.jobs.put((request, event, result, time.perf_counter()))
        event.wait()
        return result

    def run(self):
        pending = None
        while True:
            first = pending or self.jobs.get()
            pending = None
            request = first[0]
            jobs = [first]
            try:
                op = request['op']
                if op == 'predict':
                    deadline = time.perf_counter()+self.wait_ms/1000
                    while len(jobs) < self.maximum:
                        try:
                            next_job = self.jobs.get(timeout=max(0, deadline-time.perf_counter()))
                        except queue.Empty:
                            break
                        if next_job[0]['op'] != 'predict':
                            pending = next_job
                            break
                        jobs.append(next_job)
                    responses = self.policy.responses([job[0] for job in jobs])
                    self.actual_batches.append(len(jobs))
                elif op == 'health':
                    responses = [dict(protocol=self.protocol)]
                elif op == 'configure':
                    maximum = int(request['maximum'])
                    if maximum not in (1, 2, 4, 8):
                        raise ValueError('Unsupported maximum')
                    self.maximum = maximum
                    if 'wait_ms' in request:
                        wait_ms = float(request['wait_ms'])
                        if not 0 <= wait_ms <= 1000:
                            raise ValueError('Invalid aggregation wait')
                        self.wait_ms = wait_ms
                    responses = [dict(maximum=self.maximum, wait_ms=self.wait_ms, actual_batches=self.actual_batches)]
                    self.actual_batches = []
                elif op == 'control':
                    actions, raw, timing = self.policy.predict(request['records'])
                    responses = [dict(actions=actions, raw=raw, timing=timing)]
                else:
                    raise ValueError('Unknown operation')
                for job, response in zip(jobs, responses, strict=True):
                    response.update(ok=True, server_dispatch_ms=(time.perf_counter()-job[3])*1000)
                    job[2].update(response)
            except Exception:
                for job in jobs:
                    job[2].update(ok=False, error=traceback.format_exc())
            finally:
                for job in jobs:
                    job[1].set()


def serve(port, aggregator):
    threading.Thread(target=aggregator.run, daemon=True).start()
    def handle(connection):
        with connection:
            try:
                size = struct.unpack('!I', receive(connection, 4))[0]
                if size > 128*1024*1024:
                    raise ValueError('Oversized local request')
                result = aggregator.submit(pickle.loads(receive(connection, size)))
            except Exception:
                result = dict(ok=False, error=traceback.format_exc())
            payload = pickle.dumps(result, protocol=5)
            connection.sendall(struct.pack('!I', len(payload))+payload)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', port))
        server.listen(32)
        while True:
            connection, _ = server.accept()
            threading.Thread(target=handle, args=(connection,), daemon=True).start()

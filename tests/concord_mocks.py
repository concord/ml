from __future__ import division, print_function

from collections import namedtuple

Record = namedtuple("Record", ["key", "data", "userStream"])

class Runner(object):
    def __init__(self, computations):
        self.computations = {c.name: c for c in computations}
        self.contexts = {c.name: ComputationContext(self, c)
                         for c in computations}

    def run(self):
        for ctx in self.contexts.values():
            ctx.computation.init(ctx)

class ComputationContext(object):
    def __init__(self, runner, computation):
        self.runner = runner
        self.computation = computation
        self.subscribers = [stream
                            for stream in computation.metadata().ostreams
                            if stream is not None]
        self.records = []

    def __str__(self):
        return "ComputationContext({})".format(self.computation.name)

    def produce_record(self, stream, key, data):
        assert isinstance(key, str)   # TODO Py3 compat
        assert isinstance(data, str)  # TODO Py3 compat
        record = Record(key, data, stream)
        self.records.append(record)

        for subscriber in self.subscribers:
            ctx = self.runner.contexts[subscriber]
            computation = self.runner.computations[subscriber]

            computation.process_record(ctx, record)

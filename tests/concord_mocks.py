from __future__ import division, print_function

from collections import namedtuple

Record = namedtuple("Record", ["key", "data", "userStream"])

class ComputationContext(object):

    def __init__(self, computation):
        self.computation = computation
        self.records = {name: [] for name in computation.metadata().ostreams}

    def __str__(self):
        return "ComputationContext({})".format(self.computation.name)

    def produce_record(self, stream, key, data):
        assert stream in self.computation.metadata().ostreams
        assert isinstance(key, str)     # TODO Py3 compat w/ bytes
        assert isinstance(data, str)

        self.records[stream].append(Record(key, data, stream))

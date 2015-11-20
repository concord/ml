from collections import namedtuple

Timer = namedtuple("Time", ["key", "time"])
Record = namedtuple("Record", ["stream", "key", "data"])

class ComputationContextMock(object):
    def __init__(self):
        self.records = []
        self.timers = []

    def produce_record(self, stream, key, data):
        self.records.append(Record(stream, key, data))

    def set_timer(self, key, time):
        self.timers.append(Timer(key, time))

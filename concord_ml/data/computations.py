import time

from concord.computation import Computation, Metadata


def current_time_millis():
    return round(time.time() * 1000)


class Generator(Computation):
    def __init__(self, iterable, name, ostreams, time_delta=500):
        self.ostreams = ostreams
        self.name = name
        self.time_delta = time_delta
        self.iterator = iter(iterable)

    def init(self, context):
        context.set_timer("{}_generator".format(self.name),
                          current_time_millis())

    def process_timer(self, context, key, time):
        try:
            value = next(self.iterator)
        except StopIteration:
            return

        for stream in self.ostreams:
            context.produce_record(stream, "key", value)
        context.set_timer("{}_generator".format(self.name),
                          current_time_millis() + self.time_delta)

    def process_record(self, context, record):
        raise NotImplementedError("process_record not implemented")

    def metadata(self):
        return Metadata(name=self.name, istreams=[], ostreams=self.ostreams)

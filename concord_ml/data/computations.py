import time

from concord.computation import Computation, Metadata


def current_time_millis():
    """ Returns the current time in milliseconds

    Returns: int
        The current time in milliseconds.
    """
    return int(time.time() * 1000)


class Generator(Computation):
    """Base class for fake-data generators for Concord

    We may refactor the naming and ostream stuff into a separate plumbing
    module at some point.

    Args:
        iterable: An iterable of the data this generator produces
        name: The name of the computation
              str
        ostreams: The list of streams this generator should output to
                  str
        time_delta: How often a new data point should be generator (in ms)
                    float
    """

    def __init__(self, iterable, name, ostreams, time_delta=500):
        self.ostreams = ostreams
        self.name = name
        self.time_delta = time_delta
        self.iterator = iter(iterable)

    def init(self, context):
        context.set_timer("{}_generator".format(self.name),
                          current_time_millis())

    def process_timer(self, context, key, time):
        """ Produces next data point and sets next timer
        """

        try:
            value = next(self.iterator)
        except StopIteration:
            return

        for stream in self.ostreams:
            context.produce_record(stream, time, value)
        context.set_timer("{}_generator".format(self.name),
                          current_time_millis() + self.time_delta)

    def process_record(self, context, record):
        raise NotImplementedError("process_record not implemented")

    def metadata(self):
        return Metadata(name=self.name, istreams=[], ostreams=self.ostreams)

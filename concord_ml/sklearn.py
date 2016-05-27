""" Integration with scikit-learn
"""
import json
import pandas as pd

from concord.computation import Computation, Metadata


class SklearnPipelineComputation(Computation):
    def __init__(self, name, pipeline, **kwargs):

        # Hack to emulate keyword-only arguments (waiting on Python 3)
        self.istreams = kwargs.pop("istreams", [])
        self.predict_ostream = kwargs.pop("predict_ostream", None)

        if kwargs:
            key, __ = kwargs.popitem()
            msg = "__init__() got an unexpected keword argument {0}"
            raise TypeError(msg.format(repr(key)))

        self.pipeline = pipeline
        self.name = name

    def init(self, ctx):
        # TODO: Logging
        pass

    def metadata(self):
        if self.predict_ostream is not None:
            ostreams = [self.predict_ostream]
        else:
            ostreams = []
        return Metadata(name=self.name,
                        istreams=self.istreams,
                        ostreams=ostreams)

    def process_record(self, ctx, record):
        # TODO: Real serialization
        data = pd.read_json(record.data, orient="records")
        prediction = self.pipeline.predict(data)
        ctx.produce_record(self.predict_ostream, record.key,
                           json.dumps(prediction.tolist()))

""" Integration with scikit-learn
"""
import json
import pandas as pd

from concord.computation import Computation, Metadata


class SklearnComputationMixin(Computation):
    def __init__(self, name, model, istreams, ostream):
        self.istreams = istreams
        self.ostream = ostream

        self.model = model
        self.name = name

    def init(self, ctx):
        pass # TODO: Logging

    def metadata(self):
        return Metadata(name=self.name,
                        istreams=self.istreams,
                        ostreams=[self.ostream])

    def process(self, data):
        raise NotImplementedError

    def process_record(self, ctx, record):
        # TODO: Real serialization
        data = pd.read_json(record.data, orient="records")

        prediction = self.process(data)

        ctx.produce_record(self.ostream, record.key,
                           json.dumps(prediction.tolist()))


class SklearnPredictComputation(SklearnComputationMixin):
    def process(self, data):
        return self.model.predict(data)

""" Integration with scikit-learn
"""
import json
import pandas as pd

from concord.computation import Computation, Metadata


class SklearnBase(Computation):
    def __init__(self, name, model, istreams, ostream):
        self.istreams = istreams
        self.ostream = ostream

        self.model = model
        self.name = name

    def init(self, ctx):
        pass

    def metadata(self):
        return Metadata(name=self.name,
                        istreams=self.istreams,
                        ostreams=[self.ostream])

    def process(self, data):
        raise NotImplementedError

    def process_record(self, ctx, record):
        data = pd.read_json(record.data, orient="records")

        prediction = self.process(data)

        ctx.produce_record(self.ostream, record.key,
                           json.dumps(prediction.tolist()))


class SklearnPredict(SklearnBase):
    def process(self, data):
        return self.model.predict(data)


class SklearnTransform(SklearnBase):
    def process(self, data):
        return self.model.transform(data)

""" Integration with scikit-learn
"""
import json
import pandas as pd

from concord.computation import Computation, Metadata

parameter_docstring = """

    Parameters
    ----------
    name : str
        Name of computation
    model : scikit-learn model
        scikit-learn model that the computation wraps
    istreams : Tuple[str, enum]
        List of input streams that the computation takes
    ostream : str
        Output stream
"""

class SklearnBase(Computation):
    __doc__ = """Base class for scikit-learn integration""" \
              + parameter_docstring

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
        """Runs scikit-learn model on record.

        The input data should be a JSON-decodeable bytestring of
        [ {column_name -> value} ], where each element of the list is a
        row to be classified.

        The output data is JSON-decodeable bytestring of a list, where
        each element represents the model output for the corresponding
        row.

        Parameters
        ----------
        ctx : concord.ComputationContext
            The Concord context
        record : concord.Record
            `record.data` : JSON-decodeable bytestring of List[Dict]
        """

        data = pd.read_json(record.data, orient="records")

        prediction = self.process(data)

        ctx.produce_record(self.ostream, record.key,
                           json.dumps(prediction.tolist()))


class SklearnPredict(SklearnBase):
    __doc__ = """Concord computation wrapping scikit-learn predictor""" \
              + parameter_docstring

    def process(self, data):
        return self.model.predict(data)


class SklearnTransform(SklearnBase):
    __doc__ = """Concord computation wrapping scikit-learn transformer""" \
              + parameter_docstring

    def process(self, data):
        return self.model.transform(data)

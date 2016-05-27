import json

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

import pandas as pd
import numpy as np
import pytest

from concord.computation import Metadata, StreamGrouping
from concord_ml.sklearn import SklearnPredictComputation

from concord_mocks import Runner


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=0.2,
                                                    random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


class IrisGenerator():

    def __init__(self, name, ostream):
        self.name = name
        self.ostream = ostream

    def init(self, ctx):
        data = pd.DataFrame(X_test).to_json(orient="records")
        ctx.produce_record(self.ostream, "iris-key", data)

    def metadata(self):
        return Metadata(name=self.name, istreams=[], ostreams=[self.ostream])


@pytest.mark.parametrize("cls", [SklearnPredictComputation])
def test_sklearn_constructors(cls):
    c = cls("test-1", None, istreams=[("iris-1", 1)], ostream="hello")
    m = c.metadata()
    assert isinstance(m, Metadata)
    assert m.name == "test-1"
    assert m.istreams == [("iris-1", 1)]
    assert m.ostreams == ["hello"]


def test_sklearn_predict_computation():
    generator = IrisGenerator("iris", "test-sklearn")

    istreams = [("iris", StreamGrouping.GROUP_BY)]
    computation = SklearnPredictComputation("test-sklearn", model,
                                             istreams=istreams, ostream=None)
    runner = Runner([generator, computation])
    runner.run()

    records = runner.contexts["test-sklearn"].records
    assert len(records) == 1

    record = records[0]
    assert record.key == "iris-key"
    assert np.allclose(model.predict(X_test), json.loads(record.data))

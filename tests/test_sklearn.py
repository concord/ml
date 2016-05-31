import json

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer

import numpy as np
import pandas as pd
import pytest

from concord.computation import Metadata
from concord_ml.sklearn import SklearnPredict, SklearnTransform

from concord_mocks import ComputationContext, Record


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=0.2,
                                                    random_state=42)

@pytest.mark.parametrize("cls", [SklearnPredict, SklearnTransform])
def test_sklearn_constructors(cls):
    c = cls("test-1", None, istreams=[("iris-1", 1)], ostream="hello")
    m = c.metadata()
    assert isinstance(m, Metadata)
    assert m.name == "test-1"
    assert m.istreams == [("iris-1", 1)]
    assert m.ostreams == ["hello"]


def test_sklearn_predict():
    predictor = LinearRegression()
    predictor.fit(X_train, y_train)

    computation = SklearnPredict("test-sklearn", predictor,
                                 istreams=[], ostream="out")
    context = ComputationContext(computation)

    data = pd.DataFrame(X_test).to_json(orient="records")
    computation.process_record(context, Record("predict", data, None))

    assert len(context.records) == 1
    assert len(context.records["out"]) == 1

    record = context.records["out"][0]
    assert record.key == "predict"
    assert np.allclose(predictor.predict(X_test), json.loads(record.data))


def test_sklearn_transform():
    transformer = Normalizer()
    transformer.fit(X_train)

    computation = SklearnTransform("test-sklearn", transformer,
                                   istreams=[], ostream="out")
    context = ComputationContext(computation)

    data = pd.DataFrame(X_test).to_json(orient="records")
    computation.process_record(context, Record("transform", data, None))

    assert len(context.records) == 1
    assert len(context.records["out"]) == 1

    record = context.records["out"][0]
    assert record.key == "transform"
    assert np.allclose(transformer.transform(X_test), json.loads(record.data))

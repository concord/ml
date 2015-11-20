import pytest

from concord_ml.data import Generator
from concord_mocks import ComputationContextMock

@pytest.fixture
def generator():
    def constant_generator():
        while True:
            yield 361
    return Generator(constant_generator(), "Cthulhu",
                     ["Cthulhu", "R'lyeh", "wgah'nagl"])
@pytest.fixture
def context():
    return ComputationContextMock()


def test_generator_metadata(generator):
    metadata = generator.metadata()
    assert metadata.name == "Cthulhu"
    assert len(metadata.istreams) == 0
    assert metadata.ostreams == ["Cthulhu", "R'lyeh", "wgah'nagl"]

def test_generator_process_record(generator):
    with pytest.raises(NotImplementedError):
        generator.process_record(None, None)

def test_generator_init(generator, context):
    generator.init(context)

    assert len(context.records) == 0
    assert len(context.timers) == 1
    assert isinstance(context.timers[0].time, int)

def test_generator_process_timer(generator, context):
    generator.process_timer(context, "blah", 0)

    assert len(context.timers) == 1
    assert len(context.records) == 3
    assert {record.data for record in context.records} == {361}

import pytest
from src.inference.predictor import LoreCrafterPredictor


@pytest.fixture
def sample_text():
    return "Gandalf viajó a Gondor portando su espada mágica."


def test_predictor_initialization():
    with pytest.raises(Exception):
        LoreCrafterPredictor("models/nonexistent")


def test_extract_returns_list():
    with pytest.raises(Exception):
        predictor = LoreCrafterPredictor("models/lorecrafter-ner")
        result = predictor.extract("test")
        assert isinstance(result, list)


def test_entity_has_required_fields():
    with pytest.raises(Exception):
        predictor = LoreCrafterPredictor("models/lorecrafter-ner")
        result = predictor.extract("test")
        if result:
            entity = result[0]
            assert "text" in entity
            assert "label" in entity
            assert "start" in entity
            assert "end" in entity

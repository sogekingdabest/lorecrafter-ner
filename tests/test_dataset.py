import pytest
from src.dataset.create_dataset import build_matcher, weak_label_text
import spacy


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def matcher(nlp):
    return build_matcher(nlp)


def test_matcher_finds_gandalf(nlp, matcher):
    text = "Gandalf el Gris viajó a Rivendel."
    doc, entities = weak_label_text(text, nlp, matcher)
    assert len(entities) > 0


def test_matcher_finds_location(nlp, matcher):
    text = "La batalla de Minas Tirith fue épica."
    doc, entities = weak_label_text(text, nlp, matcher)
    labels = [e[2] for e in entities]
    assert "LUGAR" in labels


def test_weak_label_returns_doc(nlp, matcher):
    text = "Aragorn empuñó a Andúril."
    doc, entities = weak_label_text(text, nlp, matcher)
    assert doc is not None
    assert isinstance(entities, list)

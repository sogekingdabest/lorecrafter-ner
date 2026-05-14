"""
Tests del predictor usando mocks para no requerir el modelo en disco.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

# Helpers ──────────────────────────────────────────────────────────────────────

FAKE_TEXT = "Gandalf viajó a Gondor portando Glamdring."


def _make_fake_encoding(text: str):
    """Crea un BatchEncoding falso con los campos que usa el predictor."""
    from transformers import BatchEncoding

    # word_ids: 3 "palabras" (0, 1, 2) mas [CLS]=None y [SEP]=None
    _word_ids = [None, 0, 1, 2, None]
    # offsets correspondientes a posiciones char del texto original
    _offsets = [(0, 0), (0, 7), (15, 21), (32, 41), (0, 0)]

    data = {
        "input_ids": torch.tensor([[101, 1, 2, 3, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        "offset_mapping": torch.tensor([_offsets]),
    }
    encoding = BatchEncoding(data)
    encoding._encodings = [MagicMock()]
    encoding._encodings[0].word_ids = _word_ids  # property, not a method
    return encoding


def _make_fake_model(label_ids: list[int], id2label: dict):
    """Crea un mock del modelo que devuelve logits con los label_ids dados."""
    n_labels = max(id2label.keys()) + 1
    # logits shape: (1, seq_len, n_labels)
    logits = torch.zeros(1, 5, n_labels)
    for token_pos, label_id in enumerate(label_ids):
        logits[0, token_pos, label_id] = 10.0  # forzar argmax

    model = MagicMock()
    model.config.id2label = id2label
    model.return_value = MagicMock(logits=logits)
    return model


# Tests ────────────────────────────────────────────────────────────────────────


@patch("src.inference.predictor.AutoModelForTokenClassification.from_pretrained")
@patch("src.inference.predictor.AutoTokenizer.from_pretrained")
def test_predictor_raises_if_model_not_found(mock_tokenizer_cls, mock_model_cls):
    """Si el modelo no existe en disco debe lanzar excepcion."""
    mock_model_cls.side_effect = OSError("Model path not found")

    from src.inference.predictor import LoreCrafterPredictor

    with pytest.raises(OSError):
        LoreCrafterPredictor("models/nonexistent")


@patch("src.inference.predictor.AutoModelForTokenClassification.from_pretrained")
@patch("src.inference.predictor.AutoTokenizer.from_pretrained")
def test_extract_returns_list(mock_tokenizer_cls, mock_model_cls):
    id2label = {0: "O", 1: "B-PERSONAJE", 2: "B-LUGAR", 3: "B-ARTEFACTO_MAGICO"}
    label_ids = [0, 1, 2, 3, 0]

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = _make_fake_encoding(FAKE_TEXT)
    mock_tokenizer_cls.return_value = mock_tokenizer

    mock_model = _make_fake_model(label_ids, id2label)
    mock_model_cls.return_value = mock_model

    from src.inference.predictor import LoreCrafterPredictor

    predictor = LoreCrafterPredictor("models/fake")
    result = predictor.extract(FAKE_TEXT)

    assert isinstance(result, list)


@patch("src.inference.predictor.AutoModelForTokenClassification.from_pretrained")
@patch("src.inference.predictor.AutoTokenizer.from_pretrained")
def test_entity_has_required_fields(mock_tokenizer_cls, mock_model_cls):
    id2label = {0: "O", 1: "B-PERSONAJE", 2: "B-LUGAR", 3: "B-ARTEFACTO_MAGICO"}
    label_ids = [0, 1, 0, 0, 0]

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = _make_fake_encoding(FAKE_TEXT)
    mock_tokenizer_cls.return_value = mock_tokenizer

    mock_model = _make_fake_model(label_ids, id2label)
    mock_model_cls.return_value = mock_model

    from src.inference.predictor import LoreCrafterPredictor

    predictor = LoreCrafterPredictor("models/fake")
    result = predictor.extract(FAKE_TEXT)

    if result:
        entity = result[0]
        assert "text" in entity
        assert "label" in entity
        assert "start" in entity
        assert "end" in entity


@patch("src.inference.predictor.AutoModelForTokenClassification.from_pretrained")
@patch("src.inference.predictor.AutoTokenizer.from_pretrained")
def test_offsets_are_char_positions(mock_tokenizer_cls, mock_model_cls):
    """Los offsets start/end deben ser posiciones de caracter, no de token."""
    id2label = {0: "O", 1: "B-PERSONAJE", 2: "O", 3: "O"}
    label_ids = [0, 1, 0, 0, 0]

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = _make_fake_encoding(FAKE_TEXT)
    mock_tokenizer_cls.return_value = mock_tokenizer

    mock_model = _make_fake_model(label_ids, id2label)
    mock_model_cls.return_value = mock_model

    from src.inference.predictor import LoreCrafterPredictor

    predictor = LoreCrafterPredictor("models/fake")
    result = predictor.extract(FAKE_TEXT)

    assert len(result) == 1
    assert result[0]["label"] == "PERSONAJE"
    assert result[0]["start"] == 0
    assert result[0]["end"] == 7


@patch("src.inference.predictor.AutoModelForTokenClassification.from_pretrained")
@patch("src.inference.predictor.AutoTokenizer.from_pretrained")
def test_all_o_labels_returns_empty(mock_tokenizer_cls, mock_model_cls):
    """Si todos los tokens son O, debe devolver lista vacia."""
    id2label = {0: "O"}
    label_ids = [0, 0, 0, 0, 0]

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = _make_fake_encoding(FAKE_TEXT)
    mock_tokenizer_cls.return_value = mock_tokenizer

    mock_model = _make_fake_model(label_ids, id2label)
    mock_model_cls.return_value = mock_model

    from src.inference.predictor import LoreCrafterPredictor

    predictor = LoreCrafterPredictor("models/fake")
    result = predictor.extract(FAKE_TEXT)

    assert result == []

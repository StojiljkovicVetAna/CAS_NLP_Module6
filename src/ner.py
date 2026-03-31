"""Classical NER utilities based on spaCy."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any

import spacy
from spacy import displacy
from spacy.language import Language

DEFAULT_MODEL_NAME = "en_core_web_sm"

# Soft, high-contrast palette for the most common spaCy NER labels.
DEFAULT_ENTITY_COLORS: dict[str, str] = {
    "PERSON": "#ffd166",
    "PER": "#ffd166",
    "ORG": "#06d6a0",
    "GPE": "#118ab2",
    "LOC": "#90be6d",
    "MISC": "#a0c4ff",
    "DATE": "#ef476f",
    "TIME": "#f9844a",
    "MONEY": "#8ecae6",
    "PERCENT": "#ffafcc",
    "NORP": "#cdb4db",
    "EVENT": "#adb5bd",
    "PRODUCT": "#f6bd60",
    "WORK_OF_ART": "#84a59d",
}


@lru_cache(maxsize=2)
def load_ner_pipeline(model_name: str = DEFAULT_MODEL_NAME) -> Language:
    """Load and cache a spaCy pipeline used for NER."""
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            "Install it via requirements.txt (recommended) or run "
            f"`python -m spacy download {model_name}`."
        ) from exc


def extract_entities(text: str, model_name: str = DEFAULT_MODEL_NAME) -> list[dict[str, Any]]:
    """Extract entities with character offsets and labels from text."""
    cleaned = text.strip()
    if not cleaned:
        return []

    doc = load_ner_pipeline(model_name)(cleaned)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        }
        for ent in doc.ents
    ]


def count_entities_by_label(entities: list[dict[str, Any]]) -> dict[str, int]:
    """Count extracted entities by NER label."""
    label_counts = Counter(entity["label"] for entity in entities)
    return dict(sorted(label_counts.items(), key=lambda item: item[0]))


def render_entities_html(
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
    entity_colors: dict[str, str] | None = None,
) -> str:
    """Render highlighted entities as HTML for Gradio."""
    cleaned = text.strip()
    if not cleaned:
        return "<p>No text provided.</p>"

    doc = load_ner_pipeline(model_name)(cleaned)
    colors = entity_colors or DEFAULT_ENTITY_COLORS
    html = displacy.render(
        doc,
        style="ent",
        options={"colors": colors},
        jupyter=False,
        page=False,
    )
    return f"<div style='padding: 0.5rem; background: white;'>{html}</div>"


def get_model_ner_labels(model_name: str = DEFAULT_MODEL_NAME) -> list[str]:
    """Return sorted NER labels available in the selected spaCy model."""
    nlp = load_ner_pipeline(model_name)
    if not nlp.has_pipe("ner"):
        return []
    ner = nlp.get_pipe("ner")
    return sorted(label for label in ner.labels)

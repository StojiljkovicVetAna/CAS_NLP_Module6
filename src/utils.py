"""Shared formatting and lightweight normalization utilities."""

from __future__ import annotations

import html as html_lib
import re
from typing import Any


def _text_color_for_background(hex_color: str) -> str:
    """Return readable text color for a given hex background."""
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return "#111827"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    return "#111827" if luminance >= 0.62 else "#ffffff"


def normalize_entities_for_display(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize LLM entities into display-safe text/label pairs."""
    normalized: list[dict[str, Any]] = []
    for item in entities:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        label = str(item.get("label", "NEW")).strip().upper().replace(" ", "_")
        label = label or "NEW"
        raw_label = str(item.get("raw_label", label)).strip().upper().replace(" ", "_")
        raw_label = raw_label or label
        confidence_raw = item.get("mapping_confidence")
        try:
            mapping_confidence = float(confidence_raw) if confidence_raw is not None else None
        except (TypeError, ValueError):
            mapping_confidence = None
        raw_new_topic = item.get("new_topic")
        if raw_new_topic is None:
            new_topic = None
        else:
            text_new_topic = str(raw_new_topic).strip()
            new_topic = None if (not text_new_topic or text_new_topic.lower() in {"none", "null"}) else text_new_topic
        normalized.append(
            {
                "text": text,
                "label": label,
                "raw_label": raw_label,
                "mapping_confidence": mapping_confidence,
                "new_topic": new_topic,
            }
        )
    return normalized


def build_llm_entity_rows(text: str, entities: list[dict[str, Any]]) -> list[list[Any]]:
    """Build table rows with approximate offsets for LLM entities."""
    rows: list[list[Any]] = []
    for entity in entities:
        entity_text = str(entity.get("text", "")).strip()
        label = str(entity.get("label", "NEW")).strip()
        raw_label = str(entity.get("raw_label", label)).strip()
        new_topic = entity.get("new_topic")
        mapping_confidence = entity.get("mapping_confidence")
        if not entity_text:
            continue
        match = re.search(re.escape(entity_text), text, flags=re.IGNORECASE)
        if match:
            start_char, end_char = match.start(), match.end()
        else:
            start_char, end_char = None, None
        rows.append(
            [
                entity_text,
                label,
                raw_label,
                new_topic,
                mapping_confidence,
                start_char,
                end_char,
            ]
        )
    return rows


def render_llm_entities_html(
    text: str,
    entities: list[dict[str, Any]],
    color_map: dict[str, str],
) -> str:
    """Render lightweight entity highlighting from LLM entity strings."""
    cleaned = text.strip()
    if not cleaned:
        return "<p>No text provided.</p>"

    spans: list[tuple[int, int, str, str]] = []
    for entity in entities:
        entity_text = entity["text"].strip()
        label = entity["label"]
        if not entity_text:
            continue
        for match in re.finditer(re.escape(entity_text), cleaned, flags=re.IGNORECASE):
            spans.append((match.start(), match.end(), label, cleaned[match.start() : match.end()]))

    if not spans:
        return "<p>No LLM entities were matched in the input text.</p>"

    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    selected: list[tuple[int, int, str, str]] = []
    current_end = -1
    for start, end, label, matched_text in spans:
        if start >= current_end:
            selected.append((start, end, label, matched_text))
            current_end = end

    fragments: list[str] = []
    cursor = 0
    for start, end, label, matched_text in selected:
        if cursor < start:
            fragments.append(html_lib.escape(cleaned[cursor:start]))

        color = color_map.get(label, "#d3d3d3")
        text_color = _text_color_for_background(color)
        fragments.append(
            "<mark class='entity' style='background: {color}; padding: 0.45em 0.6em;"
            " margin: 0 0.25em; line-height: 1; border-radius: 0.35em; color: {text_color};"
            " -webkit-text-fill-color: {text_color}; border: 1px solid rgba(17,24,39,0.15);'>"
            "{text}<span style='font-size: 0.8em; font-weight: bold; line-height: 1;"
            " border-radius: 999px; vertical-align: middle; margin-left: 0.5rem; padding: 0.1rem 0.4rem;"
            " background: rgba(17,24,39,0.14); color: inherit; -webkit-text-fill-color: inherit;'>{label}</span>"
            "</mark>".format(
                color=color,
                text_color=text_color,
                text=html_lib.escape(matched_text),
                label=html_lib.escape(label),
            )
        )
        cursor = end

    if cursor < len(cleaned):
        fragments.append(html_lib.escape(cleaned[cursor:]))

    rendered = "".join(fragments)
    return (
        "<div style='padding: 0.5rem; background: white;'>"
        f"<div class='entities' style='line-height: 2.5; direction: ltr'>{rendered}</div></div>"
    )

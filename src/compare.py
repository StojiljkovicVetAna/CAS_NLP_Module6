"""Comparison helpers between classical and LLM extraction outputs."""

from __future__ import annotations

import html as html_lib
import re
from typing import Any

# Chosen so the channel-average blend stays in a pleasant light green range.
CLASSICAL_ONLY_COLOR = "#ffe8a3"  # pastel yellow
LLM_ONLY_COLOR = "#9adfd6"  # pastel aqua
OVERLAP_MISMATCH_COLOR = "#f6b5b5"  # pastel red
LOW_IMPORTANCE_TOKENS = {
    "a",
    "an",
    "the",
    "of",
    "and",
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "ein",
    "eine",
    "einer",
    "einem",
    "einen",
    "und",
    "von",
}
RELATION_CANONICAL_MAP = {
    "is_in": "located_in",
    "in": "located_in",
    "located_at": "located_in",
    "based_in": "located_in",
    "hosted_in": "located_in",
    "starts_at": "starts_on",
    "begins_on": "starts_on",
    "ends_at": "ends_on",
}


def _normalize_text(text: str) -> str:
    """Normalize text for overlap matching."""
    lowered = text.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def _unique_sorted(values: set[str]) -> list[str]:
    """Return sorted list for stable display."""
    return sorted(values)


def _normalize_label(label: Any) -> str:
    """Normalize labels for cross-pipeline comparison."""
    normalized = str(label or "UNKNOWN").strip().upper().replace(" ", "_")
    return normalized or "UNKNOWN"


def _canonical_entity_key(text: str) -> str:
    """Build a soft-match key that ignores case, punctuation, and weak tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    filtered = [token for token in tokens if token not in LOW_IMPORTANCE_TOKENS]
    if filtered:
        return " ".join(filtered)
    return _normalize_text(text)


def _token_set(canonical_key: str) -> set[str]:
    """Split a canonical key into token set."""
    return {token for token in canonical_key.split() if token}


def _keys_soft_match(left_key: str, right_key: str) -> bool:
    """Soft key match for slight mention variants (e.g., trailing descriptor words)."""
    if left_key == right_key:
        return True

    left_tokens = _token_set(left_key)
    right_tokens = _token_set(right_key)
    if not left_tokens or not right_tokens:
        return False

    # Subset logic catches cases like "2025 2026" vs "2025 2026 edition".
    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        smaller = left_tokens if len(left_tokens) <= len(right_tokens) else right_tokens
        if len(smaller) >= 2:
            return True
        only = next(iter(smaller))
        return len(only) >= 5

    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    if not union:
        return False
    jaccard = len(intersection) / len(union)
    return jaccard >= 0.8


def _blend_hex_colors(first: str, second: str) -> str:
    """Blend two hex colors by channel average."""
    left = first.lstrip("#")
    right = second.lstrip("#")
    if len(left) != 6 or len(right) != 6:
        return "#b0b7c3"
    red = (int(left[0:2], 16) + int(right[0:2], 16)) // 2
    green = (int(left[2:4], 16) + int(right[2:4], 16)) // 2
    blue = (int(left[4:6], 16) + int(right[4:6], 16)) // 2
    return f"#{red:02x}{green:02x}{blue:02x}"


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


def _find_spans_by_term(text: str, term: str) -> list[tuple[int, int]]:
    """Find all case-insensitive spans of a term in text."""
    candidate = term.strip()
    if not candidate:
        return []
    return [(match.start(), match.end()) for match in re.finditer(re.escape(candidate), text, flags=re.IGNORECASE)]


def _spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """Return True when two half-open spans overlap."""
    return start_a < end_b and start_b < end_a


def _collect_mention_maps(
    entities: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Collect mention display text and label sets keyed by canonical mention key."""
    mention_text_map: dict[str, str] = {}
    mention_labels_map: dict[str, set[str]] = {}
    for entity in entities:
        raw_text = str(entity.get("text", "")).strip()
        if not raw_text:
            continue
        key = _canonical_entity_key(raw_text)
        mention_text_map.setdefault(key, raw_text)
        mention_labels_map.setdefault(key, set()).add(_normalize_label(entity.get("label")))
    return mention_text_map, mention_labels_map


def _extract_classical_spans(
    text: str,
    entities: list[dict[str, Any]],
) -> list[tuple[int, int, str, str]]:
    """Extract classical spans with fallback to term matching."""
    spans: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int, str, str]] = set()
    for entity in entities:
        raw_text = str(entity.get("text", "")).strip()
        if not raw_text:
            continue
        mention_key = _canonical_entity_key(raw_text)
        label = _normalize_label(entity.get("label"))
        start = entity.get("start_char")
        end = entity.get("end_char")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            item = (start, end, mention_key, label)
            if item not in seen:
                seen.add(item)
                spans.append(item)
            continue
        for span_start, span_end in _find_spans_by_term(text, raw_text):
            item = (span_start, span_end, mention_key, label)
            if item not in seen:
                seen.add(item)
                spans.append(item)
    return spans


def _extract_llm_spans(text: str, entities: list[dict[str, Any]]) -> list[tuple[int, int, str, str]]:
    """Extract LLM spans by matching extracted entity text in source text."""
    spans: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int, str, str]] = set()
    for entity in entities:
        raw_text = str(entity.get("text", "")).strip()
        if not raw_text:
            continue
        mention_key = _canonical_entity_key(raw_text)
        label = _normalize_label(entity.get("label"))
        for span_start, span_end in _find_spans_by_term(text, raw_text):
            item = (span_start, span_end, mention_key, label)
            if item not in seen:
                seen.add(item)
                spans.append(item)
    return spans


def _build_overlap_pairs(
    classical_keys: set[str],
    llm_keys: set[str],
    classical_spans: list[tuple[int, int, str, str]] | None = None,
    llm_spans: list[tuple[int, int, str, str]] | None = None,
) -> set[tuple[str, str]]:
    """Build aligned mention pairs using soft key matching and optional span overlap."""
    pairs: set[tuple[str, str]] = set()

    for classical_key in classical_keys:
        for llm_key in llm_keys:
            if _keys_soft_match(classical_key, llm_key):
                pairs.add((classical_key, llm_key))

    if classical_spans and llm_spans:
        for start_c, end_c, classical_key, _ in classical_spans:
            for start_l, end_l, llm_key, _ in llm_spans:
                if _spans_overlap(start_c, end_c, start_l, end_l):
                    pairs.add((classical_key, llm_key))

    return pairs


def _build_overlap_status_maps(
    classical_keys: set[str],
    llm_keys: set[str],
    overlap_pairs: set[tuple[str, str]],
    classical_label_map: dict[str, set[str]],
    llm_label_map: dict[str, set[str]],
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]], dict[str, set[str]], int, int]:
    """Return mention status maps and label-match/mismatch counts."""
    classical_to_llm: dict[str, set[str]] = {key: set() for key in classical_keys}
    llm_to_classical: dict[str, set[str]] = {key: set() for key in llm_keys}
    for classical_key, llm_key in overlap_pairs:
        classical_to_llm.setdefault(classical_key, set()).add(llm_key)
        llm_to_classical.setdefault(llm_key, set()).add(classical_key)

    classical_status: dict[str, str] = {}
    llm_status: dict[str, str] = {}

    overlap_label_match_count = 0
    overlap_label_mismatch_count = 0
    seen_overlap_pairs: set[tuple[str, str]] = set()

    for classical_key, llm_key in overlap_pairs:
        if (classical_key, llm_key) in seen_overlap_pairs:
            continue
        seen_overlap_pairs.add((classical_key, llm_key))
        if classical_label_map.get(classical_key, set()) & llm_label_map.get(llm_key, set()):
            overlap_label_match_count += 1
        else:
            overlap_label_mismatch_count += 1

    for classical_key in classical_keys:
        linked = classical_to_llm.get(classical_key, set())
        if not linked:
            classical_status[classical_key] = "classical"
            continue
        has_match = any(
            classical_label_map.get(classical_key, set()) & llm_label_map.get(llm_key, set())
            for llm_key in linked
        )
        classical_status[classical_key] = "overlap_match" if has_match else "overlap_mismatch"

    for llm_key in llm_keys:
        linked = llm_to_classical.get(llm_key, set())
        if not linked:
            llm_status[llm_key] = "llm"
            continue
        has_match = any(
            llm_label_map.get(llm_key, set()) & classical_label_map.get(classical_key, set())
            for classical_key in linked
        )
        llm_status[llm_key] = "overlap_match" if has_match else "overlap_mismatch"

    return (
        classical_status,
        llm_status,
        classical_to_llm,
        llm_to_classical,
        overlap_label_match_count,
        overlap_label_mismatch_count,
    )


def render_comparison_overlap_html(
    text: str,
    classical_entities: list[dict[str, Any]],
    llm_entities: list[dict[str, Any]],
) -> str:
    """Render a single text view with color-coded overlap between pipelines."""
    cleaned = (text or "").strip()
    if not cleaned:
        return "<p>No text provided.</p>"

    overlap_match_color = _blend_hex_colors(CLASSICAL_ONLY_COLOR, LLM_ONLY_COLOR)
    color_map = {
        "classical": CLASSICAL_ONLY_COLOR,
        "llm": LLM_ONLY_COLOR,
        "overlap_match": overlap_match_color,
        "overlap_mismatch": OVERLAP_MISMATCH_COLOR,
    }
    tag_map = {
        "classical": "C",
        "llm": "L",
        "overlap_match": "C+L",
        "overlap_mismatch": "C+L!",
    }
    tooltip_map = {
        "classical": "Classical-only entity mention",
        "llm": "LLM-only entity mention",
        "overlap_match": "Mention detected by both pipelines with matching label(s)",
        "overlap_mismatch": "Mention detected by both pipelines with different labels",
    }

    classical_text_map, classical_label_map = _collect_mention_maps(classical_entities)
    llm_text_map, llm_label_map = _collect_mention_maps(llm_entities)
    classical_keys = set(classical_text_map.keys())
    llm_keys = set(llm_text_map.keys())

    classical_spans = _extract_classical_spans(cleaned, classical_entities)
    llm_spans = _extract_llm_spans(cleaned, llm_entities)

    overlap_pairs = _build_overlap_pairs(
        classical_keys=classical_keys,
        llm_keys=llm_keys,
        classical_spans=classical_spans,
        llm_spans=llm_spans,
    )

    (
        classical_status,
        llm_status,
        classical_to_llm,
        llm_to_classical,
        _,
        _,
    ) = _build_overlap_status_maps(
        classical_keys=classical_keys,
        llm_keys=llm_keys,
        overlap_pairs=overlap_pairs,
        classical_label_map=classical_label_map,
        llm_label_map=llm_label_map,
    )

    raw_spans: list[tuple[int, int, str, str, str]] = []
    for start, end, mention, _ in classical_spans:
        category = classical_status.get(mention, "classical")
        raw_spans.append((start, end, category, mention, "classical"))
    for start, end, mention, _ in llm_spans:
        category = llm_status.get(mention, "llm")
        raw_spans.append((start, end, category, mention, "llm"))

    if not raw_spans:
        return "<p>No entities were available yet for visual comparison.</p>"

    # Keep stronger category when multiple sources map to same character span.
    priority = {
        "overlap_mismatch": 4,
        "overlap_match": 3,
        "classical": 2,
        "llm": 1,
    }
    span_best: dict[tuple[int, int], tuple[str, str, str]] = {}
    for start, end, category, mention, side in raw_spans:
        key = (start, end)
        if key not in span_best:
            span_best[key] = (category, mention, side)
            continue
        current_category, _, _ = span_best[key]
        if priority.get(category, 0) > priority.get(current_category, 0):
            span_best[key] = (category, mention, side)

    prepared: list[tuple[int, int, str, str, str]] = [
        (start, end, category, mention, side)
        for (start, end), (category, mention, side) in span_best.items()
    ]
    prepared.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    selected: list[tuple[int, int, str, str, str]] = []
    current_end = -1
    for start, end, category, mention, side in prepared:
        if start >= current_end:
            selected.append((start, end, category, mention, side))
            current_end = end

    cursor = 0
    parts: list[str] = []
    for start, end, category, mention, side in selected:
        if cursor < start:
            parts.append(html_lib.escape(cleaned[cursor:start]))

        surface = color_map[category]
        text_color = _text_color_for_background(surface)
        mention_text = html_lib.escape(cleaned[start:end])
        badge = tag_map[category]

        if category in {"overlap_match", "overlap_mismatch"}:
            if side == "classical":
                classical_labels = ", ".join(sorted(classical_label_map.get(mention, set()))) or "none"
                linked_llm = classical_to_llm.get(mention, set())
                llm_labels_union = {
                    label
                    for llm_key in linked_llm
                    for label in llm_label_map.get(llm_key, set())
                }
                llm_labels = ", ".join(sorted(llm_labels_union)) or "none"
            else:
                llm_labels = ", ".join(sorted(llm_label_map.get(mention, set()))) or "none"
                linked_classical = llm_to_classical.get(mention, set())
                classical_labels_union = {
                    label
                    for classical_key in linked_classical
                    for label in classical_label_map.get(classical_key, set())
                }
                classical_labels = ", ".join(sorted(classical_labels_union)) or "none"
            tooltip = (
                f"{tooltip_map[category]} | Classical labels: {classical_labels} | "
                f"LLM labels: {llm_labels}"
            )
        else:
            tooltip = tooltip_map[category]

        parts.append(
            "<mark title='{tooltip}' style='background:{surface}; color:{text_color}; "
            "-webkit-text-fill-color:{text_color}; padding:0.45em 0.6em; margin:0 0.25em; "
            "line-height:1; border-radius:0.35em; border:1px solid rgba(17,24,39,0.15);'>"
            "{mention}<span style='font-size:0.78em; font-weight:700; margin-left:0.45rem; "
            "padding:0.1rem 0.35rem; border-radius:999px; background:rgba(17,24,39,0.16); "
            "color:inherit; -webkit-text-fill-color:inherit;'>{badge}</span></mark>".format(
                tooltip=html_lib.escape(tooltip),
                surface=surface,
                text_color=text_color,
                mention=mention_text,
                badge=html_lib.escape(badge),
            )
        )
        cursor = end

    if cursor < len(cleaned):
        parts.append(html_lib.escape(cleaned[cursor:]))

    legend = (
        "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:0.5rem;'>"
        f"<span style='background:{CLASSICAL_ONLY_COLOR};padding:0.25rem 0.55rem;border-radius:999px;"
        "font-weight:700;border:1px solid rgba(17,24,39,.12);'>C = Classical only</span>"
        f"<span style='background:{LLM_ONLY_COLOR};padding:0.25rem 0.55rem;border-radius:999px;"
        "font-weight:700;border:1px solid rgba(17,24,39,.12);'>L = LLM only</span>"
        f"<span style='background:{overlap_match_color};padding:0.25rem 0.55rem;border-radius:999px;"
        "font-weight:700;border:1px solid rgba(17,24,39,.12);'>C+L = overlap (labels agree)</span>"
        f"<span style='background:{OVERLAP_MISMATCH_COLOR};padding:0.25rem 0.55rem;border-radius:999px;"
        "font-weight:700;border:1px solid rgba(17,24,39,.12);'>C+L! = overlap (labels differ)</span>"
        "</div>"
        "<div style='font-size:0.85rem;color:#4b5563;margin-bottom:0.55rem;'>"
        "Overlap uses soft matching (ignores articles like 'the/der/die', punctuation, and minor suffix terms). "
        "Hover highlighted spans to inspect classical vs LLM labels."
        "</div>"
    )

    content = "".join(parts)
    return (
        "<div style='padding:0.5rem;background:#ffffff;'>"
        "<div style='margin-bottom:0.4rem;'><strong>Visual Entity Overlap</strong></div>"
        f"{legend}<div style='line-height:2.5;direction:ltr;'>{content}</div></div>"
    )


def compare_extractions(
    classical_entities: list[dict[str, Any]],
    llm_entities: list[dict[str, Any]],
    llm_relations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute core overlap and disagreement metrics between pipelines."""
    classical_text_map, classical_label_map = _collect_mention_maps(classical_entities)
    llm_text_map, llm_label_map = _collect_mention_maps(llm_entities)

    classical_keys = set(classical_text_map.keys())
    llm_keys = set(llm_text_map.keys())

    overlap_pairs = _build_overlap_pairs(
        classical_keys=classical_keys,
        llm_keys=llm_keys,
        classical_spans=None,
        llm_spans=None,
    )

    overlap_classical_keys = {classical_key for classical_key, _ in overlap_pairs}
    overlap_llm_keys = {llm_key for _, llm_key in overlap_pairs}
    overlap_union_keys = overlap_classical_keys | overlap_llm_keys

    (
        _,
        _,
        classical_to_llm,
        _,
        overlap_label_match_count,
        overlap_label_mismatch_count,
    ) = _build_overlap_status_maps(
        classical_keys=classical_keys,
        llm_keys=llm_keys,
        overlap_pairs=overlap_pairs,
        classical_label_map=classical_label_map,
        llm_label_map=llm_label_map,
    )

    only_classical_keys = classical_keys - overlap_classical_keys
    only_llm_keys = llm_keys - overlap_llm_keys

    mismatch_examples: list[str] = []
    for classical_key in sorted(overlap_classical_keys):
        linked_llm = classical_to_llm.get(classical_key, set())
        if not linked_llm:
            continue
        classical_labels = classical_label_map.get(classical_key, set())
        for llm_key in sorted(linked_llm):
            llm_labels = llm_label_map.get(llm_key, set())
            if classical_labels & llm_labels:
                continue
            mention_text = classical_text_map.get(classical_key) or llm_text_map.get(llm_key) or classical_key
            mismatch_examples.append(
                f"{mention_text} (Classical: {', '.join(sorted(classical_labels)) or 'none'} | "
                f"LLM: {', '.join(sorted(llm_labels)) or 'none'})"
            )

    return {
        "classical_entity_count": len(classical_keys),
        "llm_entity_count": len(llm_keys),
        "overlap_count": len(overlap_union_keys),
        "overlap_label_match_count": overlap_label_match_count,
        "overlap_label_mismatch_count": overlap_label_mismatch_count,
        "only_classical_count": len(only_classical_keys),
        "only_llm_count": len(only_llm_keys),
        "llm_relation_count": len(llm_relations),
        "overlap_examples": _unique_sorted(
            {
                classical_text_map.get(key, key) for key in overlap_classical_keys
            }
            | {
                llm_text_map.get(key, key) for key in (overlap_llm_keys - overlap_classical_keys)
            }
        )[:8],
        "overlap_label_mismatch_examples": _unique_sorted(set(mismatch_examples))[:8],
        "only_classical_examples": _unique_sorted({classical_text_map[key] for key in only_classical_keys})[:8],
        "only_llm_examples": _unique_sorted({llm_text_map[key] for key in only_llm_keys})[:8],
    }


def build_comparison_markdown(
    model_name: str,
    classical_entities: list[dict[str, Any]],
    llm_entities: list[dict[str, Any]],
    llm_relations: list[dict[str, Any]],
    llm_status: str,
    llm_error: str | None,
) -> str:
    """Render a concise comparison section for the dashboard."""
    summary = compare_extractions(
        classical_entities=classical_entities,
        llm_entities=llm_entities,
        llm_relations=llm_relations,
    )

    if llm_status != "ok":
        return (
            "### Comparison\n"
            f"- Classical model: {model_name}\n"
            f"- Classical entities detected: {summary['classical_entity_count']}\n"
            f"- LLM status: {llm_status}\n"
            f"- LLM entities detected: {summary['llm_entity_count']}\n"
            f"- LLM relations detected: {summary['llm_relation_count']}\n\n"
            "Interpretation: classical extraction ran, but LLM extraction did not complete "
            "successfully yet.\n"
            + (f"Error detail: {llm_error}" if llm_error else "")
        )

    return (
        "### Comparison\n"
        f"- Classical model: {model_name}\n"
        f"- Classical entities detected: {summary['classical_entity_count']}\n"
        f"- LLM entities detected: {summary['llm_entity_count']}\n"
        f"- Overlap entities: {summary['overlap_count']}\n"
        f"- Overlap with matching labels: {summary['overlap_label_match_count']}\n"
        f"- Overlap with different labels: {summary['overlap_label_mismatch_count']}\n"
        f"- Classical-only entities: {summary['only_classical_count']}\n"
        f"- LLM-only entities: {summary['only_llm_count']}\n"
        f"- LLM relations detected: {summary['llm_relation_count']}\n\n"
        "Tip: use the visual overlap panel below for mention-level details and label mismatches."
    )


def _normalize_relation_name(relation: Any) -> str:
    """Normalize relation text for robust triple matching."""
    text = str(relation or "").strip().lower()
    if not text:
        return "related_to"
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    normalized = normalized or "related_to"
    return RELATION_CANONICAL_MAP.get(normalized, normalized)


def _normalize_relation_triple(triple: dict[str, Any]) -> tuple[str, str, str] | None:
    """Normalize triple into comparable canonical key."""
    subject = str(triple.get("subject", "")).strip()
    relation = str(triple.get("relation", "")).strip()
    obj = str(triple.get("object", "")).strip()
    if not (subject and relation and obj):
        return None
    return (
        _canonical_entity_key(subject),
        _normalize_relation_name(relation),
        _canonical_entity_key(obj),
    )


def compare_relations(
    classical_relations: list[dict[str, Any]],
    llm_relations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare classical and LLM typed relations using normalized triples."""
    classical_map: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    llm_map: dict[tuple[str, str, str], tuple[str, str, str]] = {}

    for triple in classical_relations:
        normalized = _normalize_relation_triple(triple)
        if normalized is None:
            continue
        classical_map.setdefault(
            normalized,
            (
                str(triple.get("subject", "")).strip(),
                str(triple.get("relation", "")).strip(),
                str(triple.get("object", "")).strip(),
            ),
        )

    for triple in llm_relations:
        normalized = _normalize_relation_triple(triple)
        if normalized is None:
            continue
        llm_map.setdefault(
            normalized,
            (
                str(triple.get("subject", "")).strip(),
                str(triple.get("relation", "")).strip(),
                str(triple.get("object", "")).strip(),
            ),
        )

    classical_keys = set(classical_map.keys())
    llm_keys = set(llm_map.keys())
    overlap_keys = classical_keys & llm_keys
    classical_only_keys = classical_keys - llm_keys
    llm_only_keys = llm_keys - classical_keys

    rows: list[list[str]] = []
    for key in sorted(overlap_keys):
        subject, relation, obj = llm_map.get(key) or classical_map[key]
        rows.append(["overlap", subject, relation, obj, "Both pipelines"])
    for key in sorted(classical_only_keys):
        subject, relation, obj = classical_map[key]
        rows.append(["classical_only", subject, relation, obj, "Classical heuristic"])
    for key in sorted(llm_only_keys):
        subject, relation, obj = llm_map[key]
        rows.append(["llm_only", subject, relation, obj, "LLM extraction"])

    return {
        "classical_relation_count": len(classical_keys),
        "llm_relation_count": len(llm_keys),
        "overlap_relation_count": len(overlap_keys),
        "classical_only_relation_count": len(classical_only_keys),
        "llm_only_relation_count": len(llm_only_keys),
        "rows": rows,
    }


def build_relation_comparison_markdown(
    classical_relations: list[dict[str, Any]],
    llm_relations: list[dict[str, Any]],
    llm_status: str,
    llm_error: str | None,
) -> str:
    """Render concise relation-comparison summary for an easy-to-read tab."""
    summary = compare_relations(
        classical_relations=classical_relations,
        llm_relations=llm_relations,
    )

    if llm_status != "ok":
        return (
            "### Relation Comparison (Typed Edges)\n"
            f"- Classical relations detected: {summary['classical_relation_count']}\n"
            f"- LLM status: {llm_status}\n"
            f"- LLM relations detected: {summary['llm_relation_count']}\n\n"
            "Interpretation: LLM relation extraction did not complete successfully, "
            "so relation overlap cannot be fully assessed yet.\n"
            + (f"Error detail: {llm_error}" if llm_error else "")
        )

    return (
        "### Relation Comparison (Typed Edges)\n"
        f"- Classical relations detected: {summary['classical_relation_count']}\n"
        f"- LLM relations detected: {summary['llm_relation_count']}\n"
        f"- Overlap relations: {summary['overlap_relation_count']}\n"
        f"- Classical-only relations: {summary['classical_only_relation_count']}\n"
        f"- LLM-only relations: {summary['llm_only_relation_count']}\n\n"
        "Interpretation: use the table below to inspect which typed edges agree or differ."
    )


def build_relation_comparison_rows(
    classical_relations: list[dict[str, Any]],
    llm_relations: list[dict[str, Any]],
) -> list[list[str]]:
    """Return table rows for relation comparison."""
    return compare_relations(
        classical_relations=classical_relations,
        llm_relations=llm_relations,
    )["rows"]

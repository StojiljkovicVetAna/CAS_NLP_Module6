"""Graph utilities for entity interaction visualization."""

from __future__ import annotations

import html as html_lib
import re
from collections import Counter
from itertools import combinations
from typing import Any

import networkx as nx
from pyvis.network import Network

from src.ner import DEFAULT_ENTITY_COLORS, load_ner_pipeline


def build_entity_interaction_graph(text: str, model_name: str) -> nx.Graph:
    """Build a sentence-level entity co-occurrence graph."""
    cleaned = text.strip()
    graph = nx.Graph()
    if not cleaned:
        return graph

    nlp = load_ner_pipeline(model_name)
    if not (
        nlp.has_pipe("parser")
        or nlp.has_pipe("senter")
        or nlp.has_pipe("sentencizer")
    ):
        nlp.add_pipe("sentencizer")

    doc = nlp(cleaned)
    if not doc.ents:
        return graph

    entity_label_counts: dict[str, Counter[str]] = {}
    node_frequency: Counter[str] = Counter()

    for ent in doc.ents:
        entity_text = ent.text.strip()
        if not entity_text:
            continue
        node_frequency[entity_text] += 1
        entity_label_counts.setdefault(entity_text, Counter())[ent.label_] += 1

    for entity_text, frequency in node_frequency.items():
        top_label = entity_label_counts[entity_text].most_common(1)[0][0]
        color = DEFAULT_ENTITY_COLORS.get(top_label, "#d3d3d3")
        graph.add_node(
            entity_text,
            label=entity_text,
            ner_label=top_label,
            frequency=frequency,
            color=color,
            size=18 + (frequency * 2),
            title=f"{entity_text} ({top_label}) - frequency: {frequency}",
        )

    try:
        sentence_spans = list(doc.sents)
    except ValueError:
        sentence_spans = [doc[:]]

    for sent in sentence_spans:
        sentence_entities = [ent.text.strip() for ent in sent.ents if ent.text.strip()]
        unique_entities = sorted(set(sentence_entities))
        for source, target in combinations(unique_entities, 2):
            if graph.has_edge(source, target):
                graph[source][target]["weight"] += 1
            else:
                graph.add_edge(source, target, weight=1)

    return graph


def _graph_to_iframe_html(graph: nx.Graph, heading: str) -> str:
    """Convert a NetworkX graph into embeddable iframe HTML via pyvis."""
    if graph.number_of_nodes() == 0:
        return (
            "<div style='padding:0.75rem;border:1px solid #e5e7eb;border-radius:8px;'>"
            f"<strong>{heading}</strong><br>No entities detected, so no interaction graph was built."
            "</div>"
        )

    network = Network(
        height="460px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#111827",
        directed=False,
        cdn_resources="in_line",
    )
    network.from_nx(graph)
    network.toggle_physics(True)

    for edge in network.edges:
        weight = edge.get("weight", 1)
        edge["value"] = weight
        edge["title"] = f"Sentence-level co-occurrence: {weight}"

    html_page = network.generate_html(notebook=False)
    escaped_html = html_lib.escape(html_page, quote=True)
    iframe = (
        "<iframe "
        "style='width:100%;height:470px;border:1px solid #e5e7eb;border-radius:8px;' "
        f"srcdoc=\"{escaped_html}\"></iframe>"
    )
    return (
        "<div style='display:flex;flex-direction:column;gap:0.5rem;'>"
        f"<div><strong>{heading}</strong></div>{iframe}</div>"
    )


def render_entity_interaction_graph_html(text: str, model_name: str) -> str:
    """Build and render the classical entity interaction graph as HTML."""
    graph = build_entity_interaction_graph(text=text, model_name=model_name)
    return _graph_to_iframe_html(
        graph=graph,
        heading="Classical Entity Interaction Graph (sentence-level co-occurrence)",
    )


def _sentence_char_spans(text: str) -> list[tuple[int, int]]:
    """Return approximate sentence spans by punctuation boundaries."""
    cleaned = text.strip()
    if not cleaned:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?]+\s+", cleaned):
        end = match.end()
        if end > start:
            spans.append((start, end))
        start = end
    if start < len(cleaned):
        spans.append((start, len(cleaned)))
    return spans


def _extract_entity_mentions(
    text: str,
    entities: list[dict[str, Any]],
) -> list[tuple[int, int, str, str]]:
    """Extract mention spans from string entities and keep non-overlapping best spans."""
    spans: list[tuple[int, int, str, str]] = []
    for entity in entities:
        entity_text = str(entity.get("text", "")).strip()
        label = str(entity.get("label", "MISC")).strip().upper()
        if not entity_text:
            continue
        for match in re.finditer(re.escape(entity_text), text, flags=re.IGNORECASE):
            spans.append((match.start(), match.end(), text[match.start() : match.end()], label))

    if not spans:
        return []

    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[tuple[int, int, str, str]] = []
    current_end = -1
    for start, end, value, label in spans:
        if start >= current_end:
            selected.append((start, end, value, label))
            current_end = end
    return selected


def build_llm_entity_interaction_graph(
    text: str,
    entities: list[dict[str, Any]],
) -> nx.Graph:
    """Build sentence-level co-occurrence graph from LLM entities."""
    cleaned = text.strip()
    graph = nx.Graph()
    if not cleaned:
        return graph

    mentions = _extract_entity_mentions(cleaned, entities)
    if not mentions:
        return graph

    node_frequency: Counter[str] = Counter()
    node_labels: dict[str, Counter[str]] = {}
    for _, _, mention_text, mention_label in mentions:
        normalized = mention_text.strip()
        if not normalized:
            continue
        node_frequency[normalized] += 1
        node_labels.setdefault(normalized, Counter())[mention_label] += 1

    for entity_text, frequency in node_frequency.items():
        top_label = node_labels[entity_text].most_common(1)[0][0]
        color = DEFAULT_ENTITY_COLORS.get(top_label, "#d3d3d3")
        graph.add_node(
            entity_text,
            label=entity_text,
            ner_label=top_label,
            frequency=frequency,
            color=color,
            size=18 + (frequency * 2),
            title=f"{entity_text} ({top_label}) - frequency: {frequency}",
        )

    sentence_spans = _sentence_char_spans(cleaned)
    for sent_start, sent_end in sentence_spans:
        sentence_mentions = [
            mention_text.strip()
            for start, end, mention_text, _ in mentions
            if start >= sent_start and end <= sent_end and mention_text.strip()
        ]
        unique_entities = sorted(set(sentence_mentions))
        for source, target in combinations(unique_entities, 2):
            if graph.has_edge(source, target):
                graph[source][target]["weight"] += 1
            else:
                graph.add_edge(source, target, weight=1)

    return graph


def render_llm_entity_interaction_graph_html(
    text: str,
    entities: list[dict[str, Any]],
) -> str:
    """Render LLM co-occurrence graph (same logic family as classical graph)."""
    graph = build_llm_entity_interaction_graph(text=text, entities=entities)
    return _graph_to_iframe_html(
        graph=graph,
        heading="LLM Entity Interaction Graph (sentence-level co-occurrence)",
    )


def build_relation_graph(relations: list[dict[str, Any]]) -> nx.DiGraph:
    """Build a directed graph from LLM relation triples."""
    graph = nx.DiGraph()
    if not relations:
        return graph

    for triple in relations:
        subject = str(triple.get("subject", "")).strip()
        relation = str(triple.get("relation", "")).strip()
        obj = str(triple.get("object", "")).strip()
        if not (subject and relation and obj):
            continue

        if not graph.has_node(subject):
            graph.add_node(subject, label=subject, color="#93c5fd", size=20, title=subject)
        if not graph.has_node(obj):
            graph.add_node(obj, label=obj, color="#86efac", size=20, title=obj)

        if graph.has_edge(subject, obj):
            graph[subject][obj]["weight"] += 1
            existing_labels = graph[subject][obj].get("relation_labels", [])
            if relation not in existing_labels:
                existing_labels.append(relation)
            graph[subject][obj]["relation_labels"] = existing_labels
        else:
            graph.add_edge(subject, obj, weight=1, relation_labels=[relation])

    for source, target, attrs in graph.edges(data=True):
        relation_labels = sorted(attrs.get("relation_labels", []))
        relation_text = ", ".join(relation_labels)
        attrs["title"] = f"{source} -> {target}<br>relation: {relation_text}"
        attrs["label"] = relation_text[:30]

    return graph


def render_relation_graph_html(
    relations: list[dict[str, Any]],
    heading: str = "LLM Relation Graph (typed edges)",
) -> str:
    """Render typed relation graph as embeddable HTML."""
    graph = build_relation_graph(relations)
    if graph.number_of_nodes() == 0:
        return (
            "<div style='padding:0.75rem;border:1px solid #e5e7eb;border-radius:8px;'>"
            f"<strong>{heading}</strong><br>No relations to visualize yet."
            "</div>"
        )

    network = Network(
        height="460px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#111827",
        directed=True,
        cdn_resources="in_line",
    )
    network.from_nx(graph)
    network.toggle_physics(True)

    for edge in network.edges:
        edge["arrows"] = "to"
        edge["value"] = edge.get("weight", 1)
        edge["width"] = 1.2
        edge["font"] = {
            "size": 10,
            "color": "#4b5563",
            "strokeWidth": 0,
            "align": "top",
        }
        edge["length"] = 230

    # Tune defaults for readability: larger node labels, smaller edge labels,
    # and longer spring length to reduce overlap.
    network.set_options(
        """
        {
          "nodes": {
            "shape": "dot",
            "size": 24,
            "font": {
              "size": 20,
              "face": "Arial",
              "color": "#111827"
            },
            "borderWidth": 1,
            "shadow": false
          },
          "edges": {
            "smooth": {
              "type": "dynamic",
              "roundness": 0.25
            },
            "color": {
              "inherit": false,
              "color": "#94a3b8",
              "highlight": "#475569"
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -45,
              "springLength": 235,
              "springConstant": 0.04,
              "damping": 0.45,
              "avoidOverlap": 0.85
            },
            "minVelocity": 0.75,
            "stabilization": {
              "enabled": true,
              "iterations": 500
            }
          }
        }
        """
    )

    html_page = network.generate_html(notebook=False)
    escaped_html = html_lib.escape(html_page, quote=True)
    iframe = (
        "<iframe "
        "style='width:100%;height:470px;border:1px solid #e5e7eb;border-radius:8px;' "
        f"srcdoc=\"{escaped_html}\"></iframe>"
    )
    return (
        "<div style='display:flex;flex-direction:column;gap:0.5rem;'>"
        f"<div><strong>{heading}</strong></div>"
        f"{iframe}</div>"
    )


def extract_classical_relations(text: str, model_name: str) -> list[dict[str, str]]:
    """Extract lightweight typed relations from classical pipeline using simple verb-between-entities rules."""
    cleaned = text.strip()
    if not cleaned:
        return []

    nlp = load_ner_pipeline(model_name)
    if not (
        nlp.has_pipe("parser")
        or nlp.has_pipe("senter")
        or nlp.has_pipe("sentencizer")
    ):
        nlp.add_pipe("sentencizer")

    doc = nlp(cleaned)
    triples: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    try:
        sentences = list(doc.sents)
    except ValueError:
        sentences = [doc[:]]

    for sent in sentences:
        sent_ents = [ent for ent in sent.ents if ent.text.strip()]
        if len(sent_ents) < 2:
            continue

        for left, right in zip(sent_ents, sent_ents[1:]):
            between = doc[left.end : right.start]
            verbs = [
                token
                for token in between
                if token.pos_ in {"VERB", "AUX"} and not token.is_space and not token.is_punct
            ]
            if verbs:
                relation = verbs[0].lemma_.strip().lower().replace(" ", "_")
            else:
                relation = "related_to"
            relation = relation or "related_to"
            subject = left.text.strip()
            obj = right.text.strip()
            if not (subject and obj):
                continue
            triple_key = (subject, relation, obj)
            if triple_key in seen:
                continue
            seen.add(triple_key)
            triples.append({"subject": subject, "relation": relation, "object": obj})

    return triples


def render_classical_relation_graph_html(text: str, model_name: str) -> str:
    """Render lightweight classical knowledge graph from heuristic typed relations."""
    relations = extract_classical_relations(text=text, model_name=model_name)
    return render_relation_graph_html(
        relations=relations,
        heading="Classical Knowledge Graph (heuristic typed relations)",
    )

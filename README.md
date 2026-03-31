---
title: NLP M6 NER Project
emoji: "🧠"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.9.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# NLP Extraction Comparison Dashboard

This dashboard compares two information extraction approaches on the same input text:

- Classical NLP extraction (NER + lightweight logic)
- LLM-based extraction (structured entities and relations)

The goal is to make methodological differences visible and easy to inspect.

## What the app shows

The app is organized into tabs:

- `Input`: paste text, choose a classical spaCy model, run analysis.
- `Classical pipeline`: highlighted entities, entity table, and label counts from spaCy.
- `LLM pipeline`: highlighted entities, entity table, label counts, and raw structured output.
- `Comparison`: overlap-focused comparison of extracted entities with concise interpretation.
- `Relation comparison`: normalized triple table with `overlap`, `classical_only`, `llm_only`.
- `Graph comparison`: entity interaction graphs (sentence-level co-occurrence) for both methods.
- `Knowledge graph`: typed relation graphs for both methods.

## Classical pipeline

- Classical spaCy NER pipeline is active (highlighting, tables, counts, interaction graph).
- Available classical NER models in the UI: English (default), German, French, Italian, Multilingual.
- Entity interaction graph is built from sentence-level co-occurrence.
- Typed relations use a lightweight heuristic (verb-between-entities rule).

## LLM pipeline

- LLM extraction uses an OpenAI-compatible external endpoint.
- The prompt requests machine-readable JSON with:
  - entities (`text`, labels, mapping confidence, optional new topic)
  - relations (`subject`, `relation`, `object`)
- Response parsing is defensive (JSON extraction, validation, graceful error states).

## Comparison principles

- Co-occurrence graphs are used for apples-to-apples interaction comparison.
- Knowledge graphs are used only for typed semantic relations.
- Relation overlap is computed on normalized triples to make disagreements explicit.

## Minimal runtime requirement

- LLM extraction requires `GPUSTACK_API_KEY` in environment variables.

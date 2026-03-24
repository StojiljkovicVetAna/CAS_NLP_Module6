"""Minimal placeholder Gradio app for initial Hugging Face Space deployment."""

from __future__ import annotations

import re
from collections import Counter

import gradio as gr


def _extract_capitalized_tokens(text: str) -> list[str]:
    """Return unique capitalized tokens as lightweight entity candidates."""
    tokens = re.findall(r"\b[A-Z][a-zA-Z]+\b", text)
    seen: set[str] = set()
    entities: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            entities.append(token)
    return entities


def run_placeholder_analysis(text: str) -> tuple[str, dict, dict, str]:
    """Generate placeholder outputs for Classical, LLM, and Comparison sections."""
    cleaned = (text or "").strip()
    if not cleaned:
        empty_msg = (
            "### Waiting for input\n"
            "Please provide a short paragraph to run the placeholder dashboard."
        )
        return empty_msg, {"entities": [], "counts_by_label": {}}, {"relations": []}, empty_msg

    entities = _extract_capitalized_tokens(cleaned)
    entity_rows = [{"entity": item, "label": "PROPN_CANDIDATE"} for item in entities]
    label_counts = Counter(row["label"] for row in entity_rows)

    if len(entities) >= 2:
        llm_relations = [
            {"subject": entities[0], "relation": "related_to", "object": entities[1]}
        ]
    else:
        llm_relations = []

    classical_md = (
        "### Classical Pipeline (Placeholder)\n"
        f"- Characters: {len(cleaned)}\n"
        f"- Words: {len(cleaned.split())}\n"
        f"- Candidate entities: {len(entity_rows)}\n\n"
        "Current logic uses a simple capitalized-token heuristic for demonstration."
    )

    llm_json = {
        "relations": llm_relations,
        "status": "placeholder_output",
        "note": "Later we will replace this with external API extraction.",
    }

    comparison_md = (
        "### Comparison (Placeholder)\n"
        f"- Classical entities found: {len(entity_rows)}\n"
        f"- LLM-style relations found: {len(llm_relations)}\n\n"
        "Interpretation: this page is wired for side-by-side comparison, but extraction "
        "methods are not final yet."
    )

    classical_json = {
        "entities": entity_rows,
        "counts_by_label": dict(label_counts),
    }
    return classical_md, classical_json, llm_json, comparison_md


with gr.Blocks(title="NLP Extraction Comparison Dashboard") as demo:
    gr.Markdown("# NLP Extraction Comparison Dashboard")
    gr.Markdown(
        "This is a placeholder deployment used to validate Hugging Face Space sync "
        "before implementing full NLP pipelines."
    )

    input_text = gr.Textbox(
        label="Input text",
        lines=8,
        placeholder="Paste a short text here...",
    )
    run_button = gr.Button("Run Placeholder Analysis", variant="primary")

    with gr.Tabs():
        with gr.TabItem("Classical pipeline"):
            classical_output = gr.Markdown()
            classical_json_output = gr.JSON(label="Classical structured output")
        with gr.TabItem("LLM pipeline"):
            llm_json_output = gr.JSON(label="LLM structured output")
        with gr.TabItem("Comparison"):
            comparison_output = gr.Markdown()

    run_button.click(
        fn=run_placeholder_analysis,
        inputs=input_text,
        outputs=[
            classical_output,
            classical_json_output,
            llm_json_output,
            comparison_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()

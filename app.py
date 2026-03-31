"""Gradio dashboard for comparing classical NLP and LLM extraction."""

from __future__ import annotations

import html as html_lib
from time import perf_counter
from typing import Any

import gradio as gr
import spacy

from src.compare import (
    build_comparison_markdown,
    build_relation_comparison_markdown,
    build_relation_comparison_rows,
    render_comparison_overlap_html,
)
from src.graphs import (
    extract_classical_relations,
    render_entity_interaction_graph_html,
    render_llm_entity_interaction_graph_html,
    render_relation_graph_html,
)
from src.llm_extraction import run_llm_extraction
from src.ner import (
    DEFAULT_ENTITY_COLORS,
    DEFAULT_MODEL_NAME,
    count_entities_by_label,
    extract_entities,
    get_model_ner_labels,
    render_entities_html,
)
from src.utils import (
    build_llm_entity_rows,
    normalize_entities_for_display,
    render_llm_entities_html,
)

NER_MODEL_OPTIONS = {
    "English (en_core_web_sm)": "en_core_web_sm",
    "German (de_core_news_sm)": "de_core_news_sm",
    "French (fr_core_news_sm)": "fr_core_news_sm",
    "Italian (it_core_news_sm)": "it_core_news_sm",
    "Multilingual (xx_ent_wiki_sm)": "xx_ent_wiki_sm",
}
DEFAULT_MODEL_LABEL = "English (en_core_web_sm)"
DEFAULT_SAMPLE_TEXT = (
    "The CAS Natural Language Processing (CAS NLP) at the University of Bern is organized "
    "by the Mathematical Institute in Bern, Switzerland. The 2026/2027 edition runs from "
    "August 10, 2026, to July 2027 and awards 16 ECTS credits. The program combines online "
    "and in-person teaching, with modules hosted in Bern, Module 6 in Muerren (Bernese "
    "Oberland), and Module 3 at Lago Maggiore, Italy. Participants work on practical NLP "
    "topics such as information extraction, transformer-based methods, and Large Language "
    "Models like ChatGPT, BERT, and Gemini."
)


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


def build_color_legend_html() -> str:
    """Create a compact color legend for entity labels."""
    chips = []
    for label, color in sorted(DEFAULT_ENTITY_COLORS.items()):
        explanation = spacy.explain(label) or "No description available."
        safe_label = html_lib.escape(label)
        safe_explanation = html_lib.escape(explanation)
        text_color = _text_color_for_background(color)
        chips.append(
            "<span class='legend-chip-wrap' tabindex='0'>"
            f"<span class='legend-chip' style='background:{color};color:{text_color} !important;"
            f"-webkit-text-fill-color:{text_color} !important;'>{safe_label}</span>"
            f"<span class='legend-tip'><strong>{safe_label}</strong>: {safe_explanation}</span>"
            "</span>"
        )
    return (
        "<style>"
        ".legend-wrap{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;}"
        ".legend-chip-wrap{position:relative;display:inline-flex;outline:none;}"
        ".legend-chip{display:inline-block;padding:4px 8px;border-radius:999px;font-size:12px;"
        "font-weight:700;}"
        ".legend-tip{position:absolute;left:50%;transform:translateX(-50%);bottom:calc(100% + 8px);"
        "background:#111827;color:#fff;font-size:12px;line-height:1.3;padding:6px 8px;border-radius:6px;"
        "width:max-content;max-width:260px;white-space:normal;z-index:20;opacity:0;visibility:hidden;"
        "transition:opacity .12s ease;box-shadow:0 6px 18px rgba(0,0,0,.2);}"
        ".legend-tip,.legend-tip *{color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;}"
        ".legend-tip::after{content:'';position:absolute;left:50%;top:100%;transform:translateX(-50%);"
        "border-width:6px;border-style:solid;border-color:#111827 transparent transparent transparent;}"
        ".legend-chip-wrap:hover .legend-tip,.legend-chip-wrap:focus .legend-tip{opacity:1;visibility:visible;}"
        "</style>"
        "<div><strong>Entity Color Legend</strong></div>"
        "<div class='legend-wrap'>"
        + "".join(chips)
        + "</div>"
    )


def _classical_summary(
    text: str,
    entity_count: int,
    unique_labels: int,
    model_name: str,
) -> str:
    """Format a concise summary for the classical pipeline tab."""
    return (
        "### Classical Pipeline (spaCy NER)\n"
        f"- Model: {model_name}\n"
        f"- Characters: {len(text)}\n"
        f"- Words: {len(text.split())}\n"
        f"- Entities detected: {entity_count}\n"
        f"- Distinct labels: {unique_labels}"
    )


def _llm_summary(
    llm_status: str,
    llm_model: str | None,
    entity_count: int,
    relation_count: int,
    allowed_labels: list[str],
    new_topics: list[str],
    llm_error: str | None,
) -> str:
    """Format a concise summary for the LLM pipeline tab."""
    allowed_str = ", ".join(allowed_labels) if allowed_labels else "none"
    new_topics_str = ", ".join(new_topics) if new_topics else "none"
    base = (
        "### LLM Pipeline (External API)\n"
        f"- Status: {llm_status}\n"
        f"- Model: {llm_model or 'unknown'}\n"
        f"- Allowed labels (from selected spaCy model): {allowed_str}\n"
        f"- Entities detected: {entity_count}\n"
        f"- Relations detected: {relation_count}\n"
        f"- NEW categories suggested by LLM: {new_topics_str}"
    )
    if llm_error:
        return f"{base}\n- Error: {llm_error}"
    return base


def _switch_to_results_tab_updates() -> tuple[gr.Tabs, gr.Tabs]:
    """Switch UI focus to results tab and classical subtab."""
    return gr.Tabs(selected="results_tab"), gr.Tabs(selected="classical_tab")


def _input_status_message(status: str, elapsed_seconds: float | None = None) -> str:
    """Render a compact input-tab status message with optional runtime."""
    if elapsed_seconds is None:
        return f"### Input Status\n- {status}"
    return f"### Input Status\n- {status}\n- Processing time: {elapsed_seconds:.2f} seconds"


def run_dashboard(
    text: str,
    model_label: str,
) -> tuple[
    str,
    str,
    list[list[Any]],
    dict[str, int],
    str,
    str,
    list[list[Any]],
    dict[str, int],
    dict[str, Any],
    str,
    str,
    str,
    list[list[str]],
    str,
    str,
    str,
    str,
    str,
    Any,
]:
    """Run the current dashboard flow with full classical NER and placeholders for the rest."""
    cleaned = (text or "").strip()
    selected_model = NER_MODEL_OPTIONS.get(model_label, DEFAULT_MODEL_NAME)
    started_at = perf_counter()

    llm_placeholder = {
        "status": "skipped_due_to_classical_error",
        "entities": [],
        "relations": [],
        "error": "LLM extraction was skipped because classical NER failed first.",
    }

    if not cleaned:
        empty_summary = "### Classical Pipeline (spaCy NER)\nPlease provide input text."
        comparison = (
            "### Comparison\n"
            "No comparison available yet because there is no input text."
        )
        comparison_visual = "<p>No visual comparison available yet because there is no input text.</p>"
        llm_not_started = {
            "status": "not_started",
            "entities": [],
            "relations": [],
            "model": None,
            "base_url": None,
            "error": None,
        }
        return (
            empty_summary,
            "<p>No text provided.</p>",
            [],
            {},
            "### LLM Pipeline (External API)\nPress **Run Analysis** to start.",
            "<p>No analysis yet.</p>",
            [],
            {},
            llm_not_started,
            comparison,
            comparison_visual,
            "### Relation Comparison (Typed Edges)\nNo input text yet.",
            [],
            "<p>No graph yet.</p>",
            "<p>No graph yet.</p>",
            "<p>No knowledge graph yet.</p>",
            render_relation_graph_html([], heading="LLM Knowledge Graph (typed relations)"),
            _input_status_message("Waiting for text input."),
            gr.Button(value="Go to Results", interactive=False),
        )

    try:
        entities = extract_entities(cleaned, model_name=selected_model)
        label_counts = count_entities_by_label(entities)
        entity_rows = [
            [item["text"], item["label"], item["start_char"], item["end_char"]]
            for item in entities
        ]
        entity_html = render_entities_html(
            cleaned,
            model_name=selected_model,
            entity_colors=DEFAULT_ENTITY_COLORS,
        )
        classical_graph_html = render_entity_interaction_graph_html(
            text=cleaned,
            model_name=selected_model,
        )
        classical_relations = extract_classical_relations(
            text=cleaned,
            model_name=selected_model,
        )
        classical_kg_graph_html = render_relation_graph_html(
            classical_relations,
            heading="Classical Knowledge Graph (heuristic typed relations)",
        )
    except RuntimeError as exc:
        error_summary = (
            "### Classical Pipeline (spaCy NER)\n"
            f"- Model: {selected_model}\n"
            f"Could not run NER: {exc}"
        )
        comparison = (
            "### Comparison\n"
            "Classical extraction did not run, so comparison is skipped."
        )
        comparison_visual = "<p>Visual comparison unavailable because classical extraction did not run.</p>"
        elapsed = perf_counter() - started_at
        return (
            error_summary,
            "<p>NER model unavailable.</p>",
            [],
            {},
            "### LLM Pipeline (External API)\nSkipped because classical pipeline failed.",
            "<p>LLM highlighting skipped.</p>",
            [],
            {},
            llm_placeholder,
            comparison,
            comparison_visual,
            "### Relation Comparison (Typed Edges)\nUnavailable because classical NER failed.",
            [],
            "<p>Graph unavailable because NER model is unavailable.</p>",
            "<p>Graph unavailable because NER model is unavailable.</p>",
            "<p>Knowledge graph unavailable because NER model is unavailable.</p>",
            render_relation_graph_html([], heading="LLM Knowledge Graph (typed relations)"),
            _input_status_message("Analysis finished with classical NER error. Open Results for details.", elapsed),
            gr.Button(value="Go to Results", interactive=True),
        )

    try:
        allowed_labels = get_model_ner_labels(selected_model)
        allowed_label_descriptions = {
            label: (spacy.explain(label) or "")
            for label in allowed_labels
        }
        llm_result = run_llm_extraction(
            cleaned,
            allowed_labels=allowed_labels,
            allowed_label_descriptions=allowed_label_descriptions,
        )
        llm_entities = normalize_entities_for_display(llm_result.get("entities", []))
        llm_result["entities"] = llm_entities
        llm_relations = llm_result.get("relations", [])
        if not isinstance(llm_relations, list):
            llm_relations = []
            llm_result["relations"] = []
        llm_status = str(llm_result.get("status", "api_error"))
        llm_error = llm_result.get("error")
        llm_rows = build_llm_entity_rows(cleaned, llm_entities)
        llm_counts = count_entities_by_label(llm_entities)
        new_topics = sorted(
            {
                str(entity.get("new_topic", "")).strip()
                for entity in llm_entities
                if str(entity.get("label", "")) == "NEW" and str(entity.get("new_topic", "")).strip()
            }
        )
        llm_entity_html = render_llm_entities_html(
            text=cleaned,
            entities=llm_entities,
            color_map=DEFAULT_ENTITY_COLORS,
        )
        llm_summary = _llm_summary(
            llm_status=llm_status,
            llm_model=llm_result.get("model"),
            entity_count=len(llm_entities),
            relation_count=len(llm_relations),
            allowed_labels=allowed_labels,
            new_topics=new_topics,
            llm_error=llm_error,
        )
        llm_graph_html = render_llm_entity_interaction_graph_html(
            text=cleaned,
            entities=llm_entities,
        )
        llm_kg_graph_html = render_relation_graph_html(
            llm_relations,
            heading="LLM Knowledge Graph (typed relations)",
        )
    except Exception as exc:  # noqa: BLE001
        llm_result = {
            "status": "runtime_error",
            "entities": [],
            "relations": [],
            "error": f"LLM branch failed: {exc}",
        }
        llm_entities = []
        llm_relations = []
        llm_status = "runtime_error"
        llm_error = str(exc)
        llm_rows = []
        llm_counts = {}
        new_topics = []
        llm_entity_html = "<p>LLM entity highlighting unavailable due to runtime error.</p>"
        llm_summary = _llm_summary(
            llm_status=llm_status,
            llm_model=None,
            entity_count=0,
            relation_count=0,
            allowed_labels=[],
            new_topics=new_topics,
            llm_error=llm_error,
        )
        llm_graph_html = "<p>LLM co-occurrence graph unavailable due to runtime error.</p>"
        llm_kg_graph_html = render_relation_graph_html(
            [],
            heading="LLM Knowledge Graph (typed relations)",
        )
    classical_summary = _classical_summary(
        text=cleaned,
        entity_count=len(entities),
        unique_labels=len(label_counts),
        model_name=selected_model,
    )

    comparison_md = build_comparison_markdown(
        model_name=selected_model,
        classical_entities=entities,
        llm_entities=llm_entities,
        llm_relations=llm_relations,
        llm_status=llm_status,
        llm_error=llm_error,
    )
    comparison_visual = render_comparison_overlap_html(
        text=cleaned,
        classical_entities=entities,
        llm_entities=llm_entities,
    )
    relation_comparison_md = build_relation_comparison_markdown(
        classical_relations=classical_relations,
        llm_relations=llm_relations,
        llm_status=llm_status,
        llm_error=llm_error,
    )
    relation_rows = build_relation_comparison_rows(
        classical_relations=classical_relations,
        llm_relations=llm_relations,
    )
    elapsed = perf_counter() - started_at

    return (
        classical_summary,
        entity_html,
        entity_rows,
        label_counts,
        llm_summary,
        llm_entity_html,
        llm_rows,
        llm_counts,
        llm_result,
        comparison_md,
        comparison_visual,
        relation_comparison_md,
        relation_rows,
        classical_graph_html,
        llm_graph_html,
        classical_kg_graph_html,
        llm_kg_graph_html,
        _input_status_message("Analysis complete. Results are ready.", elapsed),
        gr.Button(value="Go to Results", interactive=True),
    )


with gr.Blocks(title="NLP Extraction Comparison Dashboard") as demo:
    gr.Markdown("# NLP Extraction Comparison Dashboard")
    gr.Markdown(
        "This version implements the **Classical pipeline** with spaCy NER and colored "
        "entity highlights, plus an LLM extraction branch via external API."
    )

    with gr.Tabs(selected="input_tab") as app_tabs:
        with gr.TabItem("Input", id="input_tab"):
            input_text = gr.Textbox(
                label="Input text",
                lines=8,
                placeholder="Paste a short text here...",
            )
            gr.Examples(
                examples=[[DEFAULT_SAMPLE_TEXT]],
                inputs=[input_text],
                label="Sample text (click to load, then press Run Analysis)",
            )
            model_selector = gr.Dropdown(
                label="Classical NER model",
                choices=list(NER_MODEL_OPTIONS.keys()),
                value=DEFAULT_MODEL_LABEL,
            )
            run_button = gr.Button("Run Analysis", variant="primary")
            go_results_button = gr.Button("Go to Results", variant="secondary", interactive=False)
            input_status_output = gr.Markdown(value=_input_status_message("Ready to run analysis."))
            gr.Markdown(
                "Run analysis first. When finished, the button above becomes active and you can jump "
                "to the **Results** tab."
            )

        with gr.TabItem("Results", id="results_tab"):
            with gr.Tabs(selected="classical_tab") as result_tabs:
                with gr.TabItem("Classical pipeline", id="classical_tab"):
                    classical_summary_output = gr.Markdown(
                        value="### Classical Pipeline (spaCy NER)\nPress **Run Analysis** to start."
                    )
                    color_legend = gr.HTML(value=build_color_legend_html())
                    entity_html_output = gr.HTML(
                        value="<p>No analysis yet.</p>",
                        label="Entity highlighting",
                    )
                    entity_table_output = gr.Dataframe(
                        headers=["text", "label", "start_char", "end_char"],
                        datatype=["str", "str", "number", "number"],
                        label="Extracted entities",
                        value=[],
                        interactive=False,
                    )
                    label_counts_output = gr.JSON(value={}, label="Counts by label")

                with gr.TabItem("LLM pipeline", id="llm_tab"):
                    llm_summary_output = gr.Markdown(
                        value="### LLM Pipeline (External API)\nPress **Run Analysis** to start."
                    )
                    llm_color_legend = gr.HTML(value=build_color_legend_html())
                    llm_entity_html_output = gr.HTML(
                        value="<p>No analysis yet.</p>",
                        label="Entity highlighting",
                    )
                    llm_entity_table_output = gr.Dataframe(
                        headers=[
                            "text",
                            "label",
                            "raw_label",
                            "new_topic",
                            "mapping_confidence",
                            "start_char",
                            "end_char",
                        ],
                        datatype=["str", "str", "str", "str", "number", "number", "number"],
                        label="Extracted entities",
                        value=[],
                        interactive=False,
                    )
                    llm_counts_output = gr.JSON(value={}, label="Counts by label")
                    llm_output = gr.JSON(
                        value={"status": "not_started"},
                        label="LLM structured output",
                    )

                with gr.TabItem("Comparison", id="comparison_tab"):
                    comparison_output = gr.Markdown(
                        value="### Comparison\nRun analysis to populate this section."
                    )
                    comparison_visual_output = gr.HTML(
                        value="<p>Visual comparison appears after analysis.</p>",
                        label="Visual overlap",
                    )
                with gr.TabItem("Relation comparison", id="relation_comparison_tab"):
                    relation_comparison_output = gr.Markdown(
                        value="### Relation Comparison (Typed Edges)\nRun analysis to populate this section."
                    )
                    relation_table_output = gr.Dataframe(
                        headers=["status", "subject", "relation", "object", "notes"],
                        datatype=["str", "str", "str", "str", "str"],
                        label="Relation overlap table",
                        value=[],
                        interactive=False,
                    )
                with gr.TabItem("Graph comparison", id="graphs_tab"):
                    gr.Markdown(
                        "Both graphs below are **entity interaction graphs** built from "
                        "sentence-level co-occurrence, so this tab is apples-to-apples."
                    )
                    classical_graph_output = gr.HTML(
                        value="<p>Classical graph will appear after running analysis.</p>",
                        label="Classical entity interaction graph",
                    )
                    llm_graph_output = gr.HTML(
                        value="<p>LLM graph will appear after running analysis.</p>",
                        label="LLM entity interaction graph",
                    )
                with gr.TabItem("Knowledge graph", id="knowledge_tab"):
                    gr.Markdown(
                        "Both graphs below are **typed relation graphs**. "
                        "Classical uses lightweight heuristic relations; LLM uses extracted triples."
                    )
                    classical_kg_graph_output = gr.HTML(
                        value="<p>Classical knowledge graph will appear after running analysis.</p>",
                        label="Classical knowledge graph",
                    )
                    llm_kg_graph_output = gr.HTML(
                        value=render_relation_graph_html([], heading="LLM Knowledge Graph (typed relations)"),
                        label="LLM knowledge graph",
                    )

    run_button.click(
        fn=run_dashboard,
        inputs=[input_text, model_selector],
        outputs=[
            classical_summary_output,
            entity_html_output,
            entity_table_output,
            label_counts_output,
            llm_summary_output,
            llm_entity_html_output,
            llm_entity_table_output,
            llm_counts_output,
            llm_output,
            comparison_output,
            comparison_visual_output,
            relation_comparison_output,
            relation_table_output,
            classical_graph_output,
            llm_graph_output,
            classical_kg_graph_output,
            llm_kg_graph_output,
            input_status_output,
            go_results_button,
        ],
    )
    go_results_button.click(
        fn=_switch_to_results_tab_updates,
        inputs=None,
        outputs=[app_tabs, result_tabs],
    )

if __name__ == "__main__":
    demo.launch()

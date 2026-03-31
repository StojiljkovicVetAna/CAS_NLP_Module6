"""LLM extraction utilities using an OpenAI-compatible API (GPUStack)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

GPUSTACK_BASE_URL_DEFAULT = "https://gpustack.unibe.ch/v1"
GPUSTACK_MODEL_DEFAULT = "gpt-oss-120b"
GPUSTACK_MAX_TOKENS_DEFAULT = 4000

# Load local .env when available (useful for local development).
# In Hugging Face Spaces, secrets are injected as environment variables.
load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

def _get_env_value(*keys: str, default: str | None = None) -> str | None:
    """Read first non-empty environment value across key aliases."""
    for key in keys:
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()

        quoted_key = f'"{key}"'
        quoted_value = os.getenv(quoted_key)
        if quoted_value and quoted_value.strip():
            return quoted_value.strip()
    return default


def _build_system_prompt(
    allowed_labels: list[str] | None,
    label_descriptions: dict[str, str] | None = None,
) -> str:
    """Build system prompt with optional dynamic label constraints."""
    base = (
        "You are an information extraction assistant. "
        "Extract entities and semantic relations from the provided text. "
        "Return ONLY valid JSON with this exact schema (no markdown, no code fences): "
        "{\"entities\": [{\"text\": str, \"raw_label\": str, \"mapped_label\": str, "
        "\"mapping_confidence\": float, \"new_topic\": str|null}], "
        "\"relations\": [{\"subject\": str, \"relation\": str, \"object\": str}]}. "
        "Use concise labels and relation names. "
        "Limit output to at most 30 entities and 40 relations."
    )
    if not allowed_labels:
        return base

    allowed_sorted = sorted(allowed_labels)
    allowed = ", ".join(allowed_sorted)
    if label_descriptions:
        with_desc = [
            f"{label}: {label_descriptions.get(label, '')}".strip()
            for label in allowed_sorted
        ]
        allowed_detail = "; ".join(with_desc)
    else:
        allowed_detail = allowed

    return (
        f"{base} Allowed labels are: [{allowed}]. "
        f"Label descriptions: {allowed_detail}. "
        "For each entity, set raw_label to your best free-form category. "
        "Then set mapped_label to one of the allowed labels OR NEW. "
        "Set mapping_confidence between 0 and 1. "
        "If mapped_label is NEW, set new_topic to the proposed category string. "
        "If mapped_label is one of the allowed labels, set new_topic to null."
    )


def _clean_optional_text(value: Any) -> str | None:
    """Return trimmed text or None for null/empty-like values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "n/a"}:
        return None
    return text


def _extract_first_json_object(raw_text: str) -> str:
    """Extract the first JSON object/list from text if wrapped with extra tokens."""
    text = raw_text.strip()
    if not text:
        return ""

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
        if candidate:
            return candidate

    if text.startswith("{") or text.startswith("["):
        return text

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
    return text


def _normalize_entities(
    raw_entities: Any,
    allowed_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Normalize entity objects into a stable list schema."""
    normalized: list[dict[str, Any]] = []
    if not isinstance(raw_entities, list):
        return normalized

    for item in raw_entities:
        if isinstance(item, str):
            entity_text = item.strip()
            if entity_text:
                if allowed_labels:
                    normalized.append(
                        {
                            "text": entity_text,
                            "label": "NEW",
                            "raw_label": None,
                            "mapping_confidence": None,
                            "new_topic": "unspecified",
                        }
                    )
                else:
                    normalized.append(
                        {
                            "text": entity_text,
                            "label": "UNKNOWN",
                            "raw_label": None,
                            "mapping_confidence": None,
                            "new_topic": None,
                        }
                    )
            continue

        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()
        raw_label_value = _clean_optional_text(item.get("raw_label"))
        label_value = _clean_optional_text(item.get("mapped_label")) or _clean_optional_text(item.get("label"))
        label = (label_value or "UNKNOWN").upper().replace(" ", "_")
        raw_label = (raw_label_value or label).upper().replace(" ", "_")
        new_topic = _clean_optional_text(item.get("new_topic"))
        confidence_raw = item.get("mapping_confidence")
        try:
            mapping_confidence = float(confidence_raw) if confidence_raw is not None else None
        except (TypeError, ValueError):
            mapping_confidence = None
        if text:
            if allowed_labels:
                if label in allowed_labels:
                    normalized.append(
                        {
                            "text": text,
                            "label": label,
                            "raw_label": raw_label,
                            "mapping_confidence": mapping_confidence,
                            "new_topic": None,
                        }
                    )
                elif label == "NEW":
                    normalized.append(
                        {
                            "text": text,
                            "label": "NEW",
                            "raw_label": raw_label,
                            "mapping_confidence": mapping_confidence,
                            "new_topic": new_topic or "unspecified",
                        }
                    )
                else:
                    normalized.append(
                        {
                            "text": text,
                            "label": "NEW",
                            "raw_label": raw_label,
                            "mapping_confidence": mapping_confidence,
                            "new_topic": label,
                        }
                    )
            else:
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


def _normalize_relations(raw_relations: Any) -> list[dict[str, str]]:
    """Normalize relation objects into subject-relation-object triples."""
    normalized: list[dict[str, str]] = []
    if not isinstance(raw_relations, list):
        return normalized

    for item in raw_relations:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        relation = str(item.get("relation", "")).strip()
        obj = str(item.get("object", "")).strip()
        if subject and relation and obj:
            normalized.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                }
            )
    return normalized


def _parse_extraction_payload(
    content: str,
    allowed_labels: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Parse model content into normalized entities and relations."""
    candidate = _extract_first_json_object(content)
    payload = json.loads(candidate)

    if isinstance(payload, list):
        relations = _normalize_relations(payload)
        return [], relations

    if not isinstance(payload, dict):
        raise ValueError("Parsed payload is neither a JSON object nor a JSON list.")

    entities = _normalize_entities(payload.get("entities", []), allowed_labels=allowed_labels)
    raw_relations = payload.get("relations", payload.get("triples", []))
    relations = _normalize_relations(raw_relations)
    return entities, relations


def run_llm_extraction(
    text: str,
    allowed_labels: list[str] | None = None,
    allowed_label_descriptions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Call the configured LLM endpoint and return defensive structured output."""
    cleaned = (text or "").strip()
    if not cleaned:
        return {
            "status": "not_started",
            "entities": [],
            "relations": [],
            "model": None,
            "base_url": None,
            "allowed_labels": allowed_labels or [],
            "error": None,
        }

    api_key = _get_env_value("GPUSTACK_API_KEY")
    base_url = _get_env_value(
        "GPUSTACK_BASE_URL",
        "GPUSTACK_API_URL",
        default=GPUSTACK_BASE_URL_DEFAULT,
    )
    model_name = _get_env_value("GPUSTACK_MODEL", default=GPUSTACK_MODEL_DEFAULT)

    if not api_key:
        return {
            "status": "configuration_error",
            "entities": [],
            "relations": [],
            "model": model_name,
            "base_url": base_url,
            "allowed_labels": allowed_labels or [],
            "error": "GPUSTACK_API_KEY is not set. Add it as an environment variable.",
        }

    client = OpenAI(base_url=base_url, api_key=api_key)
    allowed_label_set = {label.strip().upper() for label in (allowed_labels or []) if label.strip()}
    system_prompt = _build_system_prompt(
        allowed_labels=allowed_labels,
        label_descriptions=allowed_label_descriptions,
    )

    content = ""
    finish_reason = None
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            top_p=1,
            max_tokens=GPUSTACK_MAX_TOKENS_DEFAULT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cleaned},
            ],
        )
        finish_reason = response.choices[0].finish_reason
        content = (response.choices[0].message.content or "").strip()
        entities, relations = _parse_extraction_payload(content, allowed_labels=allowed_label_set)
        return {
            "status": "ok",
            "entities": entities,
            "relations": relations,
            "model": model_name,
            "base_url": base_url,
            "allowed_labels": sorted(allowed_label_set),
            "error": None,
        }
    except json.JSONDecodeError:
        if finish_reason == "length":
            try:
                retry_response = client.chat.completions.create(
                    model=model_name,
                    temperature=0,
                    top_p=1,
                    max_tokens=GPUSTACK_MAX_TOKENS_DEFAULT * 2,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": cleaned},
                    ],
                )
                content = (retry_response.choices[0].message.content or "").strip()
                entities, relations = _parse_extraction_payload(content, allowed_labels=allowed_label_set)
                return {
                    "status": "ok",
                    "entities": entities,
                    "relations": relations,
                    "model": model_name,
                    "base_url": base_url,
                    "allowed_labels": sorted(allowed_label_set),
                    "error": None,
                }
            except json.JSONDecodeError:
                pass

        return {
            "status": "parse_error",
            "entities": [],
            "relations": [],
            "model": model_name,
            "base_url": base_url,
            "allowed_labels": sorted(allowed_label_set),
            "error": "LLM response was not valid JSON.",
            "raw_response_preview": content[:500],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "api_error",
            "entities": [],
            "relations": [],
            "model": model_name,
            "base_url": base_url,
            "allowed_labels": sorted(allowed_label_set),
            "error": str(exc),
        }

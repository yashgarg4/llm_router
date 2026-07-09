"""Optional output verification hooks.

A model can return a 200 and still be *wrong for the caller's purpose* — empty
text, or JSON that doesn't match the expected shape. `verify` turns that into
a signal the router can act on: a failed verification escalates to a stronger
model **via the same fallback path as a 5xx**.

⚠️  Verification-triggered escalation is the exact silent cost-explosion
vector this project exists around (see INTERNAL_NOTES §1): a verifier that
starts failing systemically escalates *everything*. So every
verification-triggered escalation is logged and counted (the logging happens
in the router; this module just decides pass/fail).

A `verify_spec` is a dict naming a check:
    {"check": "non_empty"}
    {"check": "json"}                              # must parse as JSON
    {"check": "json_schema", "required": ["a", "b"],
     "types": {"a": "str", "b": "int"}}            # keys present + typed

`json_schema` is intentionally a lightweight required-keys/types check — no
extra dependency. It is enough to catch the "provider changed output
formatting" failure mode without pulling in a full JSON-Schema validator.
"""

from __future__ import annotations

import json
import logging

from .client import ModelResponse

logger = logging.getLogger("llmrouter.verify")

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "str": str,
    "string": str,
    "int": int,
    "float": (int, float),
    "number": (int, float),
    "bool": bool,
    "list": list,
    "array": list,
    "dict": dict,
    "object": dict,
}


def _check_non_empty(response: ModelResponse) -> bool:
    return bool(response.text and response.text.strip())


def _parse_json(response: ModelResponse):
    """Parse response text as JSON, tolerating a ```json fenced block."""
    text = response.text.strip()
    if text.startswith("```"):
        # strip a leading fence line and a trailing fence
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _check_json(response: ModelResponse) -> bool:
    try:
        _parse_json(response)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _check_json_schema(response: ModelResponse, spec: dict) -> bool:
    try:
        obj = _parse_json(response)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(obj, dict):
        return False
    for key in spec.get("required", []):
        if key not in obj:
            return False
    for key, type_name in (spec.get("types") or {}).items():
        expected = _TYPE_MAP.get(type_name)
        if expected is None:
            raise ValueError(f"unknown type in verify_spec types: {type_name!r}")
        if key in obj:
            # bool is a subclass of int; guard so an int type doesn't accept True
            if expected in (int, (int, float)) and isinstance(obj[key], bool):
                return False
            if not isinstance(obj[key], expected):
                return False
    return True


def verify(response: ModelResponse, spec: dict | None) -> bool:
    """Return True if ``response`` satisfies ``spec``. No spec == no check."""
    if not spec:
        return True
    check = spec.get("check")
    if check == "non_empty":
        return _check_non_empty(response)
    if check == "json":
        return _check_json(response)
    if check == "json_schema":
        return _check_json_schema(response, spec)
    raise ValueError(f"unknown verify check: {check!r}")

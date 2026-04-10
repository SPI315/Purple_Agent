import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, APIError, RateLimitError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message


load_dotenv()


PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"
DEFAULT_MODEL = "qwen/qwen3.6-plus:free"
DEFAULT_PROVIDER = "auto"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_TIMEOUT = 45.0
MAX_RECENT_TURNS = 12
MAX_PROVIDER_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 1.5
TOOL_SECTION_HEADERS = (
    "available tools:",
    "tool list:",
    "tools:",
)
TAU2_TOOLS_ANCHOR = "Here's a list of tools you can use"
HANDOFF_PHRASES = (
    "transfer",
    "human agent",
    "human representative",
    "escalate",
    "specialist",
    "another team",
    "supervisor",
)
INCAPABILITY_PHRASES = (
    "can't access",
    "cannot access",
    "unable to access",
    "don't have access",
    "do not have access",
    "can't check",
    "cannot check",
    "unable to check",
    "can't look up",
    "cannot look up",
    "unable to look up",
)
BROAD_CLARIFY_PHRASES = (
    "clarify your request",
    "clarify the request",
    "clarify your issue",
    "please clarify",
    "how can i help",
    "how may i help",
)
FALLBACK_ACTION = {
    "name": "respond",
    "arguments": {"content": "I'm sorry, I couldn't complete that request right now."},
}
USER_ID_PATTERN = re.compile(r"\b[a-z]+(?:_[a-z]+)+_\d{3,}\b", re.IGNORECASE)
RESERVATION_ID_PATTERN = re.compile(r"\b[A-Z0-9]{6}\b")
RESERVATION_ID_STOPWORDS = {
    "ENTIRE",
    "PLEASE",
    "NUMBER",
    "STATUS",
    "RESULT",
    "ACTIVE",
    "ERROR",
}
OPERATOR_REQUEST_PHRASES = (
    "operator",
    "human",
    "representative",
    "agent",
    "someone real",
    "person",
)
INTENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cancel", ("cancel", "refund", "void")),
    ("change", ("change", "modify", "move", "reschedule", "switch", "update")),
    ("check", ("check", "lookup", "look up", "status", "details", "find", "see")),
    ("book", ("book", "reserve", "purchase", "buy")),
    ("operator", ("operator", "human", "representative", "person", "agent")),
)
TOOL_RESULT_MARKERS = (
    "tool result",
    "result:",
    "env:",
    "reservation data",
    "flight data",
    "lookup result",
    "search result",
)
TRANSFER_TOOL_TOKENS = (
    "transfer_to_human_agents",
    "transfer",
    "human",
    "escalate",
)
TRANSFER_HOLD_MESSAGE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def validate_action(
    action: Any,
    allowed_tools: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise ValueError("Action must be a JSON object.")

    name = action.get("name")
    arguments = action.get("arguments")

    if not isinstance(name, str) or not name:
        raise ValueError("Action must include a non-empty string 'name'.")
    if not isinstance(arguments, dict):
        raise ValueError("Action must include an object 'arguments'.")

    if name == "respond":
        content = arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("'respond' action requires non-empty arguments.content.")
        return action

    if allowed_tools is not None and name not in allowed_tools:
        raise ValueError("Action name is not present in the runtime tool list.")

    return action


def parse_action(
    raw_output: str,
    allowed_tools: set[str] | None = None,
) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if isinstance(parsed, list):
        raise ValueError("Top-level JSON arrays are not allowed.")

    return validate_action(parsed, allowed_tools=allowed_tools)


def summarize_message(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= 240:
        return normalized
    return normalized[:237].rstrip() + "..."


def extract_identifiers(text: str) -> dict[str, list[str]]:
    reservation_ids: list[str] = []
    user_ids: list[str] = []

    for match in RESERVATION_ID_PATTERN.finditer(text):
        token = match.group(0)
        lowered = token.lower()
        if lowered in {"json", "http", "text"} or token.upper() in RESERVATION_ID_STOPWORDS:
            continue
        reservation_ids.append(token)

    for match in USER_ID_PATTERN.finditer(text):
        user_ids.append(match.group(0))

    return {
        "reservation_ids": list(dict.fromkeys(reservation_ids)),
        "user_ids": list(dict.fromkeys(user_ids)),
    }


def build_fallback_action(
    input_text: str,
    *,
    context_text: str | None = None,
) -> dict[str, Any]:
    del input_text, context_text
    return {
        "name": "respond",
        "arguments": {"content": FALLBACK_ACTION["arguments"]["content"]},
    }


def _dedupe_append(target: list[str], values: list[str]) -> None:
    existing = set(target)
    for value in values:
        if value not in existing:
            target.append(value)
            existing.add(value)


def _extract_tool_objects(tool_block: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    current_lines: list[str] = []
    brace_depth = 0
    started = False

    for line in tool_block.splitlines():
        stripped = line.strip()
        if not started:
            if not stripped:
                continue
            if not stripped.startswith("{"):
                break
            started = True

        if not stripped and brace_depth == 0:
            continue

        if brace_depth == 0 and current_lines and not stripped.startswith("{"):
            break

        current_lines.append(line)
        brace_depth += line.count("{")
        brace_depth -= line.count("}")

        if started and brace_depth == 0 and current_lines:
            raw_object = "\n".join(current_lines).strip()
            current_lines = []
            if not raw_object:
                continue
            try:
                parsed = json.loads(raw_object)
            except json.JSONDecodeError:
                break
            if isinstance(parsed, dict):
                objects.append(parsed)

    return objects


def format_runtime_tool_contract(tool_specs: dict[str, dict[str, Any]]) -> str:
    if not tool_specs:
        return "Runtime tools: none"

    lines = ["Runtime tools available right now:"]
    for name in sorted(tool_specs):
        spec = tool_specs[name]
        description = spec.get("description")
        parameters = spec.get("parameters")
        properties: list[str] = []
        required: list[str] = []
        if isinstance(parameters, dict):
            raw_properties = parameters.get("properties")
            if isinstance(raw_properties, dict):
                for field_name, field_spec in raw_properties.items():
                    if not isinstance(field_name, str):
                        continue
                    field_type = ""
                    if isinstance(field_spec, dict):
                        raw_type = field_spec.get("type")
                        if isinstance(raw_type, str):
                            field_type = raw_type
                    properties.append(f"{field_name}:{field_type}" if field_type else field_name)
            raw_required = parameters.get("required")
            if isinstance(raw_required, list):
                required = [item for item in raw_required if isinstance(item, str)]

        parts = [f"- {name}"]
        if isinstance(description, str) and description.strip():
            parts.append(f"description={description.strip()}")
        if properties:
            parts.append(f"fields={', '.join(properties)}")
        if required:
            parts.append(f"required={', '.join(required)}")
        lines.append("; ".join(parts))

    return "\n".join(lines)


def extract_allowed_tools_from_tau2_prompt(text: str) -> set[str]:
    anchor_start = text.find(TAU2_TOOLS_ANCHOR)
    if anchor_start == -1:
        return set()

    array_start = text.find("[", anchor_start)
    if array_start == -1:
        return set()

    depth = 0
    in_string = False
    escaped = False
    array_end = -1
    for index in range(array_start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "[":
            depth += 1
            continue
        if char == "]":
            depth -= 1
            if depth == 0:
                array_end = index
                break

    if array_end == -1:
        return set()

    raw_array = text[array_start : array_end + 1]
    try:
        parsed = json.loads(raw_array)
    except json.JSONDecodeError:
        return set()

    if not isinstance(parsed, list):
        return set()

    names: set[str] = set()
    for tool in parsed:
        if not isinstance(tool, dict):
            continue
        function_obj = tool.get("function")
        if isinstance(function_obj, dict) and isinstance(function_obj.get("name"), str):
            if function_obj["name"] != "respond":
                names.add(function_obj["name"])
            continue
        if isinstance(tool.get("name"), str) and tool["name"] != "respond":
            names.add(tool["name"])
    return names


def extract_allowed_tools(input_text: str) -> set[str]:
    tau2_tools = extract_allowed_tools_from_tau2_prompt(input_text)
    if tau2_tools:
        return tau2_tools

    lowered = input_text.lower()
    start_index = -1
    for header in TOOL_SECTION_HEADERS:
        start_index = lowered.find(header)
        if start_index != -1:
            start_index += len(header)
            break

    if start_index == -1:
        return set()

    tool_block = input_text[start_index:]
    allowed_tools: set[str] = set()
    for tool_object in _extract_tool_objects(tool_block):
        name = tool_object.get("name")
        if isinstance(name, str) and name and name != "respond":
            allowed_tools.add(name)
    return allowed_tools


def extract_tool_specs(input_text: str) -> dict[str, dict[str, Any]]:
    tau2_tools = extract_allowed_tools_from_tau2_prompt(input_text)
    specs: dict[str, dict[str, Any]] = {}
    if tau2_tools:
        anchor_start = input_text.find(TAU2_TOOLS_ANCHOR)
        array_start = input_text.find("[", anchor_start)
        depth = 0
        in_string = False
        escaped = False
        array_end = -1
        for index in range(array_start, len(input_text)):
            char = input_text[index]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "[":
                depth += 1
                continue
            if char == "]":
                depth -= 1
                if depth == 0:
                    array_end = index
                    break
        if array_start != -1 and array_end != -1:
            try:
                parsed = json.loads(input_text[array_start : array_end + 1])
            except json.JSONDecodeError:
                parsed = []
            if isinstance(parsed, list):
                for tool in parsed:
                    if not isinstance(tool, dict):
                        continue
                    function_obj = tool.get("function")
                    if not isinstance(function_obj, dict):
                        continue
                    name = function_obj.get("name")
                    if not isinstance(name, str) or not name or name == "respond":
                        continue
                    parameters = function_obj.get("parameters")
                    if not isinstance(parameters, dict):
                        parameters = {}
                    specs[name] = {
                        "name": name,
                        "description": function_obj.get("description", ""),
                        "parameters": parameters,
                    }
                if specs:
                    return specs

    lowered = input_text.lower()
    start_index = -1
    for header in TOOL_SECTION_HEADERS:
        start_index = lowered.find(header)
        if start_index != -1:
            start_index += len(header)
            break

    if start_index == -1:
        return {}

    tool_block = input_text[start_index:]
    for tool_object in _extract_tool_objects(tool_block):
        name = tool_object.get("name")
        if not isinstance(name, str) or not name or name == "respond":
            continue
        parameters = tool_object.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        specs[name] = {
            "name": name,
            "description": tool_object.get("description", ""),
            "parameters": parameters,
        }
    return specs


def policy_allows_handoff(prompt: str | None) -> bool:
    if not prompt:
        return False
    lowered = prompt.lower()
    policy_markers = ("policy", "rule", "instruction", "must", "should", "requires")
    return any(marker in lowered for marker in policy_markers) and any(
        phrase in lowered for phrase in HANDOFF_PHRASES
    )


def has_lookup_or_action_tools(allowed_tools: set[str] | None) -> bool:
    return bool(allowed_tools)


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def is_disallowed_respond_content(
    content: str,
    *,
    initial_prompt: str | None,
    allowed_tools: set[str] | None,
    current_input_text: str,
) -> bool:
    if _contains_any_phrase(content, HANDOFF_PHRASES) and not policy_allows_handoff(initial_prompt):
        return True

    if has_lookup_or_action_tools(allowed_tools) and _contains_any_phrase(content, INCAPABILITY_PHRASES):
        return True

    return False


def extract_intent(text: str) -> str | None:
    lowered = text.lower()
    if any(phrase in lowered for phrase in ("most recent booking", "latest booking", "last booking", "most recent reservation")):
        return "check"
    for intent, keywords in INTENT_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return intent
    return None


def _tool_text(spec: dict[str, Any]) -> str:
    parameters = spec.get("parameters")
    properties = []
    required = []
    if isinstance(parameters, dict):
        raw_properties = parameters.get("properties")
        if isinstance(raw_properties, dict):
            properties = list(raw_properties.keys())
        raw_required = parameters.get("required")
        if isinstance(raw_required, list):
            required = [item for item in raw_required if isinstance(item, str)]
    tokens = [spec.get("name", ""), spec.get("description", ""), " ".join(properties), " ".join(required)]
    return " ".join(token for token in tokens if isinstance(token, str)).lower()


def _tool_required_fields(spec: dict[str, Any]) -> list[str]:
    parameters = spec.get("parameters")
    if not isinstance(parameters, dict):
        return []
    required = parameters.get("required")
    if not isinstance(required, list):
        return []
    return [item for item in required if isinstance(item, str)]


def _tool_optional_fields(spec: dict[str, Any]) -> list[str]:
    parameters = spec.get("parameters")
    if not isinstance(parameters, dict):
        return []
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return []
    return [key for key in properties.keys() if isinstance(key, str)]


def _is_recent_booking_request(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in ("most recent booking", "latest booking", "last booking", "most recent reservation"))


def _is_operator_request(text: str) -> bool:
    lowered = text.lower()
    return (
        any(phrase in lowered for phrase in OPERATOR_REQUEST_PHRASES)
        and any(phrase in lowered for phrase in ("want", "need", "speak", "talk", "connect", "transfer"))
    )


def _looks_like_tool_result(text: str) -> bool:
    lowered = text.lower()
    stripped = text.strip()
    return (
        any(marker in lowered for marker in TOOL_RESULT_MARKERS)
        or lowered.startswith("error:")
        or _looks_like_empty_result(stripped)
        or (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    )


def _is_lookup_tool(spec: dict[str, Any]) -> bool:
    text = _tool_text(spec)
    return any(token in text for token in ("lookup", "search", "find", "list", "retrieve", "get", "recent", "latest"))


def _is_action_tool(spec: dict[str, Any], intent: str | None) -> bool:
    text = _tool_text(spec)
    if intent == "cancel":
        return any(token in text for token in ("cancel", "refund", "void"))
    if intent == "change":
        return any(token in text for token in ("change", "modify", "update", "reschedule"))
    if intent == "book":
        return any(token in text for token in ("book", "create", "purchase", "reserve"))
    return False


def _tool_uses_reservation_id(spec: dict[str, Any]) -> bool:
    fields = _tool_optional_fields(spec) + _tool_required_fields(spec)
    return any(any(token in field.lower() for token in ("reservation", "booking", "confirmation")) for field in fields)


def _tool_uses_user_id(spec: dict[str, Any]) -> bool:
    fields = _tool_optional_fields(spec) + _tool_required_fields(spec)
    return any(any(token in field.lower() for token in ("user", "customer", "account", "profile")) for field in fields)


def _tool_result_indicates_not_found(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("not found", "no reservation", "no booking", "none found", "unable to locate"))


def _tool_result_indicates_completion(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("success", "completed", "cancelled", "canceled", "confirmed", "updated", "booked"))


def _is_transfer_tool(spec: dict[str, Any]) -> bool:
    text = _tool_text(spec)
    return any(token in text for token in TRANSFER_TOOL_TOKENS)


def extract_structured_facts(text: str) -> dict[str, Any]:
    lowered = text.lower()
    identifiers = extract_identifiers(text)
    status: str | None = None
    for candidate in ("active", "confirmed", "cancelled", "canceled", "pending", "eligible", "ineligible"):
        if re.search(rf"\b{re.escape(candidate)}\b", lowered):
            status = candidate
            break

    completion: str | None = None
    if any(token in lowered for token in ("cancelled", "canceled", "cancellation complete")):
        completion = "cancelled"
    elif any(token in lowered for token in ("updated", "change complete", "rescheduled")):
        completion = "updated"
    elif any(token in lowered for token in ("booked", "booking complete", "reservation created")):
        completion = "booked"

    facts: dict[str, Any] = {
        "reservation_ids": identifiers["reservation_ids"],
        "user_ids": identifiers["user_ids"],
        "status": status,
        "not_found": _tool_result_indicates_not_found(text),
        "completion": completion,
        "latest_requested": _is_recent_booking_request(text),
    }

    if "eligible" in lowered:
        facts["eligibility"] = "eligible"
    elif "ineligible" in lowered or "not eligible" in lowered:
        facts["eligibility"] = "ineligible"

    return facts


def _merge_identifier_state(
    target: dict[str, list[str]],
    new_values: dict[str, list[str]],
) -> None:
    _dedupe_append(target["reservation_ids"], new_values.get("reservation_ids", []))
    _dedupe_append(target["user_ids"], new_values.get("user_ids", []))


def extract_user_focus_text(text: str) -> str:
    user_lines = re.findall(r"(?:^|\n)\s*user:\s*(.+)", text, flags=re.IGNORECASE)
    if user_lines:
        return "\n".join(user_lines)

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("policy:") or lowered.startswith("available tools:"):
            continue
        if stripped.startswith("{") or stripped.startswith("[") or stripped.startswith("]"):
            continue
        if '"name"' in stripped or '"parameters"' in stripped or '"function"' in stripped:
            continue
        lines.append(stripped)
    return "\n".join(lines)


def extract_requested_tasks(text: str) -> list[dict[str, str]]:
    user_text = extract_user_focus_text(text) or text
    normalized = " ".join(user_text.split())
    if not normalized:
        return []

    clauses = [
        chunk.strip(" .,;")
        for chunk in re.split(r"(?i)\b(?:and|also)\b", normalized)
        if chunk.strip(" .,;")
    ]
    tasks: list[dict[str, str]] = []
    last_explicit_intent: str | None = None
    for clause in clauses:
        identifiers = extract_identifiers(clause)
        reservation_ids = identifiers.get("reservation_ids", [])
        user_ids = identifiers.get("user_ids", [])
        intent = extract_intent(clause)
        if intent:
            last_explicit_intent = intent
        elif reservation_ids and last_explicit_intent in {"cancel", "change", "check", "book"}:
            intent = last_explicit_intent
        if reservation_ids:
            for reservation_id in reservation_ids:
                tasks.append(
                    {
                        "intent": intent or "check",
                        "reservation_id": reservation_id,
                        "summary": summarize_message(clause),
                        "status": "pending",
                    }
                )
            continue
        if user_ids and intent:
            for user_id in user_ids:
                tasks.append(
                    {
                        "intent": intent,
                        "user_id": user_id,
                        "summary": summarize_message(clause),
                        "status": "pending",
                    }
                )

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for task in tasks:
        key = (
            task.get("intent", ""),
            task.get("reservation_id", ""),
            task.get("user_id", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(task)
    return deduped


def _looks_like_empty_result(text: str) -> bool:
    stripped = text.strip()
    return stripped in {"[]", "{}", "null", "None"}


def split_provider_model(model: str) -> tuple[str | None, str]:
    if "/" not in model:
        return None, model
    provider, candidate_model = model.split("/", 1)
    if provider in {"openai", "gemini", "deepseek", "openrouter"}:
        return provider, candidate_model
    return None, model


def resolve_provider(
    model: str,
    configured_provider: str,
    openai_base_url: str,
    has_openrouter_key: bool,
) -> tuple[str, str]:
    prefixed_provider, stripped_model = split_provider_model(model)
    if configured_provider != "auto":
        if configured_provider == "openrouter":
            return configured_provider, model
        return configured_provider, stripped_model

    if prefixed_provider:
        if prefixed_provider == "openrouter":
            return prefixed_provider, model
        return prefixed_provider, stripped_model

    if has_openrouter_key or "openrouter.ai" in openai_base_url:
        return "openrouter", model

    return "openai", model


def resolve_provider_config(model: str) -> dict[str, str]:
    configured_provider = os.getenv("AGENT_PROVIDER", DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    provider, client_model = resolve_provider(
        model=model,
        configured_provider=configured_provider,
        openai_base_url=openai_base_url,
        has_openrouter_key=bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
    )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = openai_base_url or DEFAULT_OPENAI_BASE_URL
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        base_url = os.getenv("GEMINI_BASE_URL", "").strip() or DEFAULT_GEMINI_BASE_URL
    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip() or DEFAULT_DEEPSEEK_BASE_URL
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        base_url = openai_base_url or os.getenv("OPENROUTER_BASE_URL", "").strip() or DEFAULT_OPENROUTER_BASE_URL
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return {
        "provider": provider,
        "model": client_model,
        "api_key": api_key,
        "base_url": base_url,
    }


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class Agent:
    def __init__(self) -> None:
        configure_logging()
        self.model = os.getenv("AGENT_LLM", DEFAULT_MODEL)
        provider_config = resolve_provider_config(self.model) if self.model not in {"mock", "test"} else {}
        self.logger = logging.getLogger(__name__)
        self.system_prompt = load_system_prompt()
        self.provider = provider_config.get("provider", "mock")
        self.client_model = provider_config.get("model", self.model)
        self.base_url = provider_config.get("base_url", "")
        self.api_key = provider_config.get("api_key", "")
        self.timeout = float(os.getenv("OPENAI_TIMEOUT", str(DEFAULT_TIMEOUT)))
        self.allow_empty_toolset_debug = os.getenv("AGENT_ALLOW_EMPTY_TOOLSET_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.client = None
        if self.model not in {"mock", "test"}:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=1,
            )
        self.initial_prompt: str | None = None
        self.current_snapshot: str | None = None
        self.allowed_tools: set[str] | None = None
        self.tool_specs: dict[str, dict[str, Any]] = {}
        self.turn_history: list[dict[str, str]] = []
        self.current_input_text = ""
        self.intent_history: list[str] = []
        self.current_intent: str | None = None
        self.goal_history: list[dict[str, str]] = []
        self.active_goal: dict[str, str] | None = None
        self.pending_tasks: list[dict[str, Any]] = []
        self.completed_tasks: list[dict[str, Any]] = []
        self.active_task: dict[str, Any] | None = None
        self.known_identifiers = {
            "reservation_ids": [],
            "user_ids": [],
        }
        self.tool_journal: list[dict[str, str | None]] = []
        self.structured_facts: dict[str, Any] = {
            "status": None,
            "eligibility": None,
            "not_found": False,
            "completion": None,
            "latest_requested": False,
        }
        self.last_action_name: str | None = None
        self.awaiting_tool_result = False
        self.operator_request_count = 0

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        self.current_input_text = input_text
        self.logger.info(
            "Received request with %s characters of input.",
            len(input_text),
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Selecting next action..."),
        )

        if not input_text.strip():
            action_json = json.dumps(build_fallback_action(input_text), ensure_ascii=True)
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=action_json))],
                name="Action",
            )
            return

        self._store_user_turn(input_text)
        action = self._generate_action()
        action_json = json.dumps(action, ensure_ascii=True)
        self.turn_history.append({"role": "assistant", "content": action_json})
        self._trim_turn_history()
        self._record_action(action)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=action_json))],
            name="Action",
        )

    def _record_action(self, action: dict[str, Any]) -> None:
        action_name = action.get("name")
        if not isinstance(action_name, str):
            return
        self.last_action_name = action_name
        matched_task = self._match_task_for_action(action)
        if matched_task is not None:
            self.active_task = matched_task
            self.active_task["status"] = "in_progress"
        arguments = action.get("arguments")
        if isinstance(arguments, dict):
            self._update_structured_facts(extract_structured_facts(json.dumps(arguments)))
        inferred_intent = self._infer_intent_from_action(action)
        if inferred_intent:
            if not self.intent_history or self.intent_history[-1] != inferred_intent:
                self.intent_history.append(inferred_intent)
            self.current_intent = inferred_intent
            self._update_goal_state(intent=inferred_intent, source_text=json.dumps(action, ensure_ascii=True))
        if action_name == "respond":
            self.awaiting_tool_result = False
            return
        self.structured_facts["not_found"] = False
        self.structured_facts["completion"] = None
        self.awaiting_tool_result = True
        task_summary = None
        if self.active_task is not None:
            self.active_task["last_tool_name"] = action_name
            task_summary = self.active_task.get("summary")
        self.tool_journal.append({"name": action_name, "result_summary": None, "task_summary": task_summary})

    def _infer_intent_from_action(self, action: dict[str, Any]) -> str | None:
        action_name = action.get("name")
        if not isinstance(action_name, str):
            return None
        if action_name == "respond":
            content = str(action.get("arguments", {}).get("content", "")).lower()
            if _contains_any_phrase(content, HANDOFF_PHRASES):
                return "operator"
            return self.current_intent

        tool_text = action_name.lower()
        spec = self.tool_specs.get(action_name)
        if spec is not None:
            tool_text = _tool_text(spec)

        if _is_transfer_tool(spec or {"name": action_name, "description": "", "parameters": {}}):
            return "operator"
        if _is_lookup_tool(spec or {"name": action_name, "description": "", "parameters": {}}):
            user_text = extract_user_focus_text(self.current_snapshot or self.current_input_text)
            return extract_intent(user_text) or self.current_intent or "check"
        if any(token in tool_text for token in ("cancel", "refund", "void")):
            return "cancel"
        if any(token in tool_text for token in ("change", "modify", "update", "reschedule")):
            return "change"
        if any(token in tool_text for token in ("book", "create", "purchase", "reserve")):
            return "book"
        return self.current_intent

    def _update_goal_state(self, *, intent: str, source_text: str) -> None:
        goal_summary = summarize_message(source_text)
        if self.active_goal and self.active_goal.get("intent") == intent:
            self.active_goal["summary"] = goal_summary
            return
        goal = {"intent": intent, "summary": goal_summary}
        self.goal_history.append(goal)
        self.active_goal = goal

    def _merge_requested_tasks(self, tasks: list[dict[str, str]]) -> None:
        for task in tasks:
            reservation_id = task.get("reservation_id")
            user_id = task.get("user_id")
            intent = task.get("intent")
            existing = next(
                (
                    candidate
                    for candidate in self.pending_tasks
                    if candidate.get("reservation_id") == reservation_id
                    and candidate.get("user_id") == user_id
                    and candidate.get("intent") == intent
                ),
                None,
            )
            if existing is not None:
                existing["summary"] = task.get("summary", existing.get("summary", ""))
                if existing.get("status") == "completed":
                    existing["status"] = "pending"
                continue
            self.pending_tasks.append(
                {
                    "intent": intent,
                    "reservation_id": reservation_id,
                    "user_id": user_id,
                    "summary": task.get("summary", ""),
                    "status": task.get("status", "pending"),
                    "last_tool_name": None,
                    "last_result_summary": None,
                    "facts": {},
                }
            )
        if self.active_task is None:
            self.active_task = self._find_next_pending_task()

    def _find_next_pending_task(self) -> dict[str, Any] | None:
        for task in self.pending_tasks:
            if task.get("status") != "completed":
                return task
        return None

    def _match_task_for_action(self, action: dict[str, Any]) -> dict[str, Any] | None:
        arguments = action.get("arguments")
        reservation_id = None
        user_id = None
        if isinstance(arguments, dict):
            for key, value in arguments.items():
                lowered = key.lower()
                if not isinstance(value, str):
                    continue
                if any(token in lowered for token in ("reservation", "booking", "confirmation")):
                    reservation_id = value
                elif any(token in lowered for token in ("user", "customer", "account", "profile")):
                    user_id = value

        if reservation_id:
            for task in self.pending_tasks:
                if task.get("reservation_id") == reservation_id and task.get("status") != "completed":
                    return task
        if user_id:
            for task in self.pending_tasks:
                if task.get("user_id") == user_id and task.get("status") != "completed":
                    return task
        if self.active_task and self.active_task.get("status") != "completed":
            return self.active_task
        return self._find_next_pending_task()

    def _mark_active_task_completed(self) -> None:
        if self.active_task is None:
            return
        self.active_task["status"] = "completed"
        self.completed_tasks.append(dict(self.active_task))
        self.active_task = self._find_next_pending_task()

    def _update_structured_facts(self, facts: dict[str, Any]) -> None:
        reservation_ids = facts.get("reservation_ids")
        if isinstance(reservation_ids, list):
            _dedupe_append(self.known_identifiers["reservation_ids"], [item for item in reservation_ids if isinstance(item, str)])
        user_ids = facts.get("user_ids")
        if isinstance(user_ids, list):
            _dedupe_append(self.known_identifiers["user_ids"], [item for item in user_ids if isinstance(item, str)])

        for key in ("status", "eligibility", "completion"):
            value = facts.get(key)
            if isinstance(value, str) and value:
                self.structured_facts[key] = value

        if facts.get("not_found"):
            self.structured_facts["not_found"] = True
        if facts.get("latest_requested"):
            self.structured_facts["latest_requested"] = True
        if self.active_task is not None:
            task_facts = self.active_task.setdefault("facts", {})
            if isinstance(task_facts, dict):
                task_facts.update(facts)
            if reservation_ids and not self.active_task.get("reservation_id"):
                self.active_task["reservation_id"] = reservation_ids[-1]
            if user_ids and not self.active_task.get("user_id"):
                self.active_task["user_id"] = user_ids[-1]

    def _store_user_turn(self, input_text: str) -> None:
        self.current_snapshot = input_text
        extracted_tools = extract_tool_specs(input_text)
        if self.initial_prompt is None:
            self.initial_prompt = input_text
            self.allowed_tools = set(extracted_tools) or extract_allowed_tools(input_text)
            self.tool_specs.update(extracted_tools)
            self.logger.info(
                "Initialized context with %s allowed runtime tools.",
                len(self.allowed_tools),
            )
            if not self.allowed_tools:
                self.logger.error(
                    "Runtime tools extraction failed: allowed_tools is empty on first turn"
                )
        else:
            if self.allowed_tools is None:
                self.allowed_tools = set()
            self.allowed_tools.update(extracted_tools.keys())
            self.allowed_tools.update(extract_allowed_tools(input_text))
            self.tool_specs.update(extracted_tools)
            self.turn_history.append({"role": "user", "content": input_text})
            self._trim_turn_history()

        user_focus_text = extract_user_focus_text(input_text)
        if _is_operator_request(user_focus_text):
            self.operator_request_count += 1

        if not (self.awaiting_tool_result and self.last_action_name and _looks_like_tool_result(input_text)):
            requested_tasks = extract_requested_tasks(user_focus_text)
            if requested_tasks:
                self._merge_requested_tasks(requested_tasks)

        if self.awaiting_tool_result and self.last_action_name and _looks_like_tool_result(input_text):
            facts = extract_structured_facts(input_text)
            self._update_structured_facts(facts)
            if self.tool_journal and self.tool_journal[-1].get("name") == self.last_action_name:
                self.tool_journal[-1]["result_summary"] = summarize_message(input_text)
            else:
                self.tool_journal.append(
                    {
                        "name": self.last_action_name,
                        "result_summary": summarize_message(input_text),
                        "task_summary": self.active_task.get("summary") if self.active_task else None,
                    }
                )
            if self.active_task is not None:
                self.active_task["last_result_summary"] = summarize_message(input_text)
            self.awaiting_tool_result = False

    def _trim_turn_history(self) -> None:
        if len(self.turn_history) > MAX_RECENT_TURNS:
            self.turn_history = self.turn_history[-MAX_RECENT_TURNS:]

    def _generate_action(self) -> dict[str, Any]:
        deterministic_action = self._choose_deterministic_action()
        if deterministic_action is not None:
            return deterministic_action

        runtime_messages = self._build_runtime_messages()
        try:
            raw_output = self._call_model(runtime_messages)
        except (
            APITimeoutError,
            APIConnectionError,
            APIError,
            RateLimitError,
            TimeoutError,
            OSError,
            ValueError,
        ) as exc:
            self.logger.exception(
                "Provider request failed; using fallback action: %s",
                exc,
            )
            return build_fallback_action(
                self.current_input_text,
                context_text=self._build_context_text(),
            )

        action_name: str | None = None
        try:
            parsed_output = json.loads(raw_output)
            if isinstance(parsed_output, dict):
                maybe_name = parsed_output.get("name")
                if isinstance(maybe_name, str):
                    action_name = maybe_name
        except json.JSONDecodeError:
            action_name = None

        validation_tools = self.allowed_tools
        if validation_tools == set():
            if action_name == "respond":
                validation_tools = None
            elif self.allow_empty_toolset_debug:
                self.logger.error(
                    "allowed_tools is empty; skipping tool-name validation for this turn because AGENT_ALLOW_EMPTY_TOOLSET_DEBUG is enabled"
                )
                validation_tools = None
            else:
                self.logger.error(
                    "allowed_tools is empty on validation path; returning generic fallback"
                )
                return build_fallback_action(
                    self.current_input_text,
                    context_text=self._build_context_text(),
                )
        self.logger.info(
            "Validating action: name=%r allowed_tools_count=%d allowed_tools=%s initial_prompt_detected=%s",
            action_name,
            len(self.allowed_tools or set()),
            sorted(list(self.allowed_tools or set()))[:20],
            self.initial_prompt is not None,
        )

        try:
            action = parse_action(raw_output, allowed_tools=validation_tools)
        except ValueError as exc:
            self.logger.warning("Primary action validation failed: %s", exc)
            self.logger.warning(
                "Action validation failed: action_name=%r not in allowed_tools=%s",
                action_name,
                sorted(list(self.allowed_tools or set()))[:20],
            )
            self.logger.debug("Primary raw output: %s", summarize_message(raw_output))
            repaired_output = self._repair_action(raw_output, validation_tools, str(exc))
            if repaired_output is None:
                return build_fallback_action(
                    self.current_input_text,
                    context_text=self._build_context_text(),
                )
            try:
                action = parse_action(repaired_output, allowed_tools=validation_tools)
            except ValueError:
                return build_fallback_action(
                    self.current_input_text,
                    context_text=self._build_context_text(),
                )

        if action["name"] == "respond":
            content = str(action["arguments"].get("content", ""))
            if is_disallowed_respond_content(
                content,
                initial_prompt=self.initial_prompt,
                allowed_tools=self.allowed_tools,
                current_input_text=self.current_input_text,
            ):
                self.logger.warning(
                    "Rejecting respond action due to disallowed surrender-style content."
                )
                return build_fallback_action(
                    self.current_input_text,
                    context_text=self._build_context_text(),
                )
        action = self._route_validated_action(action)

        return action

    def _route_validated_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if action.get("name") != "respond":
            return action
        content = str(action.get("arguments", {}).get("content", ""))
        if not _contains_any_phrase(content, HANDOFF_PHRASES):
            return action
        transfer_action = self._deterministic_transfer_action()
        if transfer_action is not None:
            return transfer_action
        return action

    def _repair_action(
        self,
        raw_output: str,
        error_text: str,
        validation_tools: set[str] | None,
    ) -> str | None:
        if self.model in {"mock", "test"}:
            return None
        repair_messages = self._build_runtime_messages()
        repair_messages.append(
            {
                "role": "system",
                "content": (
                    "Repair the previous assistant output. "
                    "Return exactly one valid JSON action object with fields name and arguments. "
                    "Use only runtime tool names already provided."
                ),
            }
        )
        repair_messages.append(
            {
                "role": "assistant",
                "content": raw_output,
            }
        )
        repair_messages.append(
            {
                "role": "user",
                "content": f"Validation error: {error_text}",
            }
        )
        try:
            return self._call_model(repair_messages)
        except (
            APITimeoutError,
            APIConnectionError,
            APIError,
            RateLimitError,
            TimeoutError,
            OSError,
            ValueError,
        ):
            return None

    def _choose_deterministic_action(self) -> dict[str, Any] | None:
        post_tool_action = self._handle_post_tool_result()
        if post_tool_action is not None:
            return post_tool_action
        return None

    def _effective_intent(self) -> str | None:
        if self.active_task and isinstance(self.active_task.get("intent"), str):
            return self.active_task["intent"]
        if self.current_intent:
            return self.current_intent
        if self.active_goal and isinstance(self.active_goal.get("intent"), str):
            return self.active_goal["intent"]
        return None

    def _deterministic_transfer_action(self) -> dict[str, Any] | None:
        if not policy_allows_handoff(self.current_snapshot or self.initial_prompt):
            return None

        for spec in self.tool_specs.values():
            if not _is_transfer_tool(spec):
                continue
            arguments = self._build_tool_arguments(spec, recent_request=False)
            if arguments is None:
                required_fields = _tool_required_fields(spec)
                if required_fields:
                    continue
                arguments = {}
            return {"name": spec["name"], "arguments": arguments}

        return None

    def _handle_post_tool_result(self) -> dict[str, Any] | None:
        if not _looks_like_tool_result(self.current_input_text):
            return None

        effective_intent = self._effective_intent()
        active_task_summary = None
        if self.active_task is not None:
            active_task_summary = self.active_task.get("summary")

        if self.last_action_name:
            transfer_spec = self.tool_specs.get(self.last_action_name)
            if transfer_spec and _is_transfer_tool(transfer_spec):
                return {
                    "name": "respond",
                    "arguments": {
                        "content": TRANSFER_HOLD_MESSAGE
                    },
                }

        if _looks_like_empty_result(self.current_input_text):
            detail = " for the current request" if active_task_summary else ""
            return {
                "name": "respond",
                "arguments": {
                    "content": f"I checked the latest tool result{detail}, but it came back empty. If you'd like, I can try a different approach with the available options."
                },
            }

        if self.structured_facts.get("not_found") or _tool_result_indicates_not_found(self.current_input_text):
            if effective_intent in {"cancel", "change", "check"}:
                return {
                    "name": "respond",
                    "arguments": {
                        "content": "I couldn't find a matching reservation from the available information. Please share your reservation ID or user ID."
                    },
                }

        if effective_intent == "check":
            status = self.structured_facts.get("status")
            if isinstance(status, str) and status:
                return {
                    "name": "respond",
                    "arguments": {
                        "content": f"I've reviewed the reservation and its current status is {status}. Let me know if you want to change or cancel it."
                    },
                }
            return {
                "name": "respond",
                "arguments": {
                    "content": "I've reviewed the reservation details from the latest tool result. Let me know if you want to change or cancel it."
                },
            }

        if effective_intent == "cancel" and (
            self.structured_facts.get("completion") == "cancelled"
            or _tool_result_indicates_completion(self.current_input_text)
        ):
            self._mark_active_task_completed()
            next_task_action = self._deterministic_tool_action(prefer_actions=True)
            if next_task_action is not None:
                return next_task_action
            return {
                "name": "respond",
                "arguments": {
                    "content": "Your cancellation has been completed."
                },
            }

        if effective_intent == "change" and (
            self.structured_facts.get("completion") == "updated"
            or _tool_result_indicates_completion(self.current_input_text)
        ):
            self._mark_active_task_completed()
            next_task_action = self._deterministic_tool_action(prefer_actions=True)
            if next_task_action is not None:
                return next_task_action
            return {
                "name": "respond",
                "arguments": {
                    "content": "Your reservation has been updated."
                },
            }

        next_action = self._deterministic_tool_action(prefer_actions=True)
        if next_action is not None:
            return next_action

        return None

    def _deterministic_tool_action(self, *, prefer_actions: bool = False) -> dict[str, Any] | None:
        if not self.tool_specs:
            return None

        recent_request = _is_recent_booking_request(self.current_input_text)
        candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        for name, spec in self.tool_specs.items():
            score = self._score_tool(spec, prefer_actions=prefer_actions)
            if score <= 0:
                continue
            arguments = self._build_tool_arguments(spec, recent_request=recent_request)
            if arguments is None:
                continue
            candidates.append((score, spec, arguments))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, spec, arguments = candidates[0]
        return {"name": spec["name"], "arguments": arguments}

    def _score_tool(self, spec: dict[str, Any], *, prefer_actions: bool = False) -> int:
        text = _tool_text(spec)
        score = 0
        is_lookup = _is_lookup_tool(spec)
        effective_intent = self._effective_intent()
        is_action = _is_action_tool(spec, effective_intent)
        if effective_intent == "cancel":
            if is_action:
                score += 8
            if is_lookup:
                score += 2
            if "reservation" in text or "booking" in text:
                score += 3
        elif effective_intent == "check":
            if is_lookup:
                score += 8
            if "reservation" in text or "booking" in text or "flight" in text:
                score += 3
        elif effective_intent == "change":
            if is_action:
                score += 8
            if is_lookup:
                score += 2
        elif effective_intent == "book":
            if is_action:
                score += 8

        if _is_recent_booking_request(self.current_input_text) and any(
            token in text for token in ("recent", "latest", "list", "search", "find")
        ):
            score += 6

        if self.known_identifiers["reservation_ids"] and _tool_uses_reservation_id(spec):
            score += 2
        if self.known_identifiers["user_ids"] and _tool_uses_user_id(spec):
            score += 2

        if effective_intent in {"cancel", "change"}:
            if self.known_identifiers["reservation_ids"] and is_action:
                score += 7
            if not self.known_identifiers["reservation_ids"] and is_lookup:
                score += 5
            if prefer_actions and is_action:
                score += 6
            if prefer_actions and is_lookup:
                score -= 3

        if effective_intent == "check" and is_lookup:
            score += 2

        if self.awaiting_tool_result and self.last_action_name == spec.get("name"):
            score -= 10

        return score

    def _build_tool_arguments(
        self,
        spec: dict[str, Any],
        *,
        recent_request: bool,
    ) -> dict[str, Any] | None:
        fields = _tool_optional_fields(spec)
        required_fields = _tool_required_fields(spec)
        if not fields and not required_fields:
            return None
        arguments: dict[str, Any] = {}
        active_reservation_id = None
        active_user_id = None
        if self.active_task is not None:
            active_reservation_id = self.active_task.get("reservation_id")
            active_user_id = self.active_task.get("user_id")
        for field in fields:
            lowered = field.lower()
            if any(token in lowered for token in ("reservation", "booking", "confirmation")):
                if isinstance(active_reservation_id, str) and active_reservation_id:
                    arguments[field] = active_reservation_id
                elif self.known_identifiers["reservation_ids"]:
                    arguments[field] = self.known_identifiers["reservation_ids"][-1]
            elif "user" in lowered or "customer" in lowered or "account" in lowered:
                if isinstance(active_user_id, str) and active_user_id:
                    arguments[field] = active_user_id
                elif self.known_identifiers["user_ids"]:
                    arguments[field] = self.known_identifiers["user_ids"][-1]
            elif recent_request and any(token in lowered for token in ("recent", "latest", "most_recent")):
                arguments[field] = True
            elif recent_request and "sort" in lowered:
                arguments[field] = "desc"
            elif recent_request and any(token in lowered for token in ("limit", "count", "top")):
                arguments[field] = 1

        if any(field not in arguments for field in required_fields):
            return None
        if not arguments and not recent_request:
            return None

        return arguments

    def _build_missing_info_response(self) -> dict[str, Any] | None:
        if self.tool_specs and self.known_identifiers["user_ids"] and not self.known_identifiers["reservation_ids"]:
            for spec in self.tool_specs.values():
                text = _tool_text(spec)
                if any(token in text for token in ("lookup", "search", "find", "list", "recent", "reservation", "booking")):
                    arguments = self._build_tool_arguments(spec, recent_request=_is_recent_booking_request(self.current_input_text))
                    if arguments is not None:
                        return {"name": spec["name"], "arguments": arguments}

        if not self.known_identifiers["reservation_ids"] and not self.known_identifiers["user_ids"]:
            return {
                "name": "respond",
                "arguments": {
                    "content": "Please share your reservation ID or user ID so I can look this up."
                },
            }
        if not self.known_identifiers["reservation_ids"] and not _is_recent_booking_request(self.current_input_text):
            return {
                "name": "respond",
                "arguments": {
                    "content": "Please share your reservation ID, or your user ID if you do not have the reservation ID."
                },
            }
        return None

    def _call_model(self, messages: list[dict[str, str]]) -> str:
        if self.model in {"mock", "test"}:
            content = json.dumps(
                {
                    "name": "respond",
                    "arguments": {"content": "Mock response for local testing."},
                },
                ensure_ascii=True,
            )
            self.logger.debug("Model raw output: %s", content)
            return content

        if not self.client:
            raise ValueError("OpenAI client is not configured.")
        if not self.api_key:
            raise ValueError(f"API key for provider '{self.provider}' is not set.")

        self.logger.info(
            "Calling provider provider=%s model=%s base_url=%s turns=%s timeout=%ss",
            self.provider,
            self.client_model,
            self.base_url,
            len(messages),
            self.timeout,
        )
        response = None
        for attempt in range(1, MAX_PROVIDER_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.client_model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                break
            except RateLimitError as exc:
                if attempt == MAX_PROVIDER_RETRIES:
                    raise
                delay = RATE_LIMIT_BACKOFF_SECONDS * attempt
                self.logger.warning(
                    "Provider rate limited request on attempt %s/%s; retrying in %.1fs: %s",
                    attempt,
                    MAX_PROVIDER_RETRIES,
                    delay,
                    exc,
                )
                time.sleep(delay)

        if response is None:
            raise ValueError("Model returned no response object.")

        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("Model returned no choices.")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Model returned empty or malformed content.")
        self.logger.info("Provider call completed successfully.")
        self.logger.debug("Model raw output: %s", content)
        return content

    def _build_runtime_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        messages.append(
            {
                "role": "system",
                "content": format_runtime_tool_contract(self.tool_specs),
            }
        )
        state_lines = [
            f"Current intent: {self._effective_intent() or 'unknown'}",
            f"Intent history: {', '.join(self.intent_history[-5:]) or 'none'}",
            (
                "Goal history: "
                + " | ".join(
                    f"{goal.get('intent', 'unknown')}: {goal.get('summary', '')}"
                    for goal in self.goal_history[-5:]
                )
                if self.goal_history
                else "Goal history: none"
            ),
            f"Known reservation IDs: {', '.join(self.known_identifiers['reservation_ids'][-3:]) or 'none'}",
            f"Known user IDs: {', '.join(self.known_identifiers['user_ids'][-3:]) or 'none'}",
            (
                "Active task: "
                + (
                    f"{self.active_task.get('intent', 'unknown')} "
                    f"{self.active_task.get('reservation_id') or self.active_task.get('user_id') or ''} "
                    f"[{self.active_task.get('status', 'pending')}]"
                ).strip()
                if self.active_task
                else "Active task: none"
            ),
            (
                "Pending tasks: "
                + " | ".join(
                    (
                        f"{task.get('intent', 'unknown')} "
                        f"{task.get('reservation_id') or task.get('user_id') or ''} "
                        f"[{task.get('status', 'pending')}]"
                    ).strip()
                    for task in self.pending_tasks[-5:]
                )
                if self.pending_tasks
                else "Pending tasks: none"
            ),
            f"Structured status: {self.structured_facts.get('status') or 'unknown'}",
            f"Structured eligibility: {self.structured_facts.get('eligibility') or 'unknown'}",
            f"Structured completion: {self.structured_facts.get('completion') or 'none'}",
            f"Structured not_found: {str(bool(self.structured_facts.get('not_found'))).lower()}",
            f"Recent tool calls: {', '.join(entry.get('name') or '' for entry in self.tool_journal[-3:]) or 'none'}",
            (
                "Recent tool results: "
                + " | ".join(
                    (entry.get("result_summary") or "")
                    for entry in self.tool_journal[-2:]
                    if entry.get("result_summary")
                )
                if self.tool_journal
                else "Recent tool results: none"
            ),
            (
                "If the current message contains the result of a previous tool call and it resolves the request, "
                "respond to the user instead of greeting again or repeating the same question."
            ),
        ]
        messages.append({"role": "system", "content": "\n".join(state_lines)})
        if self.initial_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "First task snapshot retained for long-term policy context. "
                        "Do not rely on it when it conflicts with the latest user message.\n"
                        f"First snapshot summary: {summarize_message(extract_user_focus_text(self.initial_prompt) or self.initial_prompt)}"
                    ),
                }
            )
        if self.current_snapshot is not None:
            messages.append({"role": "user", "content": self.current_snapshot})
        messages.extend(self.turn_history[-MAX_RECENT_TURNS:])
        return messages

    def _build_context_text(self) -> str:
        chunks: list[str] = []
        if self.initial_prompt:
            chunks.append(self.initial_prompt)
        for turn in self.turn_history:
            content = turn.get("content")
            if isinstance(content, str) and content:
                chunks.append(content)
        if self.current_input_text:
            chunks.append(self.current_input_text)
        return "\n".join(chunks)

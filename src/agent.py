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
        if lowered in {"json", "http", "text"}:
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


def policy_allows_handoff(prompt: str | None) -> bool:
    if not prompt:
        return False
    lowered = prompt.lower()
    return any(phrase in lowered for phrase in HANDOFF_PHRASES)


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
        self.allowed_tools: set[str] | None = None
        self.turn_history: list[dict[str, str]] = []
        self.current_input_text = ""

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

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=action_json))],
            name="Action",
        )

    def _store_user_turn(self, input_text: str) -> None:
        if self.initial_prompt is None:
            self.initial_prompt = input_text
            self.allowed_tools = extract_allowed_tools(input_text)
            self.logger.info(
                "Initialized context with %s allowed runtime tools.",
                len(self.allowed_tools),
            )
            if not self.allowed_tools:
                self.logger.error(
                    "Runtime tools extraction failed: allowed_tools is empty on first turn"
                )
            return

        self.turn_history.append({"role": "user", "content": input_text})
        self._trim_turn_history()

    def _trim_turn_history(self) -> None:
        if len(self.turn_history) > MAX_RECENT_TURNS:
            self.turn_history = self.turn_history[-MAX_RECENT_TURNS:]

    def _generate_action(self) -> dict[str, Any]:
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
            if self.allow_empty_toolset_debug:
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

        return action

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
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        if self.initial_prompt is not None:
            messages.append({"role": "user", "content": self.initial_prompt})
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

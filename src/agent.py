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
MAX_RECENT_TURNS = 6
MAX_MESSAGE_CHARS = 4000
MAX_SUMMARY_ITEMS = 8
MAX_PROVIDER_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 1.5
FALLBACK_ACTION = {
    "name": "respond",
    "arguments": {"content": "I'm sorry, could you clarify your request?"},
}
GENERIC_CLARIFY_RESPONSE = FALLBACK_ACTION["arguments"]["content"]
USER_ID_PATTERN = re.compile(r"\b[a-z]+(?:_[a-z]+)+_\d{3,}\b", re.IGNORECASE)
RESERVATION_ID_PATTERN = re.compile(r"\b[A-Z0-9]{6}\b")
TOOL_NAME_JSON_PATTERN = re.compile(r'"name"\s*:\s*"([A-Za-z][A-Za-z0-9_]*)"')
TOOL_NAME_TEXT_PATTERN = re.compile(r"\b([a-z][a-z0-9_]{2,})\b")


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def validate_action(action: Any) -> dict[str, Any]:
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


def parse_action(raw_output: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if isinstance(parsed, list):
        raise ValueError("Top-level JSON arrays are not allowed.")

    return validate_action(parsed)


def truncate_text(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 15].rstrip() + "\n...[truncated]"


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


def build_fallback_action(input_text: str) -> dict[str, Any]:
    identifiers = extract_identifiers(input_text)
    if identifiers["reservation_ids"]:
        reservation_id = identifiers["reservation_ids"][0]
        content = (
            "I'm sorry, I'm having trouble accessing the reservation right now. "
            f"Please confirm the reservation ID {reservation_id} and I'll try again."
        )
    elif identifiers["user_ids"]:
        user_id = identifiers["user_ids"][0]
        content = (
            "I'm sorry, I'm having trouble accessing the account right now. "
            f"Please confirm the user ID {user_id} and I'll try again."
        )
    else:
        content = "Could you share your reservation ID or user ID so I can look up your booking?"

    return {"name": "respond", "arguments": {"content": content}}


def extract_allowed_tools(input_text: str) -> set[str]:
    tool_names = {
        name
        for name in TOOL_NAME_JSON_PATTERN.findall(input_text)
        if name != "respond"
    }
    if tool_names:
        return tool_names

    fallback_candidates = set()
    for candidate in TOOL_NAME_TEXT_PATTERN.findall(input_text):
        if "_" not in candidate or candidate == "respond":
            continue
        if candidate.startswith(("first_", "last_", "zip_", "flight_", "user_", "reservation_")):
            continue
        fallback_candidates.add(candidate)
    return fallback_candidates


def is_generic_clarify_response(action: dict[str, Any]) -> bool:
    return (
        action.get("name") == "respond"
        and str(action.get("arguments", {}).get("content", "")).strip()
        == GENERIC_CLARIFY_RESPONSE
    )


def split_provider_model(model: str) -> tuple[str | None, str]:
    if "/" not in model:
        return None, model
    provider, candidate_model = model.split("/", 1)
    if provider in {"openai", "gemini", "deepseek", "openrouter"}:
        return provider, candidate_model
    return None, model


def resolve_provider(model: str, configured_provider: str, openai_base_url: str, has_openrouter_key: bool) -> tuple[str, str]:
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
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
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
        self.client = None
        if self.model not in {"mock", "test"}:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=1,
            )
        self.turn_history: list[dict[str, str]] = []
        self.memory_items: list[str] = []
        self.current_input_text = ""

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        self.current_input_text = input_text
        self.logger.info(
            "Received request with %s characters of input.", len(input_text)
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

        self._remember_user_input(input_text)
        action = self._generate_action()
        action_json = json.dumps(action, ensure_ascii=True)
        self.turn_history.append({"role": "assistant", "content": action_json})
        self._remember_assistant_action(action)
        self._trim_memory()

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=action_json))],
            name="Action",
        )

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
                "Provider request failed; using fallback action: %s", exc
            )
            return build_fallback_action(self.current_input_text)

        try:
            action = parse_action(raw_output)
        except Exception as exc:
            self.logger.warning("Primary action parse failed: %s", exc)
            self.logger.debug("Primary raw output: %s", raw_output)
            try:
                repaired_output = self._repair_action(
                    runtime_messages,
                    raw_output,
                    (
                        "Return the previous answer as valid JSON only. "
                        "Do not use markdown or code fences. "
                        "It must be exactly one action with keys 'name' and 'arguments'. "
                        "If the action is 'respond', include a non-empty string in arguments.content."
                    ),
                )
                action = parse_action(repaired_output)
            except (
                APITimeoutError,
                APIConnectionError,
                APIError,
                RateLimitError,
                TimeoutError,
                OSError,
                ValueError,
            ) as repair_exc:
                self.logger.warning("Repair action parse failed: %s", repair_exc)
                self.logger.debug(
                    "Repair raw output: %s",
                    repaired_output if "repaired_output" in locals() else "",
                )
                return build_fallback_action(self.current_input_text)

        return self._postprocess_action(action, runtime_messages, raw_output)

    def _postprocess_action(
        self,
        action: dict[str, Any],
        runtime_messages: list[dict[str, str]],
        raw_output: str,
    ) -> dict[str, Any]:
        allowed_tools = extract_allowed_tools(self.current_input_text)

        if action["name"] != "respond" and allowed_tools and action["name"] not in allowed_tools:
            self.logger.warning(
                "Model selected tool '%s' not present in current context. Allowed tools: %s",
                action["name"],
                sorted(allowed_tools),
            )
            repair_instruction = (
                "The selected tool is not available in the current tool list. "
                f"Use exactly one of these tools if needed: {', '.join(sorted(allowed_tools))}. "
                "If none applies, return a respond action."
            )
            try:
                repaired_output = self._repair_action(runtime_messages, raw_output, repair_instruction)
                repaired_action = parse_action(repaired_output)
                if repaired_action["name"] == "respond" or repaired_action["name"] in allowed_tools:
                    return repaired_action
            except (
                APITimeoutError,
                APIConnectionError,
                APIError,
                RateLimitError,
                TimeoutError,
                OSError,
                ValueError,
            ) as exc:
                self.logger.warning("Tool repair failed: %s", exc)
            return build_fallback_action(self.current_input_text)

        if is_generic_clarify_response(action) and extract_identifiers(self.current_input_text)["reservation_ids"] + extract_identifiers(self.current_input_text)["user_ids"]:
            self.logger.info("Replacing generic clarify response with contextual fallback.")
            return build_fallback_action(self.current_input_text)

        return action

    def _repair_action(
        self,
        runtime_messages: list[dict[str, str]],
        raw_output: str,
        instruction: str,
    ) -> str:
        repair_messages = runtime_messages + [
            {"role": "assistant", "content": raw_output},
            {"role": "user", "content": instruction},
        ]
        return self._call_model(repair_messages)

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
        if self.memory_items:
            memory_block = "Compressed memory from earlier turns:\n- " + "\n- ".join(
                self.memory_items[-MAX_SUMMARY_ITEMS:]
            )
            messages.append({"role": "system", "content": memory_block})
        messages.extend(self.turn_history[-MAX_RECENT_TURNS:])
        return messages

    def _remember_user_input(self, input_text: str) -> None:
        clipped_input = truncate_text(input_text)
        self.turn_history.append({"role": "user", "content": clipped_input})
        self.memory_items.append(f"User/context: {summarize_message(input_text)}")
        self._trim_memory()

    def _remember_assistant_action(self, action: dict[str, Any]) -> None:
        name = str(action.get("name", ""))
        arguments = action.get("arguments", {})
        if name == "respond":
            content = str(arguments.get("content", ""))
            summary = f"Agent responded to user: {summarize_message(content)}"
        else:
            rendered_args = json.dumps(arguments, ensure_ascii=True, sort_keys=True)
            summary = (
                f"Agent called tool '{name}' with arguments: "
                f"{summarize_message(rendered_args)}"
            )
        self.memory_items.append(summary)

    def _trim_memory(self) -> None:
        if len(self.turn_history) > MAX_RECENT_TURNS:
            self.turn_history = self.turn_history[-MAX_RECENT_TURNS:]
        if len(self.memory_items) > MAX_SUMMARY_ITEMS:
            self.memory_items = self.memory_items[-MAX_SUMMARY_ITEMS:]

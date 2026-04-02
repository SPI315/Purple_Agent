import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import litellm
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message


load_dotenv()


PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"
DEFAULT_MODEL = "openai/gpt-4o-mini"
MAX_RECENT_TURNS = 6
MAX_MESSAGE_CHARS = 4000
MAX_SUMMARY_ITEMS = 8
FALLBACK_ACTION = {
    "name": "respond",
    "arguments": {"content": "I'm sorry, could you clarify your request?"},
}


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


class Agent:
    def __init__(self) -> None:
        self.model = os.getenv("AGENT_LLM", DEFAULT_MODEL)
        self.logger = logging.getLogger(__name__)
        self.system_prompt = load_system_prompt()
        self.turn_history: list[dict[str, str]] = []
        self.memory_items: list[str] = []

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Selecting next action..."),
        )

        if not input_text.strip():
            action_json = json.dumps(FALLBACK_ACTION, ensure_ascii=True)
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
        raw_output = self._call_model(runtime_messages)
        try:
            return parse_action(raw_output)
        except Exception as exc:
            self.logger.warning("Primary action parse failed: %s", exc)
            self.logger.debug("Primary raw output: %s", raw_output)
            repair_messages = runtime_messages + [
                {"role": "assistant", "content": raw_output},
                {
                    "role": "user",
                    "content": (
                        "Return the previous answer as valid JSON only. "
                        "Do not use markdown or code fences. "
                        "It must be exactly one action with keys 'name' and 'arguments'. "
                        "If the action is 'respond', include a non-empty string in arguments.content."
                    ),
                },
            ]
            try:
                repaired_output = self._call_model(repair_messages)
                return parse_action(repaired_output)
            except Exception as repair_exc:
                self.logger.warning("Repair action parse failed: %s", repair_exc)
                self.logger.debug(
                    "Repair raw output: %s",
                    repaired_output if "repaired_output" in locals() else "",
                )
                return FALLBACK_ACTION

    def _call_model(self, messages: list[dict[str, str]]) -> str:
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Model returned empty content.")
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

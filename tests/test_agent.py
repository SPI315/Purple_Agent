from typing import Any
from uuid import uuid4

import httpx
import pytest
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

from agent import Agent, parse_action, summarize_message, truncate_text, validate_action


def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_fields = frozenset(
        [
            "name",
            "description",
            "url",
            "version",
            "capabilities",
            "defaultInputModes",
            "defaultOutputModes",
            "skills",
        ]
    )

    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    if "url" in card_data and not (
        card_data["url"].startswith("http://")
        or card_data["url"].startswith("https://")
    ):
        errors.append(
            "Field 'url' must be an absolute URL starting with http:// or https://."
        )

    if "capabilities" in card_data and not isinstance(card_data["capabilities"], dict):
        errors.append("Field 'capabilities' must be an object.")

    for field in ["defaultInputModes", "defaultOutputModes"]:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    if "skills" in card_data:
        if not isinstance(card_data["skills"], list):
            errors.append("Field 'skills' must be an array of AgentSkill objects.")
        elif not card_data["skills"]:
            errors.append("Field 'skills' array is empty.")

    return errors


def _validate_task(data: dict[str, Any]) -> list[str]:
    errors = []
    if "id" not in data:
        errors.append("Task object missing required field: 'id'.")
    if "status" not in data or "state" not in data.get("status", {}):
        errors.append("Task object missing required field: 'status.state'.")
    return errors


def _validate_status_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if "status" not in data or "state" not in data.get("status", {}):
        errors.append("StatusUpdate object missing required field: 'status.state'.")
    return errors


def _validate_artifact_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if "artifact" not in data:
        errors.append("ArtifactUpdate object missing required field: 'artifact'.")
    elif (
        "parts" not in data.get("artifact", {})
        or not isinstance(data.get("artifact", {}).get("parts"), list)
        or not data.get("artifact", {}).get("parts")
    ):
        errors.append("Artifact object must have a non-empty 'parts' array.")
    return errors


def _validate_message(data: dict[str, Any]) -> list[str]:
    errors = []
    if (
        "parts" not in data
        or not isinstance(data.get("parts"), list)
        or not data.get("parts")
    ):
        errors.append("Message object must have a non-empty 'parts' array.")
    if "role" not in data or data.get("role") != "agent":
        errors.append("Message from agent must have 'role' set to 'agent'.")
    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    if "kind" not in data:
        return ["Response from agent is missing required 'kind' field."]

    kind = data.get("kind")
    validators = {
        "task": _validate_task,
        "status-update": _validate_status_update,
        "artifact-update": _validate_artifact_update,
        "message": _validate_message,
    }

    validator = validators.get(str(kind))
    if validator:
        return validator(data)

    return [f"Unknown message kind received: '{kind}'."]


def test_validate_action_rejects_empty_respond():
    with pytest.raises(ValueError):
        validate_action({"name": "respond", "arguments": {"content": ""}})


def test_parse_action_rejects_top_level_array():
    with pytest.raises(ValueError):
        parse_action('[{"name":"respond","arguments":{"content":"Hi"}}]')


def test_parse_action_accepts_valid_tool_call():
    action = parse_action('{"name":"lookup_order","arguments":{"order_id":"12345"}}')
    assert action["name"] == "lookup_order"
    assert action["arguments"]["order_id"] == "12345"


def test_truncate_text_shortens_long_inputs():
    text = "a" * 5000
    truncated = truncate_text(text, limit=100)
    assert len(truncated) <= 100
    assert truncated.endswith("...[truncated]")


def test_summarize_message_compacts_whitespace():
    summary = summarize_message("Hello   there\n\nthis is   a   test")
    assert summary == "Hello there this is a test"


def test_runtime_messages_include_compressed_memory():
    agent = Agent()
    agent._remember_user_input("User asks to change a flight tomorrow morning.")
    agent._remember_assistant_action(
        {
            "name": "respond",
            "arguments": {"content": "Please share your booking reference."},
        }
    )

    runtime_messages = agent._build_runtime_messages()
    system_messages = [msg for msg in runtime_messages if msg["role"] == "system"]

    assert len(system_messages) == 2
    assert "Compressed memory from earlier turns:" in system_messages[1]["content"]


async def send_text_message(
    text: str, url: str, context_id: str | None = None, streaming: bool = False
):
    async with httpx.AsyncClient(timeout=60) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]

    return events


def test_agent_card(agent):
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, "Agent card validation failed:\n" + "\n".join(errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent, streaming):
    events = await send_text_message("Hello", agent, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)
            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)
            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events
    assert not all_errors, "Message validation failed:\n" + "\n".join(all_errors)

from typing import Any
from uuid import uuid4

import httpx
import pytest
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart
from openai import RateLimitError

from agent import (
    Agent,
    build_fallback_action,
    extract_structured_facts,
    extract_allowed_tools,
    extract_allowed_tools_from_tau2_prompt,
    format_runtime_tool_contract,
    is_disallowed_respond_content,
    parse_action,
    policy_allows_handoff,
    resolve_provider_config,
    summarize_message,
    validate_action,
)


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


def test_validate_action_rejects_tool_outside_runtime_list():
    with pytest.raises(ValueError, match="runtime tool list"):
        validate_action(
            {"name": "search_flights", "arguments": {"origin": "PHX"}},
            allowed_tools={"cancel_reservation"},
        )


def test_parse_action_rejects_top_level_array():
    with pytest.raises(ValueError):
        parse_action('[{"name":"respond","arguments":{"content":"Hi"}}]')


def test_parse_action_accepts_valid_tool_call():
    action = parse_action(
        '{"name":"lookup_order","arguments":{"order_id":"12345"}}',
        allowed_tools={"lookup_order"},
    )
    assert action["name"] == "lookup_order"
    assert action["arguments"]["order_id"] == "12345"


def test_parse_action_accepts_runtime_tool_name_from_tau2():
    action = parse_action(
        '{"name":"yara_garcia_123","arguments":{"reservation_id":"HXDUBJ"}}',
        allowed_tools={"yara_garcia_123"},
    )
    assert action == {
        "name": "yara_garcia_123",
        "arguments": {"reservation_id": "HXDUBJ"},
    }


def test_summarize_message_compacts_whitespace():
    summary = summarize_message("Hello   there\n\nthis is   a   test")
    assert summary == "Hello there this is a test"


def test_build_runtime_messages_keep_first_prompt_and_recent_turns():
    agent = Agent()
    first_prompt = """
    Policy: follow the tool contract.
    Available tools:
    {"name":"get_reservation_details","arguments":{"reservation_id":"string"}}
    """
    agent._store_user_turn(first_prompt)
    agent.turn_history.append(
        {
            "role": "assistant",
            "content": '{"name":"get_reservation_details","arguments":{"reservation_id":"EHGLP3"}}',
        }
    )
    agent._store_user_turn("Tool result says the reservation is eligible.")

    runtime_messages = agent._build_runtime_messages()

    assert runtime_messages[0]["role"] == "system"
    assert any(msg["role"] == "user" and msg["content"] == "Tool result says the reservation is eligible." for msg in runtime_messages)
    assert runtime_messages[-1]["content"] == "Tool result says the reservation is eligible."


def test_store_user_turn_does_not_overwrite_first_turn_allowed_tools():
    agent = Agent()
    first_prompt = """
    Here's a list of tools you can use (you can use at most one tool at a time):
    [{"type":"function","function":{"name":"first_tool"}}]
    """
    agent._store_user_turn(first_prompt)
    assert agent.allowed_tools == {"first_tool"}

    agent._store_user_turn("second turn without full tool contract")
    assert agent.allowed_tools == {"first_tool"}


def test_store_user_turn_accumulates_new_tools_from_later_turns():
    agent = Agent()
    agent._store_user_turn(
        """
        Here's a list of tools you can use (you can use at most one tool at a time):
        [
          {"type":"function","function":{"name":"first_tool","description":"Lookup","parameters":{"type":"object","properties":{"user_id":{"type":"string"}},"required":["user_id"]}}}
        ]
        """
    )

    agent._store_user_turn(
        """
        Here's a list of tools you can use (you can use at most one tool at a time):
        [
          {"type":"function","function":{"name":"second_tool","description":"Cancel","parameters":{"type":"object","properties":{"reservation_id":{"type":"string"}},"required":["reservation_id"]}}}
        ]
        """
    )

    assert agent.allowed_tools == {"first_tool", "second_tool"}


def test_extract_structured_facts_reads_status_completion_and_ids():
    facts = extract_structured_facts(
        "Tool result: reservation HXDUBJ for user noah_muller_9847 is confirmed and cancellation completed successfully."
    )
    assert facts["reservation_ids"] == ["HXDUBJ"]
    assert facts["user_ids"] == ["noah_muller_9847"]
    assert facts["status"] == "confirmed"
    assert facts["completion"] == "cancelled"


def test_store_user_turn_tracks_goal_history_across_user_requests():
    agent = Agent()
    agent.current_snapshot = "Please check my booking."
    agent._record_action({"name": "respond", "arguments": {"content": "Please share your booking details."}})
    agent.current_snapshot = "Now cancel it."
    agent._record_action({"name": "cancel_booking", "arguments": {"reservation_id": "HXDUBJ"}})
    agent.current_snapshot = "Actually change the date."
    agent._record_action({"name": "change_booking", "arguments": {"reservation_id": "HXDUBJ"}})

    assert [goal["intent"] for goal in agent.goal_history] == ["cancel", "change"]
    assert agent.active_goal == agent.goal_history[-1]


def test_store_user_turn_builds_pending_tasks_for_multi_intent_request():
    agent = Agent()
    agent._store_user_turn(
        "I’d like to cancel the reservations IFOYYZ and NQNU5R, and for the reservation M20IZO, I’d like to change it to a nonstop flight if that’s available."
    )

    assert [
        (task["intent"], task.get("reservation_id"))
        for task in agent.pending_tasks
    ] == [
        ("cancel", "IFOYYZ"),
        ("cancel", "NQNU5R"),
        ("change", "M20IZO"),
    ]
    assert agent.active_task is not None
    assert agent.active_task["reservation_id"] == "IFOYYZ"


def test_post_tool_completion_advances_to_next_pending_task():
    agent = Agent()
    agent._store_user_turn(
        "Please cancel reservations IFOYYZ and NQNU5R."
    )
    agent.tool_specs["cancel_reservation"] = {
        "name": "cancel_reservation",
        "description": "Cancel a reservation",
        "parameters": {
            "type": "object",
            "properties": {"reservation_id": {"type": "string"}},
            "required": ["reservation_id"],
        },
    }

    agent._record_action(
        {"name": "cancel_reservation", "arguments": {"reservation_id": "IFOYYZ"}}
    )
    agent.current_input_text = '{"reservation_id":"IFOYYZ","status":"cancelled"}'
    agent._store_user_turn(agent.current_input_text)

    action = agent._choose_deterministic_action()

    assert action == {
        "name": "cancel_reservation",
        "arguments": {"reservation_id": "NQNU5R"},
    }
    assert agent.active_task is not None
    assert agent.active_task["reservation_id"] == "NQNU5R"


def test_tool_result_is_bound_to_active_task():
    agent = Agent()
    agent._store_user_turn("Please cancel reservation IFOYYZ.")
    agent._record_action(
        {"name": "get_reservation_details", "arguments": {"reservation_id": "IFOYYZ"}}
    )

    result_text = '{"reservation_id":"IFOYYZ","user_id":"aarav_ahmed_6699","status":"confirmed"}'
    agent.current_input_text = result_text
    agent._store_user_turn(result_text)

    assert agent.active_task is not None
    assert agent.active_task["reservation_id"] == "IFOYYZ"
    assert "confirmed" in (agent.active_task.get("last_result_summary") or "")
    assert agent.tool_journal[-1]["task_summary"] == agent.active_task["summary"]


def test_post_tool_empty_result_uses_honest_past_tense_response():
    agent = Agent()
    agent._store_user_turn("Please search for a direct option for reservation M20IZO.")
    agent._record_action(
        {
            "name": "search_direct_flight",
            "arguments": {"origin": "BOS", "destination": "MSP", "date": "2024-05-18"},
        }
    )
    agent.current_input_text = "[]"
    agent._store_user_turn("[]")

    action = agent._choose_deterministic_action()

    assert action["name"] == "respond"
    assert "came back empty" in action["arguments"]["content"]
    assert "I'll search" not in action["arguments"]["content"]


def test_build_fallback_action_uses_reservation_id_when_present():
    action = build_fallback_action("Please cancel reservation EHGLP3 right away.")
    assert action["name"] == "respond"
    assert "couldn't complete that request" in action["arguments"]["content"]


def test_build_fallback_action_requests_identifier_when_none_present():
    action = build_fallback_action("I need help with my booking.")
    assert action["name"] == "respond"
    assert "couldn't complete that request" in action["arguments"]["content"]


def test_build_fallback_action_does_not_reask_ids_when_context_is_sufficient():
    action = build_fallback_action(
        "Please cancel this reservation.",
        context_text=(
            "user_id: marco_polo_123\n"
            "reservation_id: HXDUBJ\n"
            "Please cancel reservation HXDUBJ for user marco_polo_123."
        ),
    )
    assert action == {
        "name": "respond",
        "arguments": {"content": "I'm sorry, I couldn't complete that request right now."},
    }


def test_extract_allowed_tools_reads_exact_tool_block():
    input_text = """
    System policy here.

    Available tools:
    {"name":"get_reservation_details","arguments":{"reservation_id":"string"}}
    {"name":"cancel_reservation","arguments":{"reservation_id":"string"}}

    Conversation:
    User: Please help me.
    """
    assert extract_allowed_tools(input_text) == {
        "get_reservation_details",
        "cancel_reservation",
    }


def test_extract_allowed_tools_from_tau2_prompt_reads_function_names():
    tau2_prompt = """
    Policy...
    Here's a list of tools you can use (you can use at most one tool at a time):
    [
      {"type":"function","function":{"name":"yara_garcia_123","description":"x"}}
    ]
    Return JSON.
    """
    assert extract_allowed_tools_from_tau2_prompt(tau2_prompt) == {"yara_garcia_123"}


def test_extract_allowed_tools_from_tau2_prompt_handles_multiline_json():
    tau2_prompt = """
    Here's a list of tools you can use (you can use at most one tool at a time):
    [
      {
        "type": "function",
        "function": {
          "name": "tool_one",
          "description": "foo",
          "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
        }
      },
      {
        "type": "function",
        "function": {
          "name": "tool_two",
          "description": "bar",
          "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}
        }
      }
    ]
    """
    assert extract_allowed_tools_from_tau2_prompt(tau2_prompt) == {"tool_one", "tool_two"}


def test_build_runtime_messages_includes_runtime_tool_contract_after_first_turn():
    agent = Agent()
    first_green_message = """
    Here's a list of tools you can use (you can use at most one tool at a time):
    [
      {
        "type": "function",
        "function": {
          "name": "get_user_details",
          "description": "Lookup a user profile",
          "parameters": {
            "type": "object",
            "properties": {
              "user_id": {"type": "string"}
            },
            "required": ["user_id"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "cancel_reservation",
          "description": "Cancel a reservation",
          "parameters": {
            "type": "object",
            "properties": {
              "reservation_id": {"type": "string"}
            },
            "required": ["reservation_id"]
          }
        }
      }
    ]
    user: Please help.
    """
    agent._store_user_turn(first_green_message)
    agent._store_user_turn("user: my user ID is raj_sanchez_7340")

    runtime_messages = agent._build_runtime_messages()
    tool_contract = "\n".join(
        message["content"]
        for message in runtime_messages
        if message["role"] == "system" and "Runtime tools available right now:" in message["content"]
    )

    assert "get_user_details" in tool_contract
    assert "cancel_reservation" in tool_contract
    assert "user_id:string" in tool_contract
    assert "reservation_id:string" in tool_contract


def test_format_runtime_tool_contract_handles_empty_specs():
    assert format_runtime_tool_contract({}) == "Runtime tools: none"


def test_extract_allowed_tools_from_green_agent_first_message_single_request():
    green_message = """
    domain_policy:
    - follow policy
    Here's a list of tools you can use (you can use at most one tool at a time):
    [
      {
        "type": "function",
        "function": {
          "name": "yara_garcia_123",
          "description": "Lookup reservation",
          "parameters": {
            "type": "object",
            "properties": {
              "reservation_id": {"type": "string"}
            },
            "required": ["reservation_id"]
          }
        }
      }
    ]

    Synthetic action:
    {"name":"respond","arguments":{"content":"..."}}
    Return strict JSON with fields "name" and "arguments".

    user: Please check reservation HXDUBJ
    """
    assert extract_allowed_tools_from_tau2_prompt(green_message) == {"yara_garcia_123"}


def test_extract_allowed_tools_from_green_agent_message_multihop_keeps_first_turn_tools(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = Agent()
    first_green_message = """
    domain_policy:
    - follow policy
    Here's a list of tools you can use (you can use at most one tool at a time):
    [
      {
        "type": "function",
        "function": {
          "name": "yara_garcia_123",
          "description": "Lookup reservation",
          "parameters": {
            "type": "object",
            "properties": {
              "reservation_id": {"type": "string"}
            },
            "required": ["reservation_id"]
          }
        }
      }
    ]
    Return strict JSON with fields "name" and "arguments".
    user: Please check reservation HXDUBJ
    """
    agent._store_user_turn(first_green_message)
    assert agent.allowed_tools == {"yara_garcia_123"}

    agent._store_user_turn("User: still waiting, please continue with HXDUBJ.")
    agent.current_input_text = "User: please run the same check for HXDUBJ."

    def fake_call_model(_: list[dict[str, str]]) -> str:
        return '{"name":"yara_garcia_123","arguments":{"reservation_id":"HXDUBJ"}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()
    assert action == {
        "name": "yara_garcia_123",
        "arguments": {"reservation_id": "HXDUBJ"},
    }


def test_generate_action_returns_generic_fallback_when_allowed_tools_empty_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = Agent()
    agent.initial_prompt = "first turn without parseable tools"
    agent.allowed_tools = set()
    agent.current_input_text = "Please check reservation HXDUBJ."

    def fake_call_model(_: list[dict[str, str]]) -> str:
        return '{"name":"made_up_tool","arguments":{"reservation_id":"HXDUBJ"}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()

    assert action == {
        "name": "respond",
        "arguments": {"content": "I'm sorry, I couldn't complete that request right now."},
    }


def test_generate_action_skips_tool_validation_when_empty_toolset_debug_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AGENT_ALLOW_EMPTY_TOOLSET_DEBUG", "1")
    agent = Agent()
    agent.initial_prompt = "first turn without parseable tools"
    agent.allowed_tools = set()
    agent.current_input_text = "Please check reservation HXDUBJ."

    def fake_call_model(_: list[dict[str, str]]) -> str:
        return '{"name":"made_up_tool","arguments":{"reservation_id":"HXDUBJ"}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()

    assert action == {
        "name": "made_up_tool",
        "arguments": {"reservation_id": "HXDUBJ"},
    }


def test_policy_allows_handoff_detects_explicit_transfer_language():
    prompt = "Policy: transfer to a human agent when the user requests legal escalation."
    assert policy_allows_handoff(prompt) is True


def test_is_disallowed_respond_content_rejects_fake_incapability_with_tools():
    assert is_disallowed_respond_content(
        "I can't access your booking right now.",
        initial_prompt="Available tools are listed below.",
        allowed_tools={"get_reservation_details"},
        current_input_text="Please check reservation EHGLP3.",
    )


def test_is_disallowed_respond_content_rejects_broad_clarify_with_identifier():
    assert not is_disallowed_respond_content(
        "Could you please clarify your request?",
        initial_prompt="Available tools are listed below.",
        allowed_tools={"cancel_reservation"},
        current_input_text="Please cancel reservation EHGLP3.",
    )


def test_is_disallowed_respond_content_allows_handoff_when_policy_explicit():
    assert not is_disallowed_respond_content(
        "I will transfer you to a human agent for this request.",
        initial_prompt="Policy: transfer to a human agent for harassment complaints.",
        allowed_tools={"get_reservation_details"},
        current_input_text="I want to report harassment.",
    )


def test_call_model_rejects_empty_choices():
    agent = Agent()
    agent.model = "real"
    agent.api_key = "test-key"

    class DummyClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**_: Any):
                    class Response:
                        choices: list[Any] = []

                    return Response()

            completions = Completions()

        chat = Chat()

    agent.client = DummyClient()

    with pytest.raises(ValueError, match="no choices"):
        agent._call_model([{"role": "system", "content": "test"}])


def test_call_model_rejects_none_content():
    agent = Agent()
    agent.model = "real"
    agent.api_key = "test-key"

    class DummyMessage:
        content = None

    class DummyChoice:
        message = DummyMessage()

    class DummyClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**_: Any):
                    class Response:
                        choices = [DummyChoice()]

                    return Response()

            completions = Completions()

        chat = Chat()

    agent.client = DummyClient()

    with pytest.raises(ValueError, match="empty or malformed content"):
        agent._call_model([{"role": "system", "content": "test"}])


def test_call_model_retries_on_rate_limit():
    agent = Agent()
    agent.model = "real"
    agent.api_key = "test-key"
    attempts = {"count": 0}

    class DummyMessage:
        content = '{"name":"respond","arguments":{"content":"ok"}}'

    class DummyChoice:
        message = DummyMessage()

    class DummyClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**_: Any):
                    attempts["count"] += 1
                    if attempts["count"] < 3:
                        raise RateLimitError(
                            "rate limited",
                            response=httpx.Response(429, request=httpx.Request("POST", "https://example.com")),
                            body=None,
                        )

                    class Response:
                        choices = [DummyChoice()]

                    return Response()

            completions = Completions()

        chat = Chat()

    agent.client = DummyClient()

    output = agent._call_model([{"role": "system", "content": "test"}])
    assert attempts["count"] == 3
    assert output == '{"name":"respond","arguments":{"content":"ok"}}'


def test_generate_action_returns_safe_respond_for_invalid_json(monkeypatch: pytest.MonkeyPatch):
    agent = Agent()
    prompt = """
    Available tools:
    {"name":"cancel_reservation","arguments":{"reservation_id":"string"}}
    """
    agent._store_user_turn(prompt)
    agent.current_input_text = "Please cancel reservation EHGLP3."

    call_count = {"count": 0}

    def fake_call_model(messages: list[dict[str, str]]) -> str:
        call_count["count"] += 1
        assert any(message["content"] == prompt for message in messages)
        return "not json"

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()

    assert call_count["count"] >= 1
    assert action["name"] == "respond"
    assert "couldn't complete that request" in action["arguments"]["content"]


def test_generate_action_returns_safe_respond_for_invalid_tool(monkeypatch: pytest.MonkeyPatch):
    agent = Agent()
    agent._store_user_turn(
        """
        Available tools:
        {"name":"cancel_reservation","arguments":{"reservation_id":"string"}}
        """
    )
    agent.current_input_text = "Please cancel reservation EHGLP3."

    call_count = {"count": 0}

    def fake_call_model(_: list[dict[str, str]]) -> str:
        call_count["count"] += 1
        return '{"name":"search_flights","arguments":{"origin":"PHX"}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()

    assert call_count["count"] >= 1
    assert action["name"] == "respond"
    assert "couldn't complete that request" in action["arguments"]["content"]


def test_generate_action_uses_single_model_call_on_happy_path(monkeypatch: pytest.MonkeyPatch):
    agent = Agent()
    prompt = """
    Available tools:
    {"name":"cancel_reservation","arguments":{"reservation_id":"string"}}
    """
    agent._store_user_turn(prompt)
    agent.current_input_text = "Please cancel reservation EHGLP3."

    call_count = {"count": 0}

    def fake_call_model(messages: list[dict[str, str]]) -> str:
        call_count["count"] += 1
        assert any(message["content"] == prompt for message in messages)
        return '{"name":"cancel_reservation","arguments":{"reservation_id":"EHGLP3"}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()

    assert call_count["count"] == 1
    assert action == {
        "name": "cancel_reservation",
        "arguments": {"reservation_id": "EHGLP3"},
    }


def test_generate_action_prefers_lookup_with_user_id_before_asking(monkeypatch: pytest.MonkeyPatch):
    agent = Agent()
    prompt = """
    Here's a list of tools you can use (you can use at most one tool at a time):
    [
      {
        "type": "function",
        "function": {
          "name": "find_booking",
          "description": "Find a reservation by user ID",
          "parameters": {
            "type": "object",
            "properties": {
              "user_id": {"type": "string"}
            },
            "required": ["user_id"]
          }
        }
      }
    ]
    user: My user ID is noah_muller_9847 and I want to cancel my booking.
    """
    agent._store_user_turn(prompt)
    agent.current_input_text = prompt

    call_count = {"count": 0}
    def fake_call_model(_: list[dict[str, str]]) -> str:
        call_count["count"] += 1
        return '{"name":"find_booking","arguments":{"user_id":"noah_muller_9847"}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()
    assert call_count["count"] == 1
    assert action == {
        "name": "find_booking",
        "arguments": {"user_id": "noah_muller_9847"},
    }


def test_generate_action_asks_operator_goal_before_transferring(monkeypatch: pytest.MonkeyPatch):
    agent = Agent()
    prompt = "I want to speak to a human agent."
    agent._store_user_turn(prompt)
    agent.current_input_text = prompt

    call_count = {"count": 0}
    def fake_call_model(_: list[dict[str, str]]) -> str:
        call_count["count"] += 1
        return '{"name":"respond","arguments":{"content":"I can help with that. What do you need assistance with today?"}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()
    assert call_count["count"] == 1
    assert action == {
        "name": "respond",
        "arguments": {"content": "I can help with that. What do you need assistance with today?"},
    }


def test_generate_action_repeated_operator_request_without_policy_still_asks_goal(monkeypatch: pytest.MonkeyPatch):
    agent = Agent()
    first = "I want to speak to a human agent."
    second = "No, connect me to a human agent now."
    agent._store_user_turn(first)
    agent.current_input_text = first
    monkeypatch.setattr(
        agent,
        "_call_model",
        lambda _: '{"name":"respond","arguments":{"content":"I can help with that. What do you need assistance with today?"}}',
    )
    agent._generate_action()
    agent._store_user_turn(second)
    agent.current_input_text = second

    call_count = {"count": 0}
    def fake_call_model(_: list[dict[str, str]]) -> str:
        call_count["count"] += 1
        return '{"name":"respond","arguments":{"content":"I can help with that. What do you need assistance with today?"}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()
    assert call_count["count"] == 1
    assert action == {
        "name": "respond",
        "arguments": {"content": "I can help with that. What do you need assistance with today?"},
    }


def test_generate_action_chains_lookup_result_into_cancel_tool(monkeypatch: pytest.MonkeyPatch):
    agent = Agent()
    first_prompt = """
    Here's a list of tools you can use (you can use at most one tool at a time):
    [
      {
        "type": "function",
        "function": {
          "name": "find_booking",
          "description": "Find reservation by user ID",
          "parameters": {
            "type": "object",
            "properties": {
              "user_id": {"type": "string"}
            },
            "required": ["user_id"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "cancel_booking",
          "description": "Cancel reservation by reservation ID",
          "parameters": {
            "type": "object",
            "properties": {
              "reservation_id": {"type": "string"}
            },
            "required": ["reservation_id"]
          }
        }
      }
    ]
    user: My user ID is noah_muller_9847 and I want to cancel my booking.
    """
    agent._store_user_turn(first_prompt)
    agent.current_input_text = first_prompt

    call_count = {"count": 0}
    def fake_call_model(_: list[dict[str, str]]) -> str:
        call_count["count"] += 1
        return '{"name":"find_booking","arguments":{"user_id":"noah_muller_9847"}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    first_action = agent._generate_action()
    assert call_count["count"] == 1
    assert first_action == {
        "name": "find_booking",
        "arguments": {"user_id": "noah_muller_9847"},
    }
    agent._record_action(first_action)

    tool_result = "Tool result: reservation_id is HXDUBJ. The reservation is active."
    agent._store_user_turn(tool_result)
    agent.current_input_text = tool_result

    second_action = agent._generate_action()
    assert second_action == {
        "name": "cancel_booking",
        "arguments": {"reservation_id": "HXDUBJ"},
    }


def test_generate_action_responds_after_check_tool_result_when_no_next_tool(monkeypatch: pytest.MonkeyPatch):
    agent = Agent()
    agent.current_intent = "check"
    agent.current_input_text = "Tool result: reservation HXDUBJ is confirmed for tomorrow."

    def should_not_run(_: list[dict[str, str]]) -> str:
        raise AssertionError("model should not be called for deterministic post-tool response")

    monkeypatch.setattr(agent, "_call_model", should_not_run)

    action = agent._generate_action()
    assert action == {
        "name": "respond",
        "arguments": {
            "content": "I've reviewed the reservation details from the latest tool result. Let me know if you want to change or cancel it."
        },
    }


def test_generate_action_uses_structured_status_in_post_tool_response(monkeypatch: pytest.MonkeyPatch):
    agent = Agent()
    agent.current_intent = "check"
    agent.awaiting_tool_result = True
    agent.last_action_name = "find_booking"
    tool_result = "Tool result: reservation HXDUBJ is confirmed."
    agent._store_user_turn(tool_result)
    agent.current_input_text = tool_result

    def should_not_run(_: list[dict[str, str]]) -> str:
        raise AssertionError("model should not be called for structured post-tool response")

    monkeypatch.setattr(agent, "_call_model", should_not_run)

    action = agent._generate_action()
    assert action == {
        "name": "respond",
        "arguments": {
            "content": "I've reviewed the reservation and its current status is confirmed. Let me know if you want to change or cancel it."
        },
    }


def test_generate_action_uses_transfer_tool_when_policy_allows(monkeypatch: pytest.MonkeyPatch):
    agent = Agent()
    prompt = """
    Policy: transfer to a human agent for harassment complaints.
    Here's a list of tools you can use (you can use at most one tool at a time):
    [
      {
        "type": "function",
        "function": {
          "name": "transfer_to_human_agents",
          "description": "Transfer the conversation to a human agent",
          "parameters": {
            "type": "object",
            "properties": {}
          }
        }
      }
    ]
    user: I need a human agent for this harassment complaint.
    """
    agent._store_user_turn(prompt)
    agent.current_input_text = prompt

    call_count = {"count": 0}
    def fake_call_model(_: list[dict[str, str]]) -> str:
        call_count["count"] += 1
        return '{"name":"respond","arguments":{"content":"I will transfer you to a human agent for this request."}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()
    assert call_count["count"] == 1
    assert action == {
        "name": "transfer_to_human_agents",
        "arguments": {},
    }


def test_generate_action_responds_with_exact_transfer_message_after_transfer_tool_result(monkeypatch: pytest.MonkeyPatch):
    agent = Agent()
    prompt = """
    Policy: transfer to a human agent for harassment complaints.
    Here's a list of tools you can use (you can use at most one tool at a time):
    [
      {
        "type": "function",
        "function": {
          "name": "transfer_to_human_agents",
          "description": "Transfer the conversation to a human agent",
          "parameters": {
            "type": "object",
            "properties": {}
          }
        }
      }
    ]
    user: I need a human agent for this harassment complaint.
    """
    agent._store_user_turn(prompt)
    first_action = {"name": "transfer_to_human_agents", "arguments": {}}
    agent._record_action(first_action)
    tool_result = "Tool 'transfer_to_human_agents' result: success"
    agent._store_user_turn(tool_result)
    agent.current_input_text = tool_result

    def should_not_run(_: list[dict[str, str]]) -> str:
        raise AssertionError("model should not be called after transfer tool result")

    monkeypatch.setattr(agent, "_call_model", should_not_run)

    action = agent._generate_action()
    assert action == {
        "name": "respond",
        "arguments": {"content": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."},
    }


def test_generate_action_rejects_handoff_respond_when_policy_does_not_allow_it(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = Agent()
    prompt = """
    Available tools:
    {"name":"get_reservation_details","arguments":{"reservation_id":"string"}}
    """
    agent._store_user_turn(prompt)
    agent.current_input_text = "Please check reservation EHGLP3."

    call_count = {"count": 0}

    def fake_call_model(_: list[dict[str, str]]) -> str:
        call_count["count"] += 1
        return '{"name":"respond","arguments":{"content":"You are being transferred to a human agent."}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()

    assert call_count["count"] == 1
    assert action["name"] == "respond"
    assert "couldn't complete that request" in action["arguments"]["content"]


def test_generate_action_allows_handoff_when_policy_explicitly_requires_it(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = Agent()
    prompt = """
    Policy: transfer to a human agent for harassment complaints.
    Available tools:
    {"name":"get_reservation_details","arguments":{"reservation_id":"string"}}
    """
    agent._store_user_turn(prompt)
    agent.current_input_text = "I want to report harassment."

    def fake_call_model(_: list[dict[str, str]]) -> str:
        return '{"name":"respond","arguments":{"content":"I will transfer you to a human agent for this request."}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()

    assert action == {
        "name": "respond",
        "arguments": {"content": "I will transfer you to a human agent for this request."},
    }


def test_generate_action_rejects_fake_incapability_claim_when_tools_exist(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = Agent()
    prompt = """
    Available tools:
    {"name":"get_reservation_details","arguments":{"reservation_id":"string"}}
    """
    agent._store_user_turn(prompt)
    agent.current_input_text = "Please check reservation EHGLP3."

    def fake_call_model(_: list[dict[str, str]]) -> str:
        return '{"name":"respond","arguments":{"content":"I cannot access your booking right now."}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()

    assert action == {
        "name": "respond",
        "arguments": {"content": "I'm sorry, I couldn't complete that request right now."},
    }


def test_generate_action_rejects_broad_clarify_when_identifier_already_present(
    monkeypatch: pytest.MonkeyPatch,
):
    agent = Agent()
    prompt = """
    Available tools:
    {"name":"cancel_reservation","arguments":{"reservation_id":"string"}}
    """
    agent._store_user_turn(prompt)
    agent.current_input_text = "Please cancel reservation EHGLP3."

    def fake_call_model(_: list[dict[str, str]]) -> str:
        return '{"name":"respond","arguments":{"content":"Could you please clarify your request?"}}'

    monkeypatch.setattr(agent, "_call_model", fake_call_model)

    action = agent._generate_action()

    assert action == {
        "name": "respond",
        "arguments": {"content": "Could you please clarify your request?"},
    }


def test_resolve_provider_config_for_openai_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENT_PROVIDER", "auto")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    config = resolve_provider_config("openai/gpt-4o-mini")

    assert config["provider"] == "openai"
    assert config["model"] == "gpt-4o-mini"
    assert config["api_key"] == "openai-key"
    assert config["base_url"] == "https://api.openai.com/v1"


def test_resolve_provider_config_for_gemini_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENT_PROVIDER", "auto")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    config = resolve_provider_config("gemini/gemini-2.5-flash")

    assert config["provider"] == "gemini"
    assert config["model"] == "gemini-2.5-flash"
    assert config["api_key"] == "gemini-key"
    assert "generativelanguage.googleapis.com" in config["base_url"]


def test_resolve_provider_config_for_deepseek_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENT_PROVIDER", "auto")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    config = resolve_provider_config("deepseek/deepseek-chat")

    assert config["provider"] == "deepseek"
    assert config["model"] == "deepseek-chat"
    assert config["api_key"] == "deepseek-key"
    assert config["base_url"] == "https://api.deepseek.com"


def test_resolve_provider_config_for_openrouter_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENT_PROVIDER", "openrouter")
    monkeypatch.setenv("AGENT_LLM", "qwen/qwen3.6-plus:free")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    config = resolve_provider_config("qwen/qwen3.6-plus:free")

    assert config["provider"] == "openrouter"
    assert config["model"] == "qwen/qwen3.6-plus:free"
    assert config["api_key"] == "openrouter-key"
    assert config["base_url"] == "https://openrouter.ai/api/v1"


def test_resolve_provider_config_auto_detects_openrouter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENT_PROVIDER", "auto")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "compat-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    config = resolve_provider_config("qwen/qwen3.6-plus:free")

    assert config["provider"] == "openrouter"
    assert config["model"] == "qwen/qwen3.6-plus:free"
    assert config["api_key"] == "compat-key"


async def send_text_message(
    text: str,
    url: str,
    context_id: str | None = None,
    streaming: bool = False,
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

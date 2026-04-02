# Tau2 Purple Agent Spec

## Purpose

This repository contains a tau2-compatible purple agent for AgentBeats.

The agent acts as a customer-service agent evaluated by the tau2 green agent across multi-turn tasks in domains such as airline, retail, telecom, and mock.

The primary goal is to build an agent that is:

- protocol-compatible with AgentBeats and A2A
- robust on OpenAI-like model endpoints
- easy to iterate on in an open-source repository
- reproducible, debuggable, and competition-ready

## External Contract

The tau2 green agent communicates with this agent through A2A text messages.

This agent must:

- accept a text A2A message
- read the full task context from that message
- return exactly one JSON action per turn

Expected output formats:

```json
{"name":"<tool_name>","arguments":{}}
```

or

```json
{"name":"respond","arguments":{"content":"..."}}
```

The agent must not:

- emit markdown
- emit explanations around the JSON
- emit multiple actions in one turn
- emit chain-of-thought

## What The Input Contains

The incoming user message from the green agent may include:

- domain policy
- available tool schemas
- role and task instructions
- conversation history
- tool results from previous steps

The agent should treat the incoming message as the complete current context for choosing the next action.

## Core Behaviors

The agent must:

- follow the provided policy exactly
- use tools when facts need verification
- avoid inventing booking, order, account, or troubleshooting details
- ask for missing information when required to complete a task
- choose only one action per turn
- respond concisely and professionally when speaking to the user

The agent should prefer:

- grounded tool use over guessing
- safe clarification over irreversible action
- short customer-facing responses over verbose explanations

## Output Rules

Every final model output must be a valid JSON object with:

- `name`: string
- `arguments`: object

Special case:

- if `name == "respond"`, then `arguments.content` must be a non-empty string

If the model returns invalid JSON:

1. run one repair pass
2. validate again
3. if still invalid, return a safe fallback response action

Recommended fallback:

```json
{"name":"respond","arguments":{"content":"I'm sorry, could you clarify your request?"}}
```

## State Management

The agent may keep conversation state within a single A2A task or context.

The agent must reset state between assessments.

The agent must not carry over:

- previous user conversations
- previous tool results
- previous domain-specific assumptions

## Recommended Architecture

## Runtime

- Python A2A server built from `agent-template`
- model access via an OpenAI-like endpoint or LiteLLM-compatible routing layer
- environment-driven configuration

## Suggested Files

- `docs/tau2-agent-spec.md`: human-facing development spec
- `src/prompts/system.txt`: model-facing instruction
- `src/agent.py`: runtime logic
- `src/server.py`: A2A server and agent card
- `tests/`: conformance and regression tests

## Agent Loop

Recommended per-turn flow:

1. receive A2A text input
2. add input to conversation state
3. call model once for next action
4. parse and validate JSON
5. if invalid, run one repair call
6. emit validated action as a text artifact

## Prompting Guidelines

The model-facing prompt should be:

- short
- strict
- operational
- focused on output discipline and policy/tool behavior

The prompt should not contain:

- long product documentation
- benchmark lore
- unnecessary examples
- chain-of-thought instructions

## OpenAI-Like Endpoint Best Practices

The implementation should assume OpenAI-compatible or LiteLLM-style chat endpoints.

Recommended practices:

- keep prompts in versioned files, not only inline constants
- validate every output locally, even when using JSON mode
- log raw model output in debug mode
- isolate model configuration in environment variables
- support easy switching between providers through `AGENT_LLM`

Recommended environment variables:

- `AGENT_LLM`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
- `LOG_LEVEL`

## Evaluation Priorities

The agent should be optimized for:

- policy compliance
- correct tool selection
- successful task completion
- robustness across domains
- cost efficiency
- reproducibility

## Failure Modes To Watch

Common failure modes:

- invalid JSON
- selecting a nonexistent tool
- wrong tool arguments
- answering from memory instead of verifying with tools
- making irreversible claims before tool confirmation
- verbose customer responses that waste turns
- confusion after tool results in multi-turn dialogue

## Development Plan

## Phase 1: MVP

- implement strict JSON output
- support `respond` and tool calls
- add one repair step
- verify local compatibility with tau2 green agent

Current status:

- strict JSON output: done
- support `respond` and tool calls: done
- add one repair step: done
- verify local compatibility with tau2 green agent: pending

## Phase 2: Stability

- add regression tests for malformed outputs
- improve logging
- harden validation and fallback behavior

Current status:

- regression tests for malformed outputs: partially done
- improve logging: partially done
- harden validation and fallback behavior: partially done

## Phase 3: Performance

- improve prompt quality
- improve tool-use heuristics
- reduce unnecessary turns
- benchmark across multiple domains

Current status:

- improve prompt quality: started
- improve tool-use heuristics: pending
- reduce unnecessary turns: pending
- benchmark across multiple domains: pending

## Competition Readiness

To align with AgentBeats and competition expectations, the repository should include:

- a clear README
- reproducible local setup
- Docker support
- environment-based configuration
- basic tests
- documented prompt and system design

Current status:

- clear README: done
- reproducible local setup: documented, not yet verified
- Docker support: done
- environment-based configuration: done
- basic tests: done
- documented prompt and system design: done

## Remaining Checklist

- run `uv sync`
- start the local A2A server successfully
- run `pytest` against the local agent
- run one local tau2 evaluation against the tau2 green agent
- inspect failures on at least one real task
- add stronger validation for suspicious tool names or malformed arguments
- tune prompt and memory handling based on real tau2 traces

## Non-Goals

This agent does not aim to:

- expose hidden reasoning
- act autonomously outside the provided tau2 context
- use multiple tools in a single turn
- optimize for human-style verbosity

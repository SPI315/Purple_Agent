# Purple Agent

This repository is the starting point for a tau2-compatible purple agent for AgentBeats.

## What This Agent Is

The agent is intended to participate in tau2 customer-service evaluations as a purple agent. It receives policy, tool schemas, and conversation state from the tau2 green agent and must return exactly one JSON action per turn.

## Repository Docs

- `docs/tau2-agent-spec.md` contains the development specification
- `src/prompts/system.txt` contains the model-facing system instruction

## Development Priorities

- strict JSON output
- correct tool selection
- policy compliance
- reproducible local development
- compatibility with OpenAI-like model endpoints

## Planned Runtime Configuration

Expected environment variables:

- `AGENT_LLM`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
- `LOG_LEVEL`

## Local Setup

1. Create a local environment and install dependencies:

```bash
uv sync
```

2. Create a local env file from the example:

```bash
cp .env.example .env
```

3. Fill in at least:

- `AGENT_LLM`
- one matching provider key such as `OPENAI_API_KEY`

4. Start the agent:

```bash
uv run src/server.py --host 127.0.0.1 --port 9009
```

5. In another terminal, run A2A conformance tests against the running server:

```bash
uv sync --extra test
uv run pytest --agent-url http://localhost:9009
```

## Docker

Build the image:

```bash
docker build -t purple-agent .
```

Run the image:

```bash
docker run -p 9009:9009 --env-file .env purple-agent --host 0.0.0.0 --port 9009
```

## Next Implementation Steps

1. Improve `src/agent.py` for tau2-specific tool selection and policy handling.
2. Add structured logging for invalid model outputs and repair attempts.
3. Test against the tau2 green agent on a small `airline` run.
4. Tune prompts and model settings for pass rate and cost.

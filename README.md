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

- `AGENT_PROVIDER`
- `AGENT_LLM`
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`
- `OPENAI_TIMEOUT`
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

- `AGENT_PROVIDER`
- `AGENT_LLM`
- one provider key:
  - `OPENAI_API_KEY`
  - `GEMINI_API_KEY`
  - `DEEPSEEK_API_KEY`
  - `OPENROUTER_API_KEY` or `OPENAI_API_KEY` for OpenRouter

Provider examples:

- `AGENT_PROVIDER=openai`, `AGENT_LLM=gpt-4o-mini`
- `AGENT_PROVIDER=gemini`, `AGENT_LLM=gemini-2.5-flash`
- `AGENT_PROVIDER=deepseek`, `AGENT_LLM=deepseek-chat`
- `AGENT_PROVIDER=openrouter`, `AGENT_LLM=qwen/qwen3.6-plus:free`, `OPENAI_BASE_URL=https://openrouter.ai/api/v1`

If `AGENT_PROVIDER=auto`, the agent infers the provider from the model prefix (`openai/...`, `gemini/...`, `deepseek/...`) or from an OpenRouter base URL/key.

4. Start the agent:

```bash
PYTHONPATH=src uv run python src/server.py --host 127.0.0.1 --port 9009
```

If you bind to `0.0.0.0`, pass `--card-url` so the advertised agent card URL stays reachable:

```bash
PYTHONPATH=src uv run python src/server.py --host 0.0.0.0 --port 9009 --card-url http://localhost:9009/
```

5. In another terminal, run A2A conformance tests against the running server:

```bash
uv sync --extra test
PYTHONPATH=src uv run pytest tests --agent-url http://localhost:9009
```

If you want a quick manual request against the running agent, use:

```bash
PYTHONPATH=src uv run python -c "import asyncio; from tests.test_agent import send_text_message; events = asyncio.run(send_text_message('Hello', 'http://localhost:9009', streaming=False)); print(events)"
```

For offline smoke tests without a real provider call, you can run the server with:

```bash
AGENT_LLM=mock PYTHONPATH=src uv run python src/server.py --host 127.0.0.1 --port 9009
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

The Docker image is locked by `uv.lock` for reproducible builds.

## Manifest

`amber-manifest.json5` now points to `purple-agent:latest` for local packaging. Before publishing to AgentBeats, replace it with your public registry reference, for example `ghcr.io/<your-github-username>/purple-agent:latest`.

## Next Implementation Steps

1. Improve `src/agent.py` for tau2-specific tool selection and policy handling.
2. Add stronger regression tests for numeric fidelity, policy edge cases, and long-context behavior.
3. Test against the tau2 green agent on a small `airline` run.
4. Tune prompts and model settings for pass rate and cost.

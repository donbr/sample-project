# PydanticAI Learning Hub: Durable Execution

This teaching repository demonstrates **Durable Agent Architecture** using PydanticAI and Temporal, based on the patterns found in `src/agents`.

It explains how to:
1.  **Wrap Agents**: Using `TemporalAgent` to make standard agents durable.
2.  **Orchestrate Workflows**: Managing `Planner -> Searcher -> Analyst` flows with `DeepResearchWorkflow`.
3.  **Handle Human-in-the-Loop**: Managing long-running state for interactive sessions.

## Project Structure
- `backend/app/lessons/lesson1_temporal_basics.py`: Minimal `TemporalAgent` example.
- `backend/app/lessons/lesson2_deep_research.py`: Parallel execution using `asyncio.TaskGroup` inside a Workflow.
- `backend/app/lessons/lesson3_human_loop.py`: Stateful interaction pattern.
- `backend/app/worker.py`: The Temporal Worker that executes the durable agents and workflows.

## Prerequisites

1.  **Temporal Server**: Ensure a local Temporal server is running.
    ```bash
    # From the project root
    docker-compose up -d
    ```
    (Or launch via `temporal server start-dev` if you have the CLI installed).

2.  **Install Dependencies**:
    ```bash
    cd pydantic-ai-learning
    uv sync
    ```

## Running the Application

You need to run **two** separate processes: the API server (Front-end/Client) and the Temporal Worker (Backend execution).

### 1. Start the API Server
This serves the Frontend and triggers workflows.
```bash
uv run uvicorn backend.app.main:app --reload
```

### 2. Start the Temporal Worker
This executes the Agent logic and Workflows.
```bash
uv run python -m backend.app.worker
```

## Validation & Testing

1.  Open your browser to: **http://localhost:8000/**
2.  Verify the status message says: `System: Real Temporal Client Connected.`
3.  **Lesson 1 (Durable Agent)**:
    *   Click "Start Workflow".
    *   Verify the chat history updates with a response from the agent.
4.  **Lesson 2 (Deep Research)**:
    *   Click the "Deep Research" tab.
    *   Enter a topic and click "Plan & Research".
    *   Verify you see multiple log updates as the Planner, Searcher, and Analyst agents execute in parallel.
5.  **Lesson 3 (Human-in-the-Loop)**:
    *   Click the "Human-in-the-Loop" tab.
    *   Start the game and interact with the workflow (simulated loop).

## Troubleshooting

- **Worker Heartbeat Warning**: You may see `WARN temporalio_sdk_core::worker::heartbeat`. This is a harmless warning when running against certain local development servers.
- **Connection Refused**: Ensure `docker-compose` is running and the Temporal Web UI is accessible at `http://localhost:8080`.

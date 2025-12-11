---
title: Overview - Pydantic AI
source: https://ai.pydantic.dev/durable_execution/overview/
author:
  - PydanticAI
published: 2025-12-07
created: 2025-12-07
description: GenAI Agent Framework, the Pydantic way
tags:
  - clippings
  - durable-exec
  - pydanticai
  - agents
---
[Skip to content](https://ai.pydantic.dev/durable_execution/overview/#durable-execution)

## Durable Execution

Pydantic AI allows you to build durable agents that can preserve their progress across transient API failures and application errors or restarts, and handle long-running, asynchronous, and human-in-the-loop workflows with production-grade reliability. Durable agents have full support for [streaming](https://ai.pydantic.dev/agents/#streaming-all-events) and [MCP](https://ai.pydantic.dev/mcp/client/), with the added benefit of fault tolerance.

Pydantic AI natively supports three durable execution solutions:

- [Temporal](https://ai.pydantic.dev/durable_execution/temporal/)
- [DBOS](https://ai.pydantic.dev/durable_execution/dbos/)
- [Prefect](https://ai.pydantic.dev/durable_execution/prefect/)

These integrations only use Pydantic AI's public interface, so they also serve as a reference for integrating with other durable systems.
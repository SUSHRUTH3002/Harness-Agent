"""LLM provider adapters. Optional: each adapter needs its own extra, e.g. `pip install agent-harness[litellm]`.

Adapters are imported explicitly (`from agent_harness.providers.litellm_provider import LiteLLMProvider`)
so that importing `agent_harness` never pulls in a provider SDK.
"""

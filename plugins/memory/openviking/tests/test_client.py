# -*- coding: utf-8 -*-
"""REST-contract and secret-safe error tests for the OpenViking client."""

import json

import httpx
import pytest

from plugins.memory.openviking.backend.client import (
    OpenVikingClient,
    OpenVikingClientConfig,
    OpenVikingConfigurationError,
    OpenVikingServiceError,
)


def _client(handler, *, api_key="test-tenant-key"):
    return OpenVikingClient(
        OpenVikingClientConfig("http://openviking.test", api_key),
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_client_uses_rest_contract_and_updates_existing_policy():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "auth_mode": "api_key",
                    "account_id": "a",
                    "user_id": "u",
                },
            )
        if request.method == "POST":
            return httpx.Response(409, json={})
        return httpx.Response(200, json={"result": {}})

    client = _client(handler)
    identity = await client.resolve_identity()
    await client.ensure_session("session-1", auto_commit_policy={"n": 2})
    await client.close()

    assert identity.namespace == "a/u"
    assert [(item.method, item.url.path) for item in seen] == [
        ("GET", "/health"),
        ("POST", "/api/v1/sessions"),
        ("PATCH", "/api/v1/sessions/session-1/config"),
    ]
    assert seen[1].headers["X-API-Key"] == "test-tenant-key"
    assert json.loads(seen[2].content) == {"auto_commit_policy": {"n": 2}}


@pytest.mark.asyncio
async def test_client_sends_bounded_context_search_request():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"result": {"digest": "memory"}})

    client = _client(handler)
    result = await client.search_context(
        query="project code",
        session_id="session-1",
        max_results=3,
        token_budget=2048,
    )
    await client.close()

    assert result == {"digest": "memory"}
    assert requests[0].url.path == "/api/v1/search/search"
    assert json.loads(requests[0].content) == {
        "query": "project code",
        "context_type": "memory",
        "session_id": "session-1",
        "limit": 3,
        "mode": "context",
        "query_expansion": "off",
        "max_tokens": 2048,
    }


@pytest.mark.asyncio
async def test_invalid_key_is_configuration_error():
    client = _client(
        lambda request: httpx.Response(
            200,
            json={"auth_mode": "api_key", "account_id": "", "user_id": ""},
        ),
    )

    with pytest.raises(OpenVikingConfigurationError, match="rejected"):
        await client.resolve_identity()
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503])
async def test_temporary_http_error_is_service_error(status):
    client = _client(lambda request: httpx.Response(status, text="temporary"))

    with pytest.raises(OpenVikingServiceError, match=f"HTTP {status}"):
        await client.search_memories(query="q", max_results=1)
    await client.close()


@pytest.mark.asyncio
async def test_server_response_never_echoes_tenant_key():
    key = "tenant-key-must-not-leak"
    client = _client(
        lambda request: httpx.Response(503, text="server saw " + key),
        api_key=key,
    )

    with pytest.raises(OpenVikingServiceError) as caught:
        await client.search_memories(query="q", max_results=1)
    await client.close()

    assert key not in str(caught.value)

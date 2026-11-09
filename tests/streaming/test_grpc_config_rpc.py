"""``GetRivaSpeechRecognitionConfig``: the static description."""

from __future__ import annotations

from support.grpc_harness import FakeServicerContext
from support.riva import config_rpc_request, rasr

from .conftest import SERVED, serving


async def _describe(served):
    result = await served.servicer.GetRivaSpeechRecognitionConfig(
        config_rpc_request(), FakeServicerContext()
    )
    assert isinstance(result, rasr.RivaSpeechRecognitionConfigResponse)
    return result


def _value_sets(entry) -> list[set[str]]:
    return [
        {part.strip() for part in value.split(",")}
        for value in entry.parameters.values()
    ]


def _assert_entry(entry, locales: set[str]) -> None:
    assert entry.model_name == SERVED
    sets = _value_sets(entry)
    assert {"8000", "16000"} in sets, dict(entry.parameters)
    assert {"LINEAR_PCM", "MULAW", "ALAW"} in sets, dict(entry.parameters)
    assert locales in sets, dict(entry.parameters)
    text = " ".join(entry.parameters.values()).lower()
    assert "punctuation" in text, dict(entry.parameters)
    assert "verbatim" in text, dict(entry.parameters)


# @spec ING-GRPC-014
# @spec ING-GRPC-006
async def test_streaming_and_offline_entries_while_files_are_enabled(served):
    assert served.plugin.files_enabled is True
    response = await _describe(served)
    types = sorted(entry.parameters["type"] for entry in response.model_config)
    assert types == ["offline", "streaming"]
    for entry in response.model_config:
        _assert_entry(entry, {"en-US", "de-DE", "auto"})


# @spec ING-GRPC-014
async def test_only_the_streaming_entry_while_files_are_disabled(clock):
    async with serving(clock=clock, tasks=("realtime",)) as served:
        assert served.plugin.files_enabled is False
        response = await _describe(served)
        assert [e.parameters["type"] for e in response.model_config] == [
            "streaming"
        ]
        _assert_entry(response.model_config[0], {"en-US", "de-DE", "auto"})


# @spec ING-GRPC-014
async def test_description_is_static(served):
    first = await _describe(served)
    second = await _describe(served)
    assert first == second
    assert "open" not in served.factory.names()

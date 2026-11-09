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


LOCALES = ["auto", "de-DE", "en-US"]


def _assert_entry(entry, locale: str) -> None:
    assert entry.model_name == SERVED
    params = dict(entry.parameters)
    assert params["language_code"] == locale, params
    assert "language_codes" not in params, params
    assert params["sample_rates"] == "8000,16000", params
    assert params["encodings"] == "LINEAR_PCM,MULAW,ALAW", params
    text = params["notes"].lower()
    assert "punctuation" in text, params
    assert "verbatim" in text, params


def _pairs(response) -> list[tuple[str, str]]:
    return [
        (e.parameters["type"], e.parameters["language_code"])
        for e in response.model_config
    ]


def _stock_list_models(response, kind: str) -> dict:
    """The stock scripts' ``--list-models`` grouping.

    As their public source describes it: keep the entries of one type and
    group them by locale.
    """
    models: dict = {}
    for config in response.model_config:
        if config.parameters["type"] == kind:
            code = config.parameters["language_code"]
            models.setdefault(code, []).append({"model": [config.model_name]})
    return dict(sorted(models.items()))


# @spec ING-GRPC-014
# @spec ING-GRPC-006
async def test_online_and_offline_entries_per_locale_while_files_are_enabled(
    served,
):
    assert served.plugin.files_enabled is True
    response = await _describe(served)
    assert _pairs(response) == [("online", c) for c in LOCALES] + [
        ("offline", c) for c in LOCALES
    ]
    for entry in response.model_config:
        _assert_entry(entry, entry.parameters["language_code"])


# @spec ING-GRPC-014
async def test_stock_list_models_lists_the_served_model_per_locale(served):
    response = await _describe(served)
    expected = {c: [{"model": [SERVED]}] for c in LOCALES}
    assert _stock_list_models(response, "online") == expected
    assert _stock_list_models(response, "offline") == expected


# @spec ING-GRPC-014
async def test_only_online_entries_while_files_are_disabled(clock):
    async with serving(clock=clock, tasks=("realtime",)) as served:
        assert served.plugin.files_enabled is False
        response = await _describe(served)
        assert _pairs(response) == [("online", c) for c in LOCALES]
        assert _stock_list_models(response, "offline") == {}
        for entry in response.model_config:
            _assert_entry(entry, entry.parameters["language_code"])


# @spec ING-GRPC-014
async def test_description_is_static(served):
    first = await _describe(served)
    second = await _describe(served)
    assert first == second
    assert "open" not in served.factory.names()

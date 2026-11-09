"""The request guard, the field allowlist, locale on the route, core pins."""

from __future__ import annotations

import asyncio
import io
import json

import pytest
from support.fake_transcription import Block, FakeTranscriptionServing
from support.http import assert_core_error, until

CORE_FIELD_NAMES_AND_ALIASES = frozenset(
    {
        "file",
        "model",
        "language",
        "hotwords",
        "prompt",
        "response_format",
        "timestamp_granularities",
        "timestamp_granularities[]",
        "stream",
        "stream_include_usage",
        "stream_continuous_usage_stats",
        "vllm_xargs",
        "to_language",
        "use_beam_search",
        "n",
        "length_penalty",
        "include_stop_str_in_output",
        "temperature",
        "watermarking",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "frequency_penalty",
        "repetition_penalty",
        "presence_penalty",
        "max_completion_tokens",
    }
)
GUARDED_SAMPLING = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "seed",
        "watermarking",
    }
)
IGNORED = frozenset({"n", "length_penalty", "include_stop_str_in_output"})


async def _post(env, fields):
    return await env.request(fields).finish()


# ------------------------------------------------------- refused features


# @spec ING-FILE-010
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_beam_search", "true"),
        ("max_completion_tokens", "100"),
        ("max_completion_tokens", "1"),
    ],
)
async def test_beam_search_and_completion_limit_are_refused(
    files_env, field, value
):
    env = await files_env()
    exchange = await _post(env, [(field, value)])
    assert_core_error(exchange, 400, "invalid_config_field", param=field)
    assert env.calls == []


# @spec ING-FILE-010
async def test_beam_search_false_is_accepted(files_env):
    env = await files_env()
    exchange = await _post(env, [("use_beam_search", "false")])
    assert exchange.status == 200, exchange.body
    assert len(env.calls) == 1


# @spec ING-FILE-020, ING-FILE-021
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", "0.5"),
        ("top_p", "0.9"),
        ("top_k", "5"),
        ("min_p", "0.1"),
        ("frequency_penalty", "0.5"),
        ("presence_penalty", "-0.5"),
        ("repetition_penalty", "1.2"),
        ("seed", "7"),
        ("seed", "0"),
        # Core's default for ``watermarking`` is absent, so any explicit
        # value is non-default, as an explicit ``seed`` of 0 is.
        ("watermarking", "true"),
        ("watermarking", "false"),
    ],
)
async def test_explicit_non_default_sampling_values_are_refused(
    files_env, field, value
):
    env = await files_env()
    exchange = await _post(env, [(field, value)])
    assert_core_error(exchange, 400, "invalid_config_field", param=field)
    assert env.calls == []


# @spec ING-FILE-020, ING-FILE-021
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", "0"),
        ("temperature", "0.0"),
        ("frequency_penalty", "0"),
        ("presence_penalty", "0.0"),
        ("top_p", "1"),
        ("top_p", "1.0"),
        ("repetition_penalty", "1"),
        ("top_k", "0"),
        ("min_p", "0"),
    ],
)
async def test_explicit_default_sampling_values_are_accepted(
    files_env, field, value
):
    env = await files_env()
    exchange = await _post(env, [(field, value)])
    assert exchange.status == 200, exchange.body
    assert len(env.calls) == 1


# @spec ING-FILE-021
async def test_omitted_fields_are_accepted_and_inherited_values_not_checked(
    files_env,
):
    service = FakeTranscriptionServing()
    service.default_sampling_params = {"temperature": 0.9, "top_k": 40}
    env = await files_env(service=service)
    exchange = await _post(env, [])
    assert exchange.status == 200, exchange.body
    assert len(env.calls) == 1


# @spec ING-FILE-022
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", "2"),
        ("length_penalty", "2.0"),
        ("include_stop_str_in_output", "true"),
    ],
)
async def test_explicit_non_default_ignored_fields_are_refused(
    files_env, field, value
):
    env = await files_env()
    exchange = await _post(env, [(field, value)])
    assert_core_error(exchange, 400, "invalid_config_field", param=field)
    assert env.calls == []


# @spec ING-FILE-022
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", "1"),
        ("length_penalty", "1.0"),
        ("include_stop_str_in_output", "false"),
    ],
)
async def test_default_ignored_fields_are_accepted(files_env, field, value):
    env = await files_env()
    exchange = await _post(env, [(field, value)])
    assert exchange.status == 200, exchange.body
    assert len(env.calls) == 1


# @spec ING-FILE-022
async def test_model_refused_fields_pass_to_core(files_env):
    env = await files_env()
    exchange = await _post(
        env,
        [("prompt", "context"), ("hotwords", "riva"), ("to_language", "de")],
    )
    assert exchange.status == 200, exchange.body
    (call,) = env.calls
    assert call.request.prompt == "context"
    assert call.request.hotwords == "riva"
    assert call.request.to_language == "de"


# @spec ING-FILE-010, ING-FILE-020, ING-FILE-022, ING-FILE-055
async def test_guard_refuses_before_taking_a_permit(files_env):
    env = await files_env(riva_frontend={"max_concurrent_files": 1})
    block = Block()
    env.service.script(block)
    holder = env.request().start()
    await asyncio.wait_for(block.entered.wait(), 2)
    for field, value in [
        ("use_beam_search", "true"),
        ("temperature", "0.3"),
        ("n", "3"),
        ("unknown_field", "x"),
    ]:
        refused = await env.request([(field, value)]).finish(timeout=2)
        assert refused.status == 400, (field, refused.body)
        assert env.plugin.file_gate.waiting == 0
    block.release()
    await holder.finish()


# ------------------------------------------------------------- allowlist


# @spec ING-FILE-023
async def test_unknown_text_part_is_refused_naming_it(files_env):
    env = await files_env()
    exchange = await _post(env, [("beam_width", "4")])
    assert_core_error(exchange, 400, "invalid_config_field", param="beam_width")
    assert env.calls == []


# @spec ING-FILE-023
async def test_alias_part_is_allowed(files_env):
    env = await files_env()
    exchange = await _post(
        env,
        [
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "segment"),
        ],
    )
    assert exchange.status == 200, exchange.body
    (call,) = env.calls
    assert call.request.timestamp_granularities == ["segment"]


# @spec ING-FILE-023
def test_allowlist_pins_core_request_fields_and_aliases():
    from vllm.entrypoints.speech_to_text.transcription.protocol import (
        TranscriptionRequest,
    )
    from vllm_riva_frontend.files import REQUEST_FIELD_ALLOWLIST

    core = set()
    for name, info in TranscriptionRequest.model_fields.items():
        core.add(name)
        if info.alias:
            core.add(info.alias)
    # Fails when core adds or renames a field: update the frontend's list.
    assert core == CORE_FIELD_NAMES_AND_ALIASES
    assert set(REQUEST_FIELD_ALLOWLIST) - {"file"} == core - {"file"}
    assert "timestamp_granularities[]" in REQUEST_FIELD_ALLOWLIST


# @spec ING-FILE-020, ING-FILE-022
def test_guard_field_sets_match_the_design():
    from vllm_riva_frontend.files import (
        FORWARDED_SAMPLING_FIELDS,
        IGNORED_FIELDS,
    )

    assert set(FORWARDED_SAMPLING_FIELDS) == GUARDED_SAMPLING
    assert set(IGNORED_FIELDS) == IGNORED


# @spec ING-FILE-057
def test_core_forwards_no_request_field_the_guard_does_not_handle():
    """Perturb each request field and see which reach ``SamplingParams``.

    ``stream`` selects the output kind (delivery, not sampling) and is
    excluded: the route serves ``stream=true``. The baseline samples
    (temperature 0.5), because greedy decoding resets ``top_p``, ``top_k``
    and ``min_p`` and would hide their forwarding.
    """
    from fastapi import UploadFile
    from vllm.entrypoints.speech_to_text.transcription.protocol import (
        TranscriptionRequest,
    )
    from vllm_riva_frontend.files import FORWARDED_SAMPLING_FIELDS

    handled = set(FORWARDED_SAMPLING_FIELDS) | {
        "max_completion_tokens",
        "vllm_xargs",
    }
    perturbations = {
        "model": "other",
        "language": "de",
        "hotwords": "x",
        "prompt": "x",
        "response_format": "text",
        "timestamp_granularities": ["word"],
        "to_language": "de",
        "use_beam_search": True,
        "n": 3,
        "length_penalty": 2.0,
        "include_stop_str_in_output": True,
        "temperature": 0.7,
        "watermarking": True,
        "top_p": 0.5,
        "top_k": 3,
        "min_p": 0.2,
        "seed": 11,
        "frequency_penalty": 0.4,
        "repetition_penalty": 1.3,
        "presence_penalty": 0.4,
        "max_completion_tokens": 9,
        "vllm_xargs": {"locale": "en-US"},
    }
    excluded = {"file", "stream", "stream_include_usage"}
    excluded |= {"stream_continuous_usage_stats"}
    fields = set(TranscriptionRequest.model_fields)
    unknown = fields - excluded - set(perturbations)
    assert not unknown, f"core added request fields: {sorted(unknown)}"

    def params(**overrides):
        request = TranscriptionRequest.model_validate(
            {
                "file": UploadFile(io.BytesIO(b"x")),
                "temperature": 0.5,
                **overrides,
            }
        )
        sampling = request.to_sampling_params(448, {})
        return {
            slot: repr(getattr(sampling, slot, None))
            for slot in sorted(dir(sampling))
            if not slot.startswith("_")
            and not callable(getattr(sampling, slot, None))
        }

    baseline = params()
    forwarded = set()
    for name, value in perturbations.items():
        if params(**{name: value}) != baseline:
            forwarded.add(name)
    assert forwarded <= handled, (
        f"core forwards unguarded fields: {sorted(forwarded - handled)}"
    )


# ------------------------------------------------------ vllm_xargs, locale


# @spec ING-FILE-024
async def test_xargs_with_only_locale_is_validated_and_forwarded(files_env):
    env = await files_env()
    exchange = await _post(
        env, [("vllm_xargs", json.dumps({"locale": "de-DE"}))]
    )
    assert exchange.status == 200, exchange.body
    (call,) = env.calls
    assert call.request.vllm_xargs == {"locale": "de-DE"}


# @spec ING-FILE-024
async def test_xargs_with_unknown_locale_is_refused(files_env):
    env = await files_env()
    exchange = await _post(
        env, [("vllm_xargs", json.dumps({"locale": "xx-XX"}))]
    )
    assert_core_error(exchange, 400, "unknown_locale")
    assert env.calls == []


# @spec ING-FILE-024
@pytest.mark.parametrize(
    "xargs",
    [{"beam": 2}, {"locale": "en-US", "extra": 1}, {"Locale": "en-US"}],
    ids=["foreign", "locale-plus-foreign", "case-variant"],
)
async def test_xargs_with_other_keys_is_refused(files_env, xargs):
    env = await files_env()
    exchange = await _post(env, [("vllm_xargs", json.dumps(xargs))])
    assert_core_error(exchange, 400, "invalid_config_field")
    assert env.calls == []


# @spec ING-FILE-025
@pytest.mark.parametrize("language", ["en", "de", "EN", "zz"])
async def test_language_is_forwarded_unchanged(files_env, language):
    env = await files_env()
    exchange = await _post(env, [("language", language)])
    assert exchange.status == 200, exchange.body
    (call,) = env.calls
    assert call.request.language == language
    assert call.request.vllm_xargs is None


# @spec ING-FILE-025
async def test_absent_language_stays_absent(files_env):
    env = await files_env()
    exchange = await _post(env, [])
    assert exchange.status == 200, exchange.body
    (call,) = env.calls
    assert call.request.language is None
    assert call.request.vllm_xargs is None


# ------------------------------------------------------------ duplicates


# @spec ING-FILE-042, ING-FILE-005
@pytest.mark.parametrize("field", ["model", "language", "temperature"])
async def test_duplicate_scalar_field_is_malformed(files_env, field):
    env = await files_env()
    value = "0" if field == "temperature" else "en"
    exchange = await _post(env, [(field, value), (field, value)])
    assert_core_error(exchange, 400, "malformed_request")
    assert env.calls == []


# @spec ING-FILE-042
async def test_repeated_list_field_is_allowed(files_env):
    env = await files_env()
    exchange = await _post(
        env,
        [
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
        ],
    )
    assert exchange.status == 200, exchange.body
    (call,) = env.calls
    assert call.request.timestamp_granularities == ["word", "segment"]


# @spec ING-FILE-042
async def test_invalid_value_gets_core_validation_error(files_env):
    env = await files_env()
    exchange = await _post(env, [("response_format", "xml")])
    # Core's validation error: its handler answers 400 in core's shape.
    assert exchange.status == 400, exchange.body
    assert set(exchange.json()["error"]) == {"message", "type", "param", "code"}
    assert env.calls == []


# @spec ING-FILE-042
async def test_request_is_built_with_core_request_model(files_env):
    from vllm.entrypoints.speech_to_text.transcription.protocol import (
        TranscriptionRequest,
    )

    env = await files_env()
    exchange = await _post(env, [("language", "en"), ("stream", "false")])
    assert exchange.status == 200
    (call,) = env.calls
    assert isinstance(call.request, TranscriptionRequest)
    await until(lambda: True)

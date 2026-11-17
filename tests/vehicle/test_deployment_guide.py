"""Checks of the repository's deployment guide, ``docs/deployment.md``.

The guide is parsed as text: the ``VLLM_PLUGINS`` assignment, the JSON
objects passed to ``--additional-config`` and ``--stream-admission-timeouts``,
``--shutdown-timeout`` and ``--endpoint-plugin-drain-timeout``, the
reference serve command (the first ``bash`` block that runs ``vllm serve``),
its ``--max-model-len`` and core's decode-duration limit. A required
statement is checked
as a paragraph or list item of the guide that carries every term the
statement needs, so a term scattered over unrelated paragraphs does not
count.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

GUIDE = Path(__file__).parents[2] / "docs" / "deployment.md"
RATE = 16_000
DEFAULT_DRAINING_TIMEOUT = 40
#: Core's decode-duration limit and its default, in seconds.
DECODE_LIMIT_VARIABLE = "VLLM_MAX_AUDIO_DECODE_DURATION_S"
DEFAULT_DECODE_LIMIT = 600
#: The default lookahead the reference checkpoint's model configuration sets;
#: the reference command serves it.
DEFAULT_LOOKAHEAD = 3
#: The lookaheads the guide offers as an override.
OFFERED_LOOKAHEADS = (0, 1, 3, 6, 13)
#: The model plugin's lookahead settings the reference command must not set.
LOOKAHEAD_KEYS = ("supported_num_lookahead_tokens", "num_lookahead_tokens")


def guide_text() -> str:
    assert GUIDE.is_file(), "docs/deployment.md is the deployment guide"
    return GUIDE.read_text(encoding="utf-8")


def json_objects_after(text: str, marker: str) -> list[dict[str, Any]]:
    """Every JSON object that directly follows an occurrence of ``marker``."""
    found: list[dict[str, Any]] = []
    for match in re.finditer(re.escape(marker), text):
        start = text.find("{", match.end())
        if start < 0:
            continue
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    found.append(json.loads(text[start : index + 1]))
                    break
    return found


def additional_config(text: str) -> dict[str, Any]:
    objects = json_objects_after(text, "--additional-config")
    assert objects, "the guide passes --additional-config"
    merged: dict[str, Any] = {}
    for item in objects:
        merged.update(item)
    return merged


def stream_admission_timeouts(text: str) -> dict[str, Any]:
    """The ``--stream-admission-timeouts`` object the guide's commands pass.

    Only fenced code is read, and only an object that directly follows the
    flag, so prose that names the flag or shows an example sets nothing.
    """
    code = "\n".join(re.findall(r"```.*?```", text, flags=re.S))
    merged: dict[str, Any] = {}
    for match in re.finditer(
        r"--stream-admission-timeouts[\s=]+['\"]?(?=\{)", code
    ):
        merged.update(json_objects_after(code[match.start() :], match[0])[0])
    return merged


def optional_flag_number(text: str, flag: str) -> float | None:
    match = re.search(
        re.escape(flag) + r"[ =]+([0-9][0-9_]*(?:\.[0-9]+)?)", text
    )
    return float(match.group(1).replace("_", "")) if match else None


def flag_number(text: str, flag: str) -> float:
    value = optional_flag_number(text, flag)
    assert value is not None, f"the guide sets {flag}"
    return value


def model_budget(seconds: float, lookahead: int) -> int:
    """The model's worst-case token budget for a file of ``seconds``."""
    samples = int(seconds * RATE)
    chunk = (lookahead + 1) * 1280
    whole_chunks = samples // chunk
    q = (lookahead + 1) * 10 + 1
    return 1 + (whole_chunks + 1) * q + 1


def reference_command(text: str) -> str:
    """The guide's reference serve command: the first ``bash`` block of it.

    The first fenced ``bash`` block that contains ``vllm serve``; prose and
    other blocks are not part of it.
    """
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.S):
        if "vllm serve" in block:
            return block
    raise AssertionError("the guide has a bash block that runs vllm serve")


def decode_limit(text: str) -> float:
    """Core's decode-duration limit as the guide sets it, else the default.

    The guide changes the limit only by assigning
    ``VLLM_MAX_AUDIO_DECODE_DURATION_S``; every assignment must agree.
    """
    values = {
        float(match.group(1))
        for match in re.finditer(
            re.escape(DECODE_LIMIT_VARIABLE)
            + r"\s*=\s*[\"']?([0-9]+(?:\.[0-9]+)?)",
            text,
        )
    }
    assert len(values) <= 1, f"the guide sets {DECODE_LIMIT_VARIABLE} twice"
    return values.pop() if values else DEFAULT_DECODE_LIMIT


def serve_command_failures(text: str) -> list[str]:
    """Why the guide's reference serve command fails ING-EP-078, if it does.

    The command sets no lookahead, so it serves the checkpoint's default,
    and its ``--max-model-len`` covers a file as long as core's
    decode-duration limit at that lookahead.
    """
    failures: list[str] = []
    command = reference_command(text)
    for key in LOOKAHEAD_KEYS:
        if re.search(rf"(?<![\w]){key}\b", command):
            failures.append(f"the reference serve command sets {key}")
    limit = decode_limit(text)
    changed = limit != DEFAULT_DECODE_LIMIT
    if changed and not any(
        DECODE_LIMIT_VARIABLE in item and re.search(rf"\b{limit:g}\s*s\b", item)
        for item in statements(text)
    ):
        failures.append(
            f"the guide changes {DECODE_LIMIT_VARIABLE} to {limit:g} s "
            "without stating the limit"
        )
    needed = model_budget(limit, DEFAULT_LOOKAHEAD)
    configured = flag_number(command, "--max-model-len")
    if configured < needed:
        failures.append(
            f"--max-model-len {configured:g} is below the {needed} tokens a "
            f"{limit:g} s file needs at the default lookahead "
            f"{DEFAULT_LOOKAHEAD}"
        )
    return failures


def statements(text: str) -> list[str]:
    """The guide's paragraphs and list items, outside fenced code."""
    prose = re.sub(r"```.*?```", "\n\n", text, flags=re.S)
    items: list[str] = []
    for paragraph in re.split(r"\n\s*\n", prose):
        current: list[str] = []
        for line in paragraph.splitlines():
            if re.match(r"\s*(?:[-*+]|[0-9]+\.)\s", line) and current:
                items.append(" ".join(current))
                current = []
            current.append(line.strip())
        if current:
            items.append(" ".join(current))
    return [item for item in items if item]


def states(text: str, *patterns: str) -> bool:
    """Whether one statement of the guide matches every regex in ``patterns``.

    Matching ignores case and Markdown code spans.
    """
    for item in statements(text):
        plain = item.replace("`", "")
        if all(re.search(p, plain, re.I) for p in patterns):
            return True
    return False


# @spec ING-EP-078
@pytest.mark.parametrize(
    "lookahead, budget",
    [(0, 82_513), (1, 78_773), (3, 76_918), (6, 76_114), (13, 75_578)],
)
def test_budget_formula_matches_the_documented_values(
    lookahead, budget
) -> None:
    assert model_budget(DEFAULT_DECODE_LIMIT, lookahead) == budget


# @spec ING-EP-113
def test_guide_allowlists_both_plugins() -> None:
    text = guide_text()
    assert re.search(
        r"VLLM_PLUGINS=[\"']?riva_frontend,nemotron_asr_streaming\b", text
    ), "VLLM_PLUGINS=riva_frontend,nemotron_asr_streaming"


# @spec ING-EP-113
def test_guide_passes_frontend_options_through_additional_config() -> None:
    config = additional_config(guide_text())
    assert isinstance(config.get("riva_frontend"), dict)


# @spec ING-EP-113
def test_guide_passes_frontend_options_that_validate() -> None:
    from vllm_riva_frontend.config import parse_config

    parse_config(additional_config(guide_text()))


# @spec ING-EP-113
def test_guide_drain_target_exceeds_the_engine_draining_timeout() -> None:
    """The drain's graceful-work target ``D`` lies past ``draining_timeout``.

    Core gives plugin drains ``--endpoint-plugin-drain-timeout`` seconds of
    ``--shutdown-timeout`` (half of it when the flag is omitted) and the
    frontend keeps ``engine_reserve`` of that budget, so ``D`` falls
    ``T - engine_reserve(T)`` after the drain begins.
    """
    from vllm_riva_frontend.lifecycle import engine_reserve

    text = guide_text()
    command = reference_command(text)
    draining = stream_admission_timeouts(text).get(
        "draining_timeout", DEFAULT_DRAINING_TIMEOUT
    )
    shutdown = flag_number(command, "--shutdown-timeout")
    drain_budget = optional_flag_number(
        command, "--endpoint-plugin-drain-timeout"
    )
    if drain_budget is None:
        drain_budget = shutdown / 2
    assert drain_budget <= shutdown, (
        "core refuses an --endpoint-plugin-drain-timeout above "
        "--shutdown-timeout"
    )
    target = drain_budget - engine_reserve(drain_budget)
    assert target > draining, (
        f"the drain target {target:g} s is not above draining_timeout "
        f"{draining:g} s (drain budget {drain_budget:g} s of "
        f"--shutdown-timeout {shutdown:g} s)"
    )


# @spec ING-EP-113
def test_guide_passes_no_stream_admission_key_in_additional_config() -> None:
    assert "stream_admission" not in additional_config(guide_text()), (
        "core does not read additional_config['stream_admission']"
    )


# @spec ING-EP-113
def test_guide_sets_changed_deadlines_with_the_scheduler_flag() -> None:
    assert states(
        guide_text(),
        r"--stream-admission-timeouts",
        r"never_fed_timeout",
        r"parked_idle_timeout",
        r"draining_timeout",
    ), "the guide says changed deadlines go to --stream-admission-timeouts"


# @spec ING-EP-113
def test_guide_stream_admission_timeouts_are_readable() -> None:
    """Whatever the guide passes, the frontend reads as core would."""
    from types import SimpleNamespace

    from vllm_riva_frontend.config import (
        DEFAULT_STREAM_DEADLINES,
        read_stream_deadlines,
    )

    timeouts = stream_admission_timeouts(guide_text())
    assert set(timeouts) <= {
        "never_fed_timeout",
        "parked_idle_timeout",
        "draining_timeout",
    }
    scheduler = SimpleNamespace(stream_admission_timeouts=timeouts)
    read = read_stream_deadlines(SimpleNamespace(scheduler_config=scheduler))
    assert read.draining_timeout == timeouts.get(
        "draining_timeout", DEFAULT_STREAM_DEADLINES.draining_timeout
    )


# @spec ING-EP-078
def test_reference_serve_command_sets_no_lookahead() -> None:
    command = reference_command(guide_text())
    for key in LOOKAHEAD_KEYS:
        assert key not in command, f"the reference serve command sets {key}"


# @spec ING-EP-078
def test_guide_sizes_max_model_len_for_cores_decode_limit() -> None:
    """``--max-model-len`` covers a file as long as core's decode limit.

    The limit is ``VLLM_MAX_AUDIO_DECODE_DURATION_S`` (600 s unless the
    guide changes it), not ``unary_max_decoded_duration_seconds``: the
    frontend keeps that option at or below core's limit, so core's limit is
    the longest file either file surface admits. The count is taken at the
    checkpoint's default lookahead, which the reference command serves.
    """
    failures = serve_command_failures(guide_text())
    assert not failures, "\n".join(failures)


# @spec ING-EP-078
@pytest.mark.parametrize("lookahead", OFFERED_LOOKAHEADS)
def test_guide_states_the_count_for_each_offered_lookahead(lookahead) -> None:
    count = model_budget(decode_limit(guide_text()), lookahead)
    assert states(
        guide_text(), rf"\br\s*=\s*{lookahead}\b", rf"\b{count:,}\b"
    ), f"the guide states that at r = {lookahead} a file needs {count:,}"


#: ING-EP-078's statements, each ``statement: patterns``.
LOOKAHEAD_STATEMENTS = {
    "the deployment needs a model plugin release whose open takes only the "
    "locale and that uses the checkpoint's default lookahead": (
        r"model plugin release",
        r"\bopen\b",
        r"only the locale",
        r"checkpoint's (?:default )?lookahead",
    ),
    "a model plugin that requires num_lookahead_tokens fails startup": (
        r"\brequires? num_lookahead_tokens",
        r"fails startup",
    ),
    "a factory whose open requires a cadence raises on every open, so the "
    "server starts and each stream ends with internal": (
        r"\bopen\b",
        r"requires? a cadence",
        r"raises on every open",
        r"server starts",
        r"each stream ends with internal",
    ),
    "clients cannot choose a cadence per request": (
        r"clients? cannot choose a cadence per request",
    ),
    "a deployment setting an override needs --max-model-len at least that "
    "count": (
        r"\boverride",
        r"--max-model-len",
        r"at least",
    ),
}

LOOKAHEAD_IDS = [
    "locale-only-release",
    "required-lookahead-fails",
    "cadence-open-internal",
    "no-per-request-cadence",
    "override-budget",
]


# @spec ING-EP-078
@pytest.mark.parametrize(
    "statement", list(LOOKAHEAD_STATEMENTS), ids=LOOKAHEAD_IDS
)
def test_guide_states_the_model_plugin_and_lookahead_requirements(
    statement,
) -> None:
    found = states(guide_text(), *LOOKAHEAD_STATEMENTS[statement])
    assert found, f"the guide states that {statement}"


_SERVE = "```bash\n{env}vllm serve m --max-model-len {length}{extra}\n```\n"
_OVERRIDE = " \\\n    --hf-overrides '{\"num_lookahead_tokens\": 6}'"
_RAISED = "export VLLM_MAX_AUDIO_DECODE_DURATION_S=1200\n"
_STATED = "Core's `VLLM_MAX_AUDIO_DECODE_DURATION_S` is raised to 1200 s."
_DEFAULT = model_budget(DEFAULT_DECODE_LIMIT, DEFAULT_LOOKAHEAD)
_RAISED_BUDGET = model_budget(1200, DEFAULT_LOOKAHEAD)


# @spec ING-EP-078
@pytest.mark.parametrize(
    "env, length, extra, prose, expected",
    [
        ("", 76_918, "", "", []),
        ("", 76_917, "", "", ["600 s file"]),
        (_RAISED, _RAISED_BUDGET, "", _STATED, []),
        (_RAISED, _RAISED_BUDGET - 1, "", _STATED, ["1200 s file"]),
        (_RAISED, _RAISED_BUDGET, "", "", ["without stating"]),
        ("", 76_918, _OVERRIDE, "", ["sets num_lookahead_tokens"]),
    ],
    ids=[
        "default-fits",
        "default-short",
        "raised-fits",
        "raised-short",
        "raised-unstated",
        "sets-lookahead",
    ],
)
def test_serve_command_check_follows_the_decode_limit_and_lookahead(
    env, length, extra, prose, expected
) -> None:
    assert _DEFAULT == 76_918
    command = _SERVE.format(env=env, length=length, extra=extra)
    text = command + "\n" + prose + "\n"
    failures = serve_command_failures(text)
    assert len(failures) == len(expected), failures
    for needle in expected:
        assert any(needle in failure for failure in failures), failures


# @spec ING-EP-078
def test_only_the_reference_command_is_read_for_lookahead() -> None:
    """Prose and later blocks may name the lookahead settings."""
    text = (
        _SERVE.format(env="", length=76_918, extra="")
        + "\nSet `num_lookahead_tokens` to serve another cadence.\n\n"
        + _SERVE.format(env="", length=76_918, extra=_OVERRIDE)
    )
    assert serve_command_failures(text) == []


#: ING-EP-104's statements, each ``statement: patterns``.
EXPOSURE_STATEMENTS = {
    "orchestrators probe core's /ready for readiness": (
        r"(?<![\w/])/ready\b",
        r"\bprobe",
        r"readiness",
    ),
    "the stock Realtime client runs with --server host:port/plugins/riva": (
        r"--server\s+\S+/plugins/riva(?![\w/])",
        r"realtime",
        r"client",
    ),
    "core's POST /v1/audio/transcriptions is outside the ingress cap, file "
    "gate and guard and is not to be exposed to untrusted clients": (
        r"POST /v1/audio/transcriptions",
        r"ingress cap",
        r"file gate",
        r"\bguard\b",
        r"untrusted",
    ),
    "under --api-key an intermediary in front of the stock Realtime client "
    "adds the bearer token to the bootstrap request and the WebSocket "
    "handshake": (
        r"--api-key",
        r"intermediary",
        r"bearer",
        r"bootstrap",
        r"handshake",
    ),
    "the gRPC listener is outside --api-key and VLLM_API_KEY and is not to "
    "be exposed to untrusted clients without an authenticating intermediary": (
        r"\bgRPC\b",
        r"--api-key",
        r"VLLM_API_KEY",
        r"untrusted",
        r"authenticating intermediary",
    ),
}

#: ING-EP-105's statements, each ``statement: patterns``.
OPERATION_STATEMENTS = {
    "one API server process (--api-server-count at most 1) and data "
    "parallel size 1 are required": (
        r"--api-server-count",
        r"data parallel",
        r"\b1\b",
    ),
    "the kill grace (terminationGracePeriodSeconds) is above the sum of "
    "--shutdown-timeout and --endpoint-plugin-stop-timeout": (
        r"terminationGracePeriodSeconds",
        r"\b(?:above|exceeds|more than)\s+the\s+sum\s+of\s+"
        r"--shutdown-timeout\s+and\s+--endpoint-plugin-stop-timeout\b",
    ),
    "at a zero drain budget stop runs the whole settlement on a stopped "
    "engine and can need up to session_cleanup_timeout plus the factory "
    "settlement, so the deployment sets --endpoint-plugin-stop-timeout above "
    "session_cleanup_timeout; otherwise core cancels stop and owner cleanup "
    "may be cut short": (
        r"\bzero\b|--shutdown-timeout 0\b|\b0 s\b",
        r"stopped engine",
        r"factory settlement",
        r"--endpoint-plugin-stop-timeout\b.*\babove\b.*session_cleanup_timeout",
        r"\bcancels\b.*\bstop\b",
        r"cut short",
    ),
    "the serve mode that runs core's Rust frontend is unsupported, because "
    "no endpoint plugin loads there": (
        r"\bRust\b",
        r"unsupported|not supported",
        r"endpoint plugin",
    ),
}

EXPOSURE_IDS = [
    "ready-probe",
    "client-server-path",
    "core-file-route",
    "api-key-intermediary",
    "grpc-exposure",
]
OPERATION_IDS = [
    "one-api-server",
    "kill-grace",
    "zero-budget-bound",
    "rust-frontend",
]


#: Statements that reverse the comparison ING-EP-105 requires; the check
#: must reject each of them.
REVERSED_OPERATION_STATEMENTS = {
    "kill-grace": (
        "The kill grace (for Kubernetes, `terminationGracePeriodSeconds`) is "
        "below the sum of `--shutdown-timeout` and "
        "`--endpoint-plugin-stop-timeout`."
    ),
    "zero-budget-bound": (
        "At a zero drain budget core calls no `drain`, so `stop` runs the "
        "whole settlement on a stopped engine and can need up to "
        "`session_cleanup_timeout` plus the factory settlement. A deployment "
        "with a zero drain budget sets `--endpoint-plugin-stop-timeout` below "
        "`session_cleanup_timeout`. Otherwise core cancels `stop` at the "
        "limit, and owner cleanup may be cut short."
    ),
}


# @spec ING-EP-105
@pytest.mark.parametrize("statement_id", list(REVERSED_OPERATION_STATEMENTS))
def test_operation_checks_reject_the_reversed_comparison(statement_id) -> None:
    statement = list(OPERATION_STATEMENTS)[OPERATION_IDS.index(statement_id)]
    reversed_text = REVERSED_OPERATION_STATEMENTS[statement_id]
    assert not states(reversed_text, *OPERATION_STATEMENTS[statement]), (
        f"the check for {statement_id} accepts: {reversed_text}"
    )
    upright = re.sub(r"\bbelow\b", "above", reversed_text)
    assert states(upright, *OPERATION_STATEMENTS[statement]), (
        f"the check for {statement_id} rejects: {upright}"
    )


# @spec ING-EP-104
@pytest.mark.parametrize(
    "statement", list(EXPOSURE_STATEMENTS), ids=EXPOSURE_IDS
)
def test_guide_states_ready_probe_client_path_core_route_and_grpc_exposure(
    statement,
) -> None:
    found = states(guide_text(), *EXPOSURE_STATEMENTS[statement])
    assert found, f"the guide states that {statement}"


# @spec ING-EP-105
@pytest.mark.parametrize(
    "statement", list(OPERATION_STATEMENTS), ids=OPERATION_IDS
)
def test_guide_states_one_api_server_kill_grace_and_rust_frontend_unsupported(
    statement,
) -> None:
    found = states(guide_text(), *OPERATION_STATEMENTS[statement])
    assert found, f"the guide states that {statement}"


# @spec ING-EP-104, ING-EP-105
@pytest.mark.parametrize(
    "statement",
    [*EXPOSURE_STATEMENTS, *OPERATION_STATEMENTS],
    ids=[*EXPOSURE_IDS, *OPERATION_IDS],
)
def test_statement_check_accepts_the_statement_and_not_scattered_terms(
    statement,
) -> None:
    """Each check accepts its own statement, not its terms spread apart."""
    patterns = {**EXPOSURE_STATEMENTS, **OPERATION_STATEMENTS}[statement]
    assert states(f"Intro.\n\n- {statement}.\n", *patterns)
    words = statement.split()
    half = len(words) // 2
    scattered = " ".join(words[:half]) + "\n\n" + " ".join(words[half:])
    assert not states(scattered, *patterns)

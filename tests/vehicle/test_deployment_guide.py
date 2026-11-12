"""Checks of the repository's deployment guide, ``docs/deployment.md``.

The guide is parsed as text: the ``VLLM_PLUGINS`` assignment, the JSON
object passed to ``--additional-config``, ``--shutdown-timeout``,
``--max-model-len``, core's decode-duration limit and the model plugin's
lookahead settings. A required statement is checked as a paragraph or list
item of the guide that carries every term the statement needs, so a term
scattered over unrelated paragraphs does not count.
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


def flag_number(text: str, flag: str) -> float:
    match = re.search(
        re.escape(flag) + r"[ =]+([0-9][0-9_]*(?:\.[0-9]+)?)", text
    )
    assert match, f"the guide sets {flag}"
    return float(match.group(1).replace("_", ""))


def model_budget(seconds: float, lookahead: int) -> int:
    """The model's worst-case token budget for a file of ``seconds``."""
    samples = int(seconds * RATE)
    chunk = (lookahead + 1) * 1280
    whole_chunks = samples // chunk
    q = (lookahead + 1) * 10 + 1
    return 1 + (whole_chunks + 1) * q + 1


def find_lookahead(text: str) -> tuple[list[int], int]:
    supported = re.search(
        r"supported_num_lookahead_tokens\W*\[([0-9,\s]+)\]", text
    )
    assert supported, "the guide sets supported_num_lookahead_tokens"
    values = [
        int(v) for v in supported.group(1).replace(" ", "").split(",") if v
    ]
    single = re.search(r"(?<!supported_)num_lookahead_tokens\W*([0-9]+)", text)
    assert single, "the guide sets num_lookahead_tokens"
    return values, int(single.group(1))


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


def max_model_len_failures(text: str) -> list[str]:
    """Why the guide's ``--max-model-len`` fails ING-EP-078, if it does."""
    failures: list[str] = []
    _, lookahead = find_lookahead(text)
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
    needed = model_budget(limit, lookahead)
    configured = flag_number(text, "--max-model-len")
    if configured < needed:
        failures.append(
            f"--max-model-len {configured:g} is below the {needed} tokens a "
            f"{limit:g} s file needs at num_lookahead_tokens {lookahead}"
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
@pytest.mark.parametrize("lookahead, budget", [(13, 75_578), (6, 76_114)])
def test_budget_formula_matches_the_documented_values(
    lookahead, budget
) -> None:
    assert model_budget(DEFAULT_DECODE_LIMIT, lookahead) == budget


# @spec ING-EP-019
def test_guide_allowlists_both_plugins() -> None:
    text = guide_text()
    assert re.search(
        r"VLLM_PLUGINS=[\"']?riva_frontend,nemotron_asr_streaming\b", text
    ), "VLLM_PLUGINS=riva_frontend,nemotron_asr_streaming"


# @spec ING-EP-019
def test_guide_passes_frontend_options_through_additional_config() -> None:
    config = additional_config(guide_text())
    assert isinstance(config.get("riva_frontend"), dict)


# @spec ING-EP-019
def test_guide_passes_frontend_options_that_validate() -> None:
    from vllm_riva_frontend.config import parse_config

    parse_config(additional_config(guide_text()))


# @spec ING-EP-019
def test_guide_shutdown_timeout_exceeds_the_engine_draining_timeout() -> None:
    text = guide_text()
    config = additional_config(text)
    draining = config.get("stream_admission", {}).get(
        "draining_timeout", DEFAULT_DRAINING_TIMEOUT
    )
    assert flag_number(text, "--shutdown-timeout") > draining


# @spec ING-EP-078
def test_guide_serves_the_streaming_arm_and_its_file_lookahead() -> None:
    supported, lookahead = find_lookahead(guide_text())
    assert 6 in supported, "6 is the 560 ms streaming arm"
    assert lookahead in supported


# @spec ING-EP-078
def test_guide_sizes_max_model_len_for_cores_decode_limit() -> None:
    """``--max-model-len`` covers a file as long as core's decode limit.

    The limit is ``VLLM_MAX_AUDIO_DECODE_DURATION_S`` (600 s unless the
    guide changes it), not ``unary_max_decoded_duration_seconds``: the
    frontend keeps that option at or below core's limit, so core's limit is
    the longest file either file surface admits.
    """
    failures = max_model_len_failures(guide_text())
    assert not failures, "\n".join(failures)


_OVERRIDES = (
    '{"supported_num_lookahead_tokens": [6, 13], "num_lookahead_tokens": 13}'
)
_SERVE = (
    "```bash\n{env}vllm serve m --max-model-len {length} \\\n"
    "    --hf-overrides '"
    + _OVERRIDES.replace("{", "{{").replace("}", "}}")
    + "'\n```\n"
)


_RAISED = "export VLLM_MAX_AUDIO_DECODE_DURATION_S=1200\n"
_STATED = "Core's `VLLM_MAX_AUDIO_DECODE_DURATION_S` is raised to 1200 s."


# @spec ING-EP-078
@pytest.mark.parametrize(
    "env, length, prose, expected",
    [
        ("", 75_578, "", []),
        ("", 75_577, "", ["600 s file"]),
        (_RAISED, 151_154, _STATED, []),
        (_RAISED, 76_800, _STATED, ["1200 s file"]),
        (_RAISED, 151_154, "", ["without stating"]),
    ],
    ids=[
        "default-fits",
        "default-short",
        "raised-fits",
        "raised-short",
        "raised-unstated",
    ],
)
def test_max_model_len_check_follows_the_decode_limit(
    env, length, prose, expected
) -> None:
    text = _SERVE.format(env=env, length=length) + "\n" + prose + "\n"
    failures = max_model_len_failures(text)
    assert len(failures) == len(expected), failures
    for needle in expected:
        assert any(needle in failure for failure in failures), failures


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
    "the kill grace (terminationGracePeriodSeconds) is above "
    "--shutdown-timeout with room for stop and the engine's shutdown": (
        r"terminationGracePeriodSeconds",
        r"--shutdown-timeout",
        r"\bstop\b",
        r"engine",
    ),
    "at a zero shutdown budget the kill grace is the only bound besides a "
    "repeated signal on a stop that waits on a stopped engine": (
        r"\bzero\b|--shutdown-timeout 0\b|\b0 s\b",
        r"repeated signal",
        r"\bstop\b",
        r"stopped engine",
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

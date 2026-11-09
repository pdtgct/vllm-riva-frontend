"""Checks of the repository's deployment guide, ``docs/deployment.md``.

The guide is parsed as text: the ``VLLM_PLUGINS`` assignment, the JSON
object passed to ``--additional-config``, ``--shutdown-timeout`` and
``--max-model-len``, and the model plugin's lookahead settings.
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
DEFAULT_MAX_DURATION = 600


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


# @spec ING-EP-058
@pytest.mark.parametrize("lookahead, budget", [(13, 75_578), (6, 76_114)])
def test_budget_formula_matches_the_documented_values(
    lookahead, budget
) -> None:
    assert model_budget(600, lookahead) == budget


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


# @spec ING-EP-058
def test_guide_serves_the_streaming_arm_and_its_file_lookahead() -> None:
    supported, lookahead = find_lookahead(guide_text())
    assert 6 in supported, "6 is the 560 ms streaming arm"
    assert lookahead in supported


# @spec ING-EP-058
def test_guide_max_model_len_covers_the_longest_admitted_file() -> None:
    text = guide_text()
    _, lookahead = find_lookahead(text)
    options = additional_config(text).get("riva_frontend", {})
    seconds = options.get(
        "unary_max_decoded_duration_seconds", DEFAULT_MAX_DURATION
    )
    assert flag_number(text, "--max-model-len") >= model_budget(
        seconds, lookahead
    )

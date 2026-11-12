"""Operations checks of the deployment guide, ``docs/deployment.md``.

The guide is parsed as text, paragraph by paragraph: it must say where a
liveness probe without the bearer token goes (core's ``/health``) and that
tooling probing the frontend's operational routes under ``--api-key``
supplies the token.
"""

from __future__ import annotations

import re
from pathlib import Path

GUIDE = Path(__file__).parents[2] / "docs" / "deployment.md"
NAMESPACE = "/plugins/riva/v1/"


def _paragraphs() -> list[str]:
    assert GUIDE.is_file(), "docs/deployment.md is the deployment guide"
    text = GUIDE.read_text(encoding="utf-8")
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def _mentions_core_health(paragraph: str) -> bool:
    return re.search(r"`/health`|core's /health", paragraph) is not None


# @spec ING-SHIM-026
def test_guide_states_liveness_probe_and_token_for_operational_routes() -> None:
    paragraphs = _paragraphs()
    liveness = [
        p
        for p in paragraphs
        if re.search(r"\bliveness\b", p, re.IGNORECASE)
        and re.search(r"\btoken\b", p)
        and _mentions_core_health(p)
    ]
    assert liveness, (
        "the guide states that a liveness probe that cannot supply the "
        "bearer token probes core's /health"
    )
    operational = [
        p
        for p in paragraphs
        if "--api-key" in p
        and NAMESPACE in p
        and re.search(r"\btoken\b", p)
        and re.search(r"\boperational\b", p)
    ]
    assert operational, (
        "the guide states that under --api-key, tooling that probes the "
        f"operational routes under {NAMESPACE} supplies the token"
    )

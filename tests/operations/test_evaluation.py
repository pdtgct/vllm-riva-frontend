"""Evaluation discipline: the frontend ships no Riva-family client.

That the surfaces are driven by the canonical ``nvidia-riva/python-clients``
tools in evaluation is a property of the evaluation procedure, not of this
package. What a CPU test can check is that the package implements no
bespoke client (no gRPC channel or Riva stub, no WebSocket client and no
client console script), and that the repository's guides (``README.md``
and ``docs/deployment.md``) run the stock Realtime client with
``--server host:port/plugins/riva`` against a server with API-key
authentication disabled.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
PACKAGE = ROOT / "src" / "vllm_riva_frontend"

CLIENT_NAMES = {
    "RivaSpeechRecognitionStub",
    "insecure_channel",
    "secure_channel",
    "ASRService",
}
CLIENT_MODULES = {"websockets", "websocket", "aiohttp", "riva.client.asr"}


def _names(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.alias):
            found.add(node.name.rsplit(".", 1)[-1])
    return found


def _modules(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


# @spec ING-CLI-005
def test_the_package_implements_no_riva_family_client() -> None:
    import vllm_riva_frontend.grpc  # noqa: F401  (the servicer module)
    import vllm_riva_frontend.nim_ws  # noqa: F401  (the adapter module)

    offenders = {}
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text())
        names = _names(tree) & CLIENT_NAMES
        modules = {
            m
            for m in _modules(tree)
            if any(m == c or m.startswith(c + ".") for c in CLIENT_MODULES)
        }
        if names or modules:
            offenders[path.name] = sorted(names | modules)
    assert offenders == {}


# @spec ING-CLI-005
def test_the_distribution_ships_no_client_script() -> None:
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    scripts = project.get("scripts", {})
    assert not any(
        "client" in name or "client" in target
        for name, target in scripts.items()
    ), scripts


GUIDES = (ROOT / "README.md", ROOT / "docs" / "deployment.md")
#: ``--server <host>:<port>/plugins/riva`` (the base, not a ``/v1`` path).
SERVER_ARGUMENT = re.compile(
    r"--server[ =]+[^\s/`'\"]+:[^\s/`'\"]+/plugins/riva(?![\w/])"
)


def _paragraphs() -> list[str]:
    paragraphs: list[str] = []
    for guide in GUIDES:
        assert guide.is_file(), guide
        text = guide.read_text(encoding="utf-8")
        paragraphs += [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs


# @spec ING-CLI-005
def test_guides_give_the_realtime_client_the_plugin_base_as_server() -> None:
    found = [p for p in _paragraphs() if SERVER_ARGUMENT.search(p)]
    assert found, (
        "a guide runs the stock Realtime client with "
        "--server host:port/plugins/riva"
    )


# @spec ING-CLI-006
def test_guides_run_the_stock_realtime_client_with_auth_disabled() -> None:
    found = [
        p
        for p in _paragraphs()
        if ("--server" in p or "Realtime client" in p)
        and "--api-key" in p
        and "VLLM_API_KEY" in p
        and re.search(r"\bneither\b|\bwithout\b|\bdisabled\b", p)
    ]
    assert found, (
        "a guide runs the stock Realtime client with neither --api-key "
        "nor VLLM_API_KEY set"
    )

"""Supported-interpreter contract for the downstream plugin package.

The plugin is imported into the host's own interpreter, so the package's
declared range is a deployability constraint rather than a packaging
preference: a floor above the host's Python makes the distribution
uninstallable where it is meant to run.

These checks deliberately avoid ``compileall``. It halts at the first
syntax error in a module, so it under-reports how much of a package is
broken, and its exit status is easily masked by a pipeline. The gate
below parses every module independently and reports all of them.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from typing import Any

#: Lowest interpreter the distribution supports, as (major, minor).
#: vLLM core's floor is 3.11.
LOWEST_SUPPORTED = (3, 11)

_PROJECT_ROOT = Path(__file__).parents[1]
_PACKAGE_ROOT = _PROJECT_ROOT / "src" / "vllm_riva_frontend"

#: Names that only exist in ``typing`` above the supported floor. Each maps
#: to the substitute that carries the same meaning at the floor.
_TYPING_ABOVE_FLOOR = {
    "Never": "NoReturn",
    "Self": "the concrete class name",
    "LiteralString": "str",
    "TypeAliasType": "TypeAlias",
    "override": "no annotation",
    "assert_never": "an explicit raise",
}


def _project() -> dict[str, Any]:
    with (_PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _modules() -> list[Path]:
    return sorted(_PACKAGE_ROOT.glob("*.py"))


# @spec ING-VEH-025
def test_declared_range_matches_the_host_supported_interpreters() -> None:
    """The floor is vLLM core's, not an independent choice."""
    requires = _project()["project"]["requires-python"]

    assert requires == ">=3.11,<3.15", (
        "the declared range must mirror vLLM core's requires-python; "
        f"found {requires!r}"
    )


# @spec ING-VEH-025
def test_every_module_parses_on_the_lowest_supported_version() -> None:
    """Report every module that uses post-floor syntax, not just the first."""
    feature_version = LOWEST_SUPPORTED[1]
    offenders: list[str] = []
    for module in _modules():
        try:
            ast.parse(
                module.read_text(encoding="utf-8"),
                filename=str(module),
                feature_version=feature_version,
            )
        except SyntaxError as error:
            offenders.append(f"{module.name}:{error.lineno}: {error.msg}")

    assert not offenders, (
        "modules use syntax newer than Python "
        f"{'.'.join(str(part) for part in LOWEST_SUPPORTED)}:\n  "
        + "\n  ".join(offenders)
    )


# @spec ING-VEH-025
def test_no_module_imports_a_typing_name_above_the_floor() -> None:
    """An import executes even when the annotation using it does not."""
    offenders: list[str] = []
    for module in _modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "typing":
                continue
            for alias in node.names:
                substitute = _TYPING_ABOVE_FLOOR.get(alias.name)
                if substitute is not None:
                    offenders.append(
                        f"{module.name}:{node.lineno}: typing.{alias.name} "
                        f"(use {substitute})"
                    )

    assert not offenders, "typing names above the supported floor:\n  " + (
        "\n  ".join(offenders)
    )


# @spec ING-VEH-025
def test_tooling_floors_name_the_lowest_supported_version() -> None:
    """Linter and type checker must judge at the floor, not the ceiling."""
    project = _project()
    floor = ".".join(str(part) for part in LOWEST_SUPPORTED)

    assert project["tool"]["ruff"]["target-version"] == "py" + floor.replace(
        ".", ""
    )
    assert project["tool"]["mypy"]["python_version"] == floor


# @spec ING-VEH-025
def test_floor_can_read_toml_from_the_stdlib() -> None:
    """3.11 has tomllib, so the dev group does not carry a tomli fallback."""
    dev_dependencies = _project()["dependency-groups"]["dev"]
    assert LOWEST_SUPPORTED >= (3, 11)
    assert not any(item.startswith("tomli") for item in dev_dependencies)

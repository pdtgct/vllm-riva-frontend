"""The development distribution ``vllm-riva-frontend-dev`` (ING-CORE-036).

The seam conformance suite ships as a second distribution, built from
``dev/``, so that the frontend's own wheel carries no test code and no test
dependency. These read both ``pyproject.toml`` files and the sources (static
checks that read the syntax tree), import the development package in a fresh
interpreter, and run the metadata half of ``scripts/check_dev_wheels.py``
against wheels built from a copy of the repository.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tomllib
import types
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "vllm_riva_frontend"
DEV_ROOT = PROJECT_ROOT / "dev"
DEV_PACKAGE_ROOT = DEV_ROOT / "src" / "vllm_riva_frontend_dev"
SUITE = DEV_PACKAGE_ROOT / "seam_conformance.py"
CONSUMER = PROJECT_ROOT / "tests" / "session" / "test_seam_conformance.py"
CHECK_SCRIPT = PROJECT_ROOT / "scripts" / "check_dev_wheels.py"

#: What the development package exports (ING-CORE-036).
EXPORTS = frozenset(
    {"Return", "Raise", "Block", "SeamError", "refused", "failed", "settle"}
)
#: Import names the frontend distribution never imports.
TEST_ONLY_IMPORTS = frozenset(
    {"pytest", "pytest_asyncio", "vllm_riva_frontend_dev"}
)


#: Imports the development package in a fresh interpreter and reports which
#: test modules that loaded, its ``__all__`` and the exports it lacks.
IMPORT_PROBE = """
import json, sys
import vllm_riva_frontend_dev as package
suite = "vllm_riva_frontend_dev.seam_conformance"
watched = ("pytest", "pytest_asyncio", suite)
print(json.dumps({
    "loaded": sorted(name for name in watched if name in sys.modules),
    "all": sorted(package.__all__),
    "missing": sorted(n for n in package.__all__ if not hasattr(package, n)),
}))
"""


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _requirement(item: str) -> Any:
    from packaging.requirements import Requirement

    return Requirement(item)


def _frontend_version() -> str:
    from vllm_riva_frontend.version import __version__

    return __version__


def _top_level_imports(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom) and node.module and not node.level
        ):
            found.add(node.module.split(".")[0])
    return found


def _is_mark(node: ast.expr, name: str) -> bool:
    """Whether ``node`` is ``pytest.mark.<name>`` (called or not)."""
    if isinstance(node, ast.Call):
        node = node.func
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    )


def _is_fixture(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        node = node.func
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "fixture"
        and isinstance(node.value, ast.Name)
        and node.value.id == "pytest"
    )


def _fixture_name(decorator: ast.expr) -> str | None:
    """The ``name=`` a fixture decorator gives, if any."""
    if isinstance(decorator, ast.Call):
        for keyword in decorator.keywords:
            if keyword.arg == "name" and isinstance(
                keyword.value, ast.Constant
            ):
                return str(keyword.value.value)
    return None


def _load_check_script() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_dev_wheels", CHECK_SCRIPT
    )
    assert spec is not None and spec.loader is not None, CHECK_SCRIPT
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("check_dev_wheels", module)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------- the frontend distribution


# @spec ING-CORE-036
def test_no_frontend_module_imports_pytest_or_the_dev_package() -> None:
    paths = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert paths, PACKAGE_ROOT
    offenders = {
        str(path.relative_to(PACKAGE_ROOT)): sorted(found)
        for path in paths
        if (
            found := _top_level_imports(
                ast.parse(path.read_text(encoding="utf-8"))
            )
            & TEST_ONLY_IMPORTS
        )
    }
    assert offenders == {}


# @spec ING-CORE-036
def test_the_frontend_declares_no_test_dependency() -> None:
    project = _toml(PROJECT_ROOT / "pyproject.toml")["project"]
    declared = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)
    names = [_requirement(item).name.lower() for item in declared]
    assert [name for name in names if name.startswith("pytest")] == []


# ---------------------------------------------- the development distribution


# @spec ING-CORE-036
def test_the_dev_distribution_pins_the_frontend_at_its_version() -> None:
    version = _frontend_version()
    root = _toml(PROJECT_ROOT / "pyproject.toml")["project"]
    dev = _toml(DEV_ROOT / "pyproject.toml")["project"]

    assert dev["name"] == "vllm-riva-frontend-dev"
    assert dev["requires-python"] == root["requires-python"]
    assert dev["license"] == root["license"]
    assert (DEV_ROOT / "LICENSE").read_bytes() == (
        PROJECT_ROOT / "LICENSE"
    ).read_bytes()
    requirements = {
        _requirement(item).name.lower(): _requirement(item)
        for item in dev["dependencies"]
    }
    assert {"vllm-riva-frontend", "pytest", "pytest-asyncio", "numpy"} <= set(
        requirements
    )
    frontend = requirements["vllm-riva-frontend"]
    assert str(frontend.specifier) == f"=={version}", (
        "the development distribution pins the frontend at exactly its version"
    )

    import vllm_riva_frontend_dev

    assert vllm_riva_frontend_dev.__version__ == version


# @spec ING-CORE-036
def test_importing_the_dev_package_imports_neither_pytest_nor_the_suite() -> (
    None
):
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["loaded"] == []
    assert report["all"] == sorted(EXPORTS | {"__version__"})
    assert report["missing"] == []


# @spec ING-CORE-036
def test_the_shared_test_support_reexports_the_dev_packages_types() -> None:
    import vllm_riva_frontend_dev as package
    from support import clock, fake_session

    for name in ("Return", "Raise", "Block", "SeamError", "refused", "failed"):
        assert getattr(fake_session, name) is getattr(package, name), name
    assert fake_session._DEFAULT is package._DEFAULT
    assert clock.settle is package.settle


# @spec ING-CORE-036, ING-CORE-022
def test_every_suite_test_is_async_and_carries_its_own_asyncio_mark() -> None:
    tree = ast.parse(SUITE.read_text(encoding="utf-8"))
    tests = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    ]
    assert tests, "the suite holds the seam conformance tests"
    not_async = [
        t.name for t in tests if not isinstance(t, ast.AsyncFunctionDef)
    ]
    unmarked = [
        t.name
        for t in tests
        if not any(_is_mark(d, "asyncio") for d in t.decorator_list)
    ]
    assert not_async == [] and unmarked == []

    seam_fixtures = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (
            node.name == "seam"
            or any(
                _is_fixture(d) and _fixture_name(d) == "seam"
                for d in node.decorator_list
            )
        )
    ]
    assert seam_fixtures == [], "the consumer defines seam"
    pytestmark = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
    ]
    assert pytestmark == [], "each test carries its own mark"


# @spec ING-CORE-036, ING-CORE-022
def test_the_cpu_lane_runs_the_packaged_suite_against_the_fake() -> None:
    tree = ast.parse(CONSUMER.read_text(encoding="utf-8"))
    own_tests = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    ]
    assert own_tests == [], "the suite's tests live in the package"
    star = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "vllm_riva_frontend_dev.seam_conformance"
        and [alias.name for alias in node.names] == ["*"]
    ]
    assert len(star) == 1, "the consumer star-imports the suite"
    seam = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "seam"
        and any(_is_fixture(d) for d in node.decorator_list)
    ]
    assert len(seam) == 1, "the consumer defines the seam fixture"


# @spec ING-CORE-036
def test_the_suite_is_rewritten_for_assertions() -> None:
    """``tests/conftest.py`` registers the suite before it is imported."""
    import vllm_riva_frontend_dev.seam_conformance as suite

    loader = type(getattr(suite, "__loader__", None)).__name__
    assert loader == "AssertionRewritingHook", loader


# @spec ING-CORE-036
def test_the_wheel_check_counts_what_the_cpu_lane_collects() -> None:
    """The count the wheel check expects is the CPU lane's collected count.

    Collected without the conftest files (which import the vLLM tree and add
    no tests), so the collection takes a fraction of a second.
    """
    expected = _load_check_script().suite_test_count(SUITE)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-cov",
            "--noconftest",
            str(CONSUMER.relative_to(PROJECT_ROOT)),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    collected = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("tests/session/test_seam_conformance.py::")
    ]
    assert expected > 0
    assert len(collected) == expected, (expected, collected)


# @spec ING-CORE-036
def test_the_built_wheels_metadata_passes_the_wheel_check(tmp_path) -> None:
    """Build both wheels from a copy and run the check's metadata half.

    The copy keeps the build's ``build/`` and ``*.egg-info`` out of the
    working tree.
    """
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH: the wheel build needs it")
    copy = tmp_path / "repo"
    ignore = shutil.ignore_patterns(
        ".git",
        ".venv",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "__pycache__",
        "build",
        "dist",
        "*.egg-info",
        ".coverage",
    )
    copy.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", "src", "dev"):
        source = PROJECT_ROOT / name
        if source.is_dir():
            shutil.copytree(source, copy / name, ignore=ignore)
        else:
            shutil.copy2(source, copy / name)
    dist = tmp_path / "dist"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("VIRTUAL_ENV", "UV_PROJECT"))
    }
    for project in (copy, copy / "dev"):
        result = subprocess.run(
            [uv, "build", "--wheel", "--out-dir", str(dist), str(project)],
            cwd=copy,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            env=environment,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    problems = _load_check_script().check_metadata(dist)
    assert problems == []


def _fake_wheel(
    dist: Path, name: str, version: str, metadata: str, files: list[str]
) -> None:
    import zipfile

    stem = f"{name.replace('-', '_')}-{version}"
    with zipfile.ZipFile(dist / f"{stem}-py3-none-any.whl", "w") as archive:
        archive.writestr(
            f"{stem}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n"
            + metadata,
        )
        for path in files:
            archive.writestr(path, "")


# @spec ING-CORE-036
@pytest.mark.parametrize(
    "path",
    [
        "vllm_riva_frontend/test_accidental.py",
        "vllm_riva_frontend/foo_test.py",
        "vllm_riva_frontend/conftest.py",
    ],
)
def test_the_wheel_check_rejects_a_test_module_in_the_frontend_wheel(
    tmp_path, path
) -> None:
    check = _load_check_script()
    _fake_wheel(
        tmp_path,
        "vllm-riva-frontend",
        "1.0",
        "Requires-Python: >=3.11\n",
        ["vllm_riva_frontend/__init__.py", path],
    )
    _fake_wheel(
        tmp_path,
        "vllm-riva-frontend-dev",
        "1.0",
        "Requires-Python: >=3.11\nRequires-Dist: vllm-riva-frontend==1.0\n",
        [
            "vllm_riva_frontend_dev/__init__.py",
            "vllm_riva_frontend_dev/seam_conformance.py",
        ],
    )
    problems = check.check_metadata(tmp_path)
    assert any(
        "contains test code" in item and path in item for item in problems
    ), problems


# @spec ING-CORE-036
def test_the_wheel_check_reports_every_metadata_problem(tmp_path) -> None:
    check = _load_check_script()
    _fake_wheel(
        tmp_path,
        "vllm-riva-frontend",
        "1.0",
        "Requires-Python: >=3.11\nRequires-Dist: pytest>=8\n",
        ["vllm_riva_frontend/__init__.py", "tests/test_x.py"],
    )
    _fake_wheel(
        tmp_path,
        "vllm-riva-frontend-dev",
        "1.1",
        "Requires-Python: >=3.12\nRequires-Dist: vllm-riva-frontend>=1.0\n",
        ["vllm_riva_frontend_dev/__init__.py"],
    )
    problems = check.check_metadata(tmp_path)
    for expected in (
        "versions differ",
        "Requires-Python differs",
        "exactly",
        "requires test code",
        "contains test code",
        "lacks the suite",
    ):
        assert any(expected in problem for problem in problems), (
            expected,
            problems,
        )

"""Check the two built wheels and the suite's collection from them.

ING-CORE-036: the frontend's wheel and the development distribution's
wheel, built from one revision, are checked in two halves.

* Metadata: both wheels carry the same ``Version`` and ``Requires-Python``;
  the development wheel requires ``vllm-riva-frontend`` at exactly that
  version and contains the suite; the frontend wheel requires no ``pytest``
  distribution and contains no test module (no path under ``tests``, a
  ``*_dev`` package or ``seam_conformance``).
* Consumer: both wheels are installed into a fresh virtual environment in a
  temporary directory outside the checkout (no vLLM tree), where a consumer
  module star-imports the suite and defines ``seam``, and the suite is
  collected under pytest-asyncio's strict mode. The check fails unless both
  packages resolve from that environment, the collected count equals the
  count the frontend's CPU lane collects (the suite's test functions times
  their parametrizations, read from the checkout's suite source), every
  collected test carries the ``asyncio`` mark and nothing is skipped.

Usage: ``python scripts/check_dev_wheels.py [DIST] [--expected N]
[--metadata-only]``, with ``DIST`` the directory holding both wheels
(default ``dist``). Exits 1 and lists every problem found.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "dev" / "src" / "vllm_riva_frontend_dev" / "seam_conformance.py"
FRONTEND = "vllm-riva-frontend"
DEV = "vllm-riva-frontend-dev"
#: A requirement's name, optional extras and specifier.
REQUIREMENT = re.compile(r"([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*(.*)")

CONSUMER_CONFTEST = """\
import importlib.util
import json
import os

import pytest

pytest.register_assert_rewrite("vllm_riva_frontend_dev.seam_conformance")

_SKIPPED = []


def pytest_collectreport(report):
    if not report.passed:
        _SKIPPED.append(f"{report.nodeid}: {report.outcome}")


def pytest_collection_finish(session):
    origins = {
        name: importlib.util.find_spec(name).origin
        for name in ("vllm_riva_frontend", "vllm_riva_frontend_dev")
    }
    items = [
        {
            "nodeid": item.nodeid,
            "markers": sorted({mark.name for mark in item.iter_markers()}),
        }
        for item in session.items
    ]
    with open(os.environ["CHECK_DEV_WHEELS_REPORT"], "w") as stream:
        json.dump(
            {"origins": origins, "items": items, "skipped": _SKIPPED}, stream
        )
"""

CONSUMER_MODULE = '''\
"""A consumer of the seam conformance suite (collection only)."""

import pytest
from vllm_riva_frontend_dev.seam_conformance import *  # noqa: F403


@pytest.fixture
def seam():
    raise AssertionError("collection only: no adapter")
'''

CONSUMER_INI = """\
[pytest]
asyncio_default_fixture_loop_scope = function
"""


def suite_test_count(path: Path = SUITE) -> int:
    """The number of tests pytest collects from the suite at ``path``.

    Each top-level ``test*`` function counts once per combination of its
    ``pytest.mark.parametrize`` decorators, whose value lists must be
    literals.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    total = 0
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test"):
            continue
        count = 1
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "parametrize"
            ):
                values = decorator.args[1] if len(decorator.args) > 1 else None
                if not isinstance(values, (ast.List, ast.Tuple)):
                    raise ValueError(
                        f"{path}:{node.lineno}: parametrize values of "
                        f"{node.name} are not a literal list"
                    )
                count *= len(values.elts)
        total += count
    return total


def _wheel(dist: Path, distribution: str) -> tuple[Path | None, str]:
    prefix = distribution.replace("-", "_") + "-"
    found = sorted(
        path
        for path in dist.glob("*.whl")
        if path.name.startswith(prefix)
        and path.name[len(prefix) :][:1].isdigit()
    )
    if len(found) != 1:
        return None, f"expected one {distribution} wheel in {dist}: {found}"
    return found[0], ""


def _metadata(wheel: Path) -> tuple[list[str], dict[str, list[str]]]:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        (path,) = [n for n in names if n.endswith(".dist-info/METADATA")]
        text = archive.read(path).decode("utf-8")
    message = Parser().parsestr(text, headersonly=True)
    fields: dict[str, list[str]] = {}
    for key, value in message.items():
        fields.setdefault(key, []).append(value)
    return names, fields


def _requirement(value: str) -> tuple[str, str]:
    """The normalized name and the specifier of a ``Requires-Dist``."""
    requirement = value.split(";", 1)[0].strip()
    match = REQUIREMENT.match(requirement)
    if match is None:
        return requirement.lower(), ""
    name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
    return name, match.group(2).replace(" ", "")


def _is_test_module(basename: str) -> bool:
    """Whether a file name is one pytest collects or loads as test code."""
    return basename == "conftest.py" or (
        basename.endswith(".py")
        and (basename.startswith("test_") or basename.endswith("_test.py"))
    )


def check_metadata(dist: Path) -> list[str]:
    """Every problem of the two wheels' metadata and contents."""
    frontend, problem = _wheel(dist, FRONTEND)
    dev, dev_problem = _wheel(dist, DEV)
    problems = [p for p in (problem, dev_problem) if p]
    if frontend is None or dev is None:
        return problems
    frontend_files, frontend_meta = _metadata(frontend)
    dev_files, dev_meta = _metadata(dev)

    version = frontend_meta.get("Version", [""])[0]
    if not version or dev_meta.get("Version") != [version]:
        problems.append(
            f"versions differ: {FRONTEND} {frontend_meta.get('Version')}, "
            f"{DEV} {dev_meta.get('Version')}"
        )
    if frontend_meta.get("Requires-Python") != dev_meta.get(
        "Requires-Python"
    ) or not frontend_meta.get("Requires-Python"):
        problems.append(
            "Requires-Python differs: "
            f"{frontend_meta.get('Requires-Python')} vs "
            f"{dev_meta.get('Requires-Python')}"
        )
    pins = [
        specifier
        for name, specifier in map(
            _requirement, dev_meta.get("Requires-Dist", [])
        )
        if name == FRONTEND
    ]
    if pins != [f"=={version}"]:
        problems.append(
            f"{DEV} must require {FRONTEND}=={version} exactly; found {pins}"
        )
    test_dependencies = [
        value
        for value in frontend_meta.get("Requires-Dist", [])
        if _requirement(value)[0].startswith("pytest")
    ]
    if test_dependencies:
        problems.append(f"{FRONTEND} requires test code: {test_dependencies}")
    test_files = [
        name
        for name in frontend_files
        if any(
            part == "tests"
            or part.endswith("_dev")
            or "seam_conformance" in part
            for part in name.split("/")
        )
        or _is_test_module(name.rsplit("/", 1)[-1])
    ]
    if test_files:
        problems.append(f"{FRONTEND} wheel contains test code: {test_files}")
    if "vllm_riva_frontend_dev/seam_conformance.py" not in dev_files:
        problems.append(f"{DEV} wheel lacks the suite")
    return problems


def _inside(path: str | None, root: Path) -> bool:
    if path is None:
        return False
    return Path(path).resolve().is_relative_to(root.resolve())


def check_consumer(dist: Path, expected: int) -> list[str]:
    """Every problem of collecting the suite from the installed wheels."""
    uv = shutil.which("uv")
    if uv is None:
        return ["uv is not on PATH"]
    wheels = [str(path) for path in sorted(dist.glob("*.whl"))]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME")
    }
    with tempfile.TemporaryDirectory(prefix="check-dev-wheels-") as scratch:
        root = Path(scratch).resolve()
        if root.is_relative_to(REPO):
            return [f"the temporary directory {root} is inside the checkout"]
        venv = root / "venv"
        python = venv / "bin" / "python"
        for command in (
            [uv, "venv", "--quiet", "--python", sys.executable, str(venv)],
            [uv, "pip", "install", "--quiet", "--python", str(python), *wheels],
        ):
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return [f"{' '.join(command)} failed:\n{result.stderr}"]
        consumer = root / "consumer"
        consumer.mkdir()
        (consumer / "conftest.py").write_text(CONSUMER_CONFTEST)
        (consumer / "test_consumer.py").write_text(CONSUMER_MODULE)
        (consumer / "pytest.ini").write_text(CONSUMER_INI)
        report = root / "report.json"
        result = subprocess.run(
            [
                str(python),
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                "-o",
                "asyncio_mode=strict",
            ],
            cwd=consumer,
            env={**environment, "CHECK_DEV_WHEELS_REPORT": str(report)},
            capture_output=True,
            text=True,
            check=False,
        )
        print(result.stdout.strip().splitlines()[-1] if result.stdout else "")
        if result.returncode != 0 or not report.is_file():
            return [
                f"collection failed ({result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}"
            ]
        data = json.loads(report.read_text())

    problems: list[str] = []
    for name, origin in data["origins"].items():
        if not _inside(origin, venv) or _inside(origin, REPO):
            problems.append(f"{name} resolves from {origin}, not the venv")
    items = data["items"]
    if len(items) != expected:
        problems.append(
            f"collected {len(items)} suite tests; the CPU lane collects "
            f"{expected}"
        )
    unmarked = [i["nodeid"] for i in items if "asyncio" not in i["markers"]]
    if unmarked:
        problems.append(f"tests without the asyncio mark: {unmarked}")
    skipped = [
        i["nodeid"] for i in items if {"skip", "skipif"} & set(i["markers"])
    ] + data["skipped"]
    if skipped:
        problems.append(f"skipped at collection: {skipped}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run both halves (or the metadata half) and report every problem."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dist", nargs="?", default="dist", type=Path)
    parser.add_argument(
        "--expected",
        type=int,
        default=None,
        help="the CPU lane's collected count (default: read from the suite)",
    )
    parser.add_argument("--metadata-only", action="store_true")
    arguments = parser.parse_args(argv)
    dist = arguments.dist.resolve()

    problems = check_metadata(dist)
    print(f"metadata: {'ok' if not problems else 'FAILED'}")
    if not arguments.metadata_only and not problems:
        expected = (
            arguments.expected
            if arguments.expected is not None
            else suite_test_count()
        )
        consumer = check_consumer(dist, expected)
        print(
            f"consumer: {'ok' if not consumer else 'FAILED'} "
            f"(expected {expected} suite tests)"
        )
        problems.extend(consumer)
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

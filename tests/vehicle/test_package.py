"""Package, declaration and structural rules of the vehicle leaf.

These read ``pyproject.toml``, the README and the CI workflow, and the
distributed modules under ``src/vllm_riva_frontend`` (static checks that read
the syntax tree, never comments or strings), run the floor gate on the lowest
supported interpreter, and import the package for the declaration.
"""

from __future__ import annotations

import ast
import collections
import importlib
import inspect
import json
import re
import shlex
import shutil
import subprocess
import sys
import threading
import tomllib
import types
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "vllm_riva_frontend"
README = PROJECT_ROOT / "README.md"
WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"

#: The interpreter range of the pinned core (``requires-python``).
CORE_RANGE = ">=3.11,<3.15"
LOWEST_SUPPORTED = (3, 11)

#: Names that exist in ``typing`` only above the floor, with the substitute.
TYPING_ABOVE_FLOOR = {
    "override": "no annotation",
    "TypeAliasType": "TypeAlias",
    "TypeIs": "TypeGuard",
    "ReadOnly": "a plain annotation",
    "NoDefault": "no default",
    "get_protocol_members": "a manual walk",
    "is_protocol": "a manual check",
}
#: Standard-library names above the floor, by module.
STDLIB_ABOVE_FLOOR = {
    ("itertools", "batched"): "3.12",
    ("warnings", "deprecated"): "3.13",
    ("copy", "replace"): "3.13",
}


def _project() -> dict[str, Any]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _modules() -> list[Path]:
    return sorted(PACKAGE_ROOT.glob("*.py"))


def _sources() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in _modules()}


def _trees() -> dict[str, ast.Module]:
    return {name: ast.parse(text) for name, text in _sources().items()}


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.\-]+)", requirement)
    assert match, requirement
    return match.group(1).lower().replace("_", "-")


def _module_name(path: Path) -> str:
    if path.stem == "__init__":
        return "vllm_riva_frontend"
    return f"vllm_riva_frontend.{path.stem}"


# ----------------------------------------------------------- declaration


# @spec ING-EP-001
def test_one_endpoint_plugin_entry_point_resolves_to_the_factory() -> None:
    entry_points = _project()["project"].get("entry-points", {})

    assert entry_points == {
        "vllm.endpoint_plugins": {"riva_frontend": "vllm_riva_frontend:plugin"}
    }, "exactly one entry point, riva_frontend in vllm.endpoint_plugins"


# @spec ING-EP-001, ING-EP-062
def test_entry_point_factory_takes_no_arguments_and_returns_the_plugin() -> (
    None
):
    import vllm_riva_frontend
    from vllm_riva_frontend.lifecycle import RivaFrontendPlugin

    signature = inspect.signature(vllm_riva_frontend.plugin)
    assert all(
        p.default is not inspect.Parameter.empty
        or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        for p in signature.parameters.values()
    )
    instance = vllm_riva_frontend.plugin()
    assert isinstance(instance, RivaFrontendPlugin)


# @spec ING-EP-062
def test_plugin_declares_name_and_required_tasks_only() -> None:
    """No class declaration asks core for one API server process.

    The one-process requirement is ``init_state``'s check (ING-EP-081);
    RFC 2's endpoint-plugin protocol has no such declaration.
    """
    from vllm_riva_frontend.lifecycle import RivaFrontendPlugin

    assert RivaFrontendPlugin.name == "riva_frontend"
    assert RivaFrontendPlugin.required_tasks == ("realtime",)
    assert not hasattr(RivaFrontendPlugin, "owns_non_http_listener")
    assert not hasattr(RivaFrontendPlugin(), "owns_non_http_listener")


# @spec ING-EP-062
def test_plugin_satisfies_cores_endpoint_plugin_protocol() -> None:
    from vllm.plugins.endpoint_plugins.interface import EndpointPlugin

    from vllm_riva_frontend.lifecycle import RivaFrontendPlugin

    plugin = RivaFrontendPlugin()
    assert isinstance(plugin, EndpointPlugin)
    for method in ("start", "drain", "stop", "init_state"):
        assert inspect.iscoroutinefunction(getattr(plugin, method)), method
    assert not inspect.iscoroutinefunction(plugin.ready)


# @spec ING-EP-002
def test_distribution_name_and_import_package() -> None:
    project = _project()["project"]
    assert project["name"] == "vllm-riva-frontend"
    package = importlib.import_module("vllm_riva_frontend")
    assert Path(package.__file__).parent == PACKAGE_ROOT


# @spec ING-EP-002
def test_runtime_dependencies_are_listed_and_riva_stubs_are_pinned() -> None:
    dependencies = _project()["project"]["dependencies"]
    names = {_requirement_name(item): item for item in dependencies}

    for required in (
        "nvidia-riva-client",
        "grpcio",
        "grpcio-health-checking",
        "grpcio-reflection",
        "numpy",
        "scipy",
        "python-multipart",
    ):
        assert required in names, f"missing runtime dependency {required}"
    assert "==" in names["nvidia-riva-client"], "the Riva stubs are pinned"


# @spec ING-EP-002, ING-EP-021
def test_no_dependency_on_vllm_or_the_model_plugin() -> None:
    project = _project()["project"]
    declared = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)
    names = {_requirement_name(item) for item in declared}

    assert not {n for n in names if n == "vllm" or n.startswith("vllm-")} - {
        "vllm-riva-frontend"
    }, "no resolver-owned dependency on vLLM (no version bound either)"
    assert not {n for n in names if "nemotron" in n}, (
        "no dependency on the model plugin"
    )


# @spec ING-EP-001
def test_core_tree_carries_no_riva_shaped_code() -> None:
    import vllm

    core = Path(vllm.__file__).parent
    pattern = re.compile(r"\briva\b|riva_|nvidia\.riva|riva\.client", re.I)
    offenders = [
        str(path.relative_to(core))
        for path in core.rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not offenders, offenders


# ------------------------------------------------- interpreter range


# @spec ING-VEH-027
def test_declared_range_is_vllms_requires_python() -> None:
    requires = _project()["project"]["requires-python"]
    assert requires == CORE_RANGE, (
        f"declare vLLM's range {CORE_RANGE!r}; found {requires!r}"
    )


# @spec ING-VEH-027
def test_declared_range_matches_the_pinned_core_tree() -> None:
    import vllm

    pyproject = Path(vllm.__file__).parents[1] / "pyproject.toml"
    assert pyproject.is_file(), "the core tree on PYTHONPATH has a pyproject"
    with pyproject.open("rb") as stream:
        core = tomllib.load(stream)
    assert (
        _project()["project"]["requires-python"]
        == core["project"]["requires-python"]
    )


# @spec ING-VEH-027
def test_every_module_parses_at_the_lowest_supported_version() -> None:
    offenders: list[str] = []
    for module in _modules():
        try:
            ast.parse(
                module.read_text(encoding="utf-8"),
                filename=str(module),
                feature_version=LOWEST_SUPPORTED,
            )
        except SyntaxError as error:
            offenders.append(f"{module.name}:{error.lineno}: {error.msg}")
    assert not offenders, "\n".join(offenders)


def _is_version_info(node: ast.expr) -> bool:
    """``sys.version_info`` or a subscript or attribute of it."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "version_info"
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
        ):
            return True
        node = node.value
    return False


#: ``sys.version_info`` on the lowest supported interpreter's first release.
FLOOR_VERSION_INFO = collections.namedtuple(
    "version_info", "major minor micro releaselevel serial"
)(*LOWEST_SUPPORTED, 0, "final", 0)

#: The node types a version predicate may contain: comparisons and boolean
#: operators over ``sys.version_info`` (indexed, sliced or by field) and
#: literals.
_PREDICATE_NODES = (
    ast.Compare,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.Subscript,
    ast.Slice,
    ast.Attribute,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Tuple,
    ast.cmpop,
)


def _floor_value(test: ast.expr) -> bool | None:
    """The value of a version predicate on the floor interpreter.

    ``None`` unless ``test`` reads ``sys.version_info`` and nothing but
    literals besides; the predicate is then evaluated with ``sys`` replaced
    by the floor's ``version_info``, whatever its threshold and whichever
    side of the comparison it is on.
    """
    nodes = list(ast.walk(test))
    if not all(isinstance(node, _PREDICATE_NODES) for node in nodes):
        return None
    if any(isinstance(n, ast.Name) and n.id != "sys" for n in nodes):
        return None
    if not any(_is_version_info(n) for n in nodes if isinstance(n, ast.expr)):
        return None
    floor_sys = types.SimpleNamespace(version_info=FLOOR_VERSION_INFO)
    code = compile(ast.Expression(body=test), "<version gate>", "eval")
    try:
        return bool(eval(code, {"__builtins__": {}, "sys": floor_sys}))
    except Exception:
        return None


def _gated_branch(test: ast.expr) -> str | None:
    """Which branch of an ``if`` never runs on the floor interpreter.

    ``"body"`` for ``TYPE_CHECKING`` and for a version predicate that is
    false on the floor; ``"orelse"`` for a version predicate that is true on
    the floor; ``None`` for any other test.
    """
    if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    ):
        return "body"
    value = _floor_value(test)
    if value is None:
        return None
    return "orelse" if value else "body"


_IMPORT_FALLBACKS = {"ImportError", "ModuleNotFoundError", "AttributeError"}


def _catches_import_failure(node: ast.Try) -> bool:
    for handler in node.handlers:
        caught = handler.type
        names = caught.elts if isinstance(caught, ast.Tuple) else [caught]
        for item in names:
            if isinstance(item, ast.Name) and item.id in _IMPORT_FALLBACKS:
                return True
    return False


def _gated(tree: ast.Module) -> set[int]:
    """The ids of nodes behind a version gate or an import fallback."""
    gated: set[int] = set()
    for node in ast.walk(tree):
        bodies: list[list[ast.stmt]] = []
        if isinstance(node, ast.If):
            branch = _gated_branch(node.test)
            if branch is not None:
                bodies.append(getattr(node, branch))
        elif isinstance(node, ast.Try) and _catches_import_failure(node):
            bodies.append(node.body)
        for body in bodies:
            for statement in body:
                gated.update(id(inner) for inner in ast.walk(statement))
    return gated


def _above_floor(name: str, tree: ast.Module) -> list[str]:
    """Unguarded uses of names introduced above the floor."""
    gated = _gated(tree)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if id(node) in gated:
            continue
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if node.module == "typing" and alias.name in TYPING_ABOVE_FLOOR:
                    offenders.append(
                        f"{name}:{node.lineno}: typing.{alias.name}"
                    )
                elif (node.module, alias.name) in STDLIB_ABOVE_FLOOR:
                    offenders.append(
                        f"{name}:{node.lineno}: {node.module}.{alias.name}"
                    )
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            key = (node.value.id, node.attr)
            if key in STDLIB_ABOVE_FLOOR or (
                node.value.id == "typing" and node.attr in TYPING_ABOVE_FLOOR
            ):
                offenders.append(
                    f"{name}:{node.lineno}: {node.value.id}.{node.attr}"
                )
    return offenders


# @spec ING-VEH-027
def test_no_module_uses_a_name_introduced_above_the_floor() -> None:
    offenders = [
        offender
        for name, tree in _trees().items()
        for offender in _above_floor(name, tree)
    ]
    assert not offenders, "\n".join(offenders)


# @spec ING-VEH-027
def test_the_floor_name_check_permits_gated_imports() -> None:
    gated = ast.parse(
        "import sys\n"
        "from typing import TYPE_CHECKING\n"
        "if sys.version_info >= (3, 12):\n"
        "    from typing import override\n"
        "else:\n"
        "    from typing_extensions import override\n"
        "if sys.version_info[:2] < (3, 13):\n"
        "    pass\n"
        "else:\n"
        "    from copy import replace\n"
        "try:\n"
        "    from itertools import batched\n"
        "except ImportError:\n"
        "    batched = None\n"
        "if TYPE_CHECKING:\n"
        "    from typing import TypeIs\n"
        "# from typing import override  (a comment is not code)\n"
        "NOTE = 'typing.override is a string, not a use'\n"
    )
    assert _above_floor("gated.py", gated) == []
    ungated = ast.parse(
        "import typing\n"
        "from typing import override\n"
        "from itertools import batched\n"
        "x = typing.TypeIs\n"
    )
    assert len(_above_floor("ungated.py", ungated)) == 3


# @spec ING-VEH-027
def test_the_floor_name_check_evaluates_the_gate_on_the_floor() -> None:
    # A gate that is true on 3.11 does not protect a 3.12 name.
    true_on_floor = ast.parse(
        "import sys\n"
        "if sys.version_info >= (3, 11):\n"
        "    from typing import override\n"
        "if sys.version_info[:2] < (3, 12):\n"
        "    pass\n"
        "else:\n"
        "    from itertools import batched\n"
        "if sys.version_info.minor > 10:\n"
        "    from copy import replace\n"
    )
    assert _above_floor("true_on_floor.py", true_on_floor) == [
        "true_on_floor.py:3: typing.override",
        "true_on_floor.py:9: copy.replace",
    ]
    # Whatever the orientation, a gate false on 3.11 does.
    reversed_gates = ast.parse(
        "import sys\n"
        "if (3, 12) <= sys.version_info:\n"
        "    from typing import override\n"
        "if (3, 13) > sys.version_info[:2]:\n"
        "    pass\n"
        "else:\n"
        "    from copy import replace\n"
        "if not sys.version_info < (3, 12):\n"
        "    from itertools import batched\n"
    )
    assert _above_floor("reversed.py", reversed_gates) == []
    # A test that is not a version predicate gates nothing.
    other = ast.parse(
        "import sys\n"
        "FLAG = True\n"
        "if FLAG and sys.version_info >= (3, 12):\n"
        "    from typing import override\n"
    )
    assert _above_floor("other.py", other) == ["other.py:4: typing.override"]


#: Compiles every path in ``argv`` from its source text (nothing is written
#: next to the sources), collects every failure and exits 1 if any.
COMPILE_GATE = """
import json, sys
bad = []
for path in sys.argv[1:]:
    try:
        with open(path, encoding="utf-8") as handle:
            compile(handle.read(), path, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as error:
        bad.append(f"{path}:{getattr(error, 'lineno', None)}: {error}")
print(json.dumps(bad))
sys.exit(1 if bad else 0)
"""

#: Reports which of the distributions named in ``argv`` are not installed
#: for the interpreter running it (by distribution metadata, so no import
#: name is guessed); exits 1 if any is missing.
PREFLIGHT = """
import importlib.metadata as metadata, json, sys
missing = []
for name in sys.argv[1:]:
    try:
        metadata.distribution(name)
    except metadata.PackageNotFoundError:
        missing.append(name)
print(json.dumps(missing))
sys.exit(1 if missing else 0)
"""

#: Imports every module named in ``argv[2:]`` with ``argv[1]`` (an
#: ``os.pathsep``-joined list: the source tree) ahead of ``sys.path``;
#: collects every failure, a missing module of any kind included (the
#: environment was checked first, so a missing module is the module's
#: defect, a misspelled import for instance).
IMPORT_GATE = """
import importlib, json, os, sys
sys.path[:0] = [p for p in sys.argv[1].split(os.pathsep) if p]
failures = []
for name in sys.argv[2:]:
    try:
        importlib.import_module(name)
    except Exception as error:
        failures.append(f"{name}: {type(error).__name__}: {error}")
print(json.dumps(failures))
sys.exit(1 if failures else 0)
"""

#: The host distribution the frontend is imported into. It is not a
#: declared dependency (ING-EP-021), but the modules import it, so the
#: floor environment must provide it.
HOST_DISTRIBUTION = "vllm"


def _floor_distributions() -> list[str]:
    """The distributions an import on the floor needs installed.

    The declared runtime dependencies whose markers hold on the floor, and
    the host.
    """
    from packaging.requirements import Requirement

    floor = ".".join(str(part) for part in LOWEST_SUPPORTED)
    environment = {"python_version": floor, "python_full_version": floor + ".0"}
    needed: list[str] = []
    for item in _project()["project"]["dependencies"]:
        requirement = Requirement(item)
        if requirement.marker is None or requirement.marker.evaluate(
            environment
        ):
            needed.append(requirement.name)
    return [*needed, HOST_DISTRIBUTION]


def _floor_interpreter() -> str:
    """A Python of the lowest supported version, or skip saying none exists.

    The running interpreter when it is the floor, else ``python3.11`` on
    ``PATH``. The skip is reported; the gate never passes without running.
    """
    floor = ".".join(str(part) for part in LOWEST_SUPPORTED)
    if sys.version_info[:2] == LOWEST_SUPPORTED:
        return sys.executable
    found = shutil.which(f"python{floor}")
    if found is None:
        pytest.skip(
            f"no Python {floor} interpreter found (python{floor} on PATH): "
            "the floor gate of ING-VEH-027 did not run"
        )
    reported = subprocess.run(
        [found, "-I", "-c", "import sys; print(list(sys.version_info[:2]))"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout.strip()
    assert json.loads(reported) == list(LOWEST_SUPPORTED), (found, reported)
    return found


def _gate(interpreter: str, script: str, *args: str) -> tuple[int, Any]:
    """Run a gate script; return its exit status (read directly) and report."""
    result = subprocess.run(
        [interpreter, "-I", "-B", "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    lines = result.stdout.strip().splitlines()
    assert lines, f"the gate printed no report: {result.stderr}"
    return result.returncode, json.loads(lines[-1])


# @spec ING-VEH-027
def test_floor_gate_compiles_every_module_and_reports_all_offenders() -> None:
    """Compile every module on the floor interpreter, in one fail-closed run.

    The gate compiles each module's source text (it writes no bytecode next
    to the sources), collects every failing module and exits non-zero; its
    status is read directly, with no pipeline.
    """
    interpreter = _floor_interpreter()
    modules = [str(path) for path in _modules()]
    assert modules, "no distributed modules found"
    status, offenders = _gate(interpreter, COMPILE_GATE, *modules)
    assert status == (1 if offenders else 0), (status, offenders)
    assert not offenders, "\n".join(offenders)


# @spec ING-VEH-027
def test_floor_gate_imports_every_module_and_reports_all_failures() -> None:
    """Import every distributed module on the floor interpreter.

    The floor interpreter imports the source tree with its own installed
    packages only: dependencies resolved for another interpreter (this run's
    ``PYTHONPATH``) are not floor evidence. A preflight first checks that
    the floor environment has every declared runtime dependency and the
    host; only a missing one of those skips, since the import half cannot
    be judged there. Then every module is tried and every failure is
    reported, any missing module included.
    """
    interpreter = _floor_interpreter()
    status, missing = _gate(interpreter, PREFLIGHT, *_floor_distributions())
    assert status == (1 if missing else 0), (status, missing)
    if missing:
        pytest.skip(
            f"the floor interpreter {interpreter} lacks {', '.join(missing)}; "
            "the import half of the floor gate needs the declared runtime "
            "dependencies and the host installed for that interpreter"
        )
    names = [_module_name(path) for path in _modules()]
    status, failures = _gate(
        interpreter, IMPORT_GATE, str(PACKAGE_ROOT.parent), *names
    )
    assert status == (1 if failures else 0), (status, failures)
    assert not failures, "\n".join(failures)


# @spec ING-VEH-027
def test_the_import_gate_fails_on_a_missing_module_it_was_not_promised(
    tmp_path,
) -> None:
    """A misspelled import is a failure of the module, never a skip."""
    package = tmp_path / "floorcheck"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "good.py").write_text("import json\n")
    (package / "typo.py").write_text("import nummpy_misspelled\n")
    status, failures = _gate(
        sys.executable,
        IMPORT_GATE,
        str(tmp_path),
        "floorcheck.good",
        "floorcheck.typo",
    )
    assert status == 1
    assert len(failures) == 1, failures
    assert failures[0].startswith("floorcheck.typo: ModuleNotFoundError")
    status, missing = _gate(sys.executable, PREFLIGHT, "no-such-dist-xyz")
    assert (status, missing) == (1, ["no-such-dist-xyz"])
    assert _gate(sys.executable, PREFLIGHT) == (0, [])
    assert HOST_DISTRIBUTION in _floor_distributions()
    assert "numpy" in _floor_distributions()


# @spec ING-VEH-027
def test_every_distributed_module_imports_and_all_failures_are_reported() -> (
    None
):
    failures: list[str] = []
    for module in _modules():
        name = _module_name(module)
        try:
            importlib.import_module(name)
        except Exception as error:  # every module is tried
            failures.append(f"{name}: {type(error).__name__}: {error}")
    assert not failures, "\n".join(failures)


# @spec ING-EP-022
def test_linter_and_type_checker_name_the_lowest_supported_version() -> None:
    tool = _project()["tool"]
    floor = ".".join(str(part) for part in LOWEST_SUPPORTED)
    assert tool["ruff"]["target-version"] == "py" + floor.replace(".", "")
    assert tool["mypy"]["python_version"] == floor


def _declared_versions() -> list[str]:
    """Every ``3.<minor>`` that ``requires-python`` admits, in order."""
    from packaging.specifiers import SpecifierSet

    requires = SpecifierSet(_project()["project"]["requires-python"])
    return [f"3.{m}" for m in range(40) if requires.contains(f"3.{m}.0")]


#: A reference to the matrix's interpreter.
_MATRIX_VERSION = re.compile(r"\$\{\{\s*matrix\.python-version\s*\}\}")
#: Any GitHub expression.
_EXPRESSION = re.compile(r"\$\{\{.*?\}\}", re.DOTALL)
#: A literal condition, bare or as an expression (``if: ${{ false }}``).
_LITERAL_CONDITION = re.compile(
    r"\s*(?:\$\{\{\s*(true|false)\s*\}\}|(true|false))\s*"
)
#: The one comparison ``if`` the check evaluates; any other condition is
#: unknown, so the step is not counted as running.
_VERSION_CONDITION = re.compile(
    r"\s*(?:\$\{\{)?\s*matrix\.python-version\s*(==|!=)\s*'([^']*)'"
    r"\s*(?:\}\})?\s*"
)
#: The words a ``run`` script's expressions become before it is split: the
#: matrix's interpreter, and any other expression.
_MATRIX_WORD = "__matrix.python-version__"
_OTHER_WORD = "__expression__"
#: The characters of a shell operator that ends a simple command.
_OPERATOR_CHARS = frozenset("();&|")
#: Words that may precede a command's name.
_COMMAND_PREFIXES = frozenset(
    ("if", "then", "elif", "else", "do", "while", "until", "!")  # keywords
    + ("time", "exec")  # commands that run the words after them
)
#: An environment assignment before a command's name (``CI=1 pytest``).
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.DOTALL)
#: Commands, and ``python -m`` modules, that run the test suite.
_TEST_RUNNERS = frozenset({"pytest", "py.test", "tox", "nox"})
#: An interpreter command: ``python``, ``python3.12``, the ``py`` launcher.
_PYTHON = re.compile(r"python(?:\d+(?:\.\d+)*)?|py")
#: The ``uv run`` options that take the next word as their value.
_UV_RUN_VALUE_OPTIONS = frozenset(
    {
        "--python",
        "-p",
        "--with",
        "--with-editable",
        "--with-requirements",
        "--group",
        "--only-group",
        "--no-group",
        "--extra",
        "--no-extra",
        "--package",
        "--directory",
        "--project",
        "--env-file",
        "--index",
        "--default-index",
        "--index-url",
        "--extra-index-url",
        "--find-links",
        "-f",
        "--config-file",
        "--cache-dir",
        "--color",
        "--python-preference",
    }
)


def _matrix_combinations(matrix: Any) -> list[dict[str, Any]]:
    """The job combinations of a ``strategy.matrix``, as GitHub forms them.

    The product of the list-valued keys, less every ``exclude`` match. Each
    ``include`` entry is then merged into every product combination whose
    original values it does not overwrite (a value an earlier entry added
    may be overwritten); an entry that fits none is added as a combination
    of its own, which later entries never match. With no list-valued keys
    there is no product, so every entry is its own combination. A matrix,
    ``include`` or ``exclude`` given by an expression yields nothing (it
    cannot be read statically, so it is no evidence).
    """
    if not isinstance(matrix, dict):
        return []
    base = {k: v for k, v in matrix.items() if k not in ("include", "exclude")}
    include = matrix.get("include") or []
    exclude = matrix.get("exclude") or []
    if not all(
        isinstance(values, list)
        for values in [*base.values(), include, exclude]
    ):
        return []
    originals: list[dict[str, Any]] = [{}] if base else []
    for key, values in base.items():
        originals = [
            {**combo, key: value} for combo in originals for value in values
        ]
    originals = [
        combo
        for combo in originals
        if not any(
            isinstance(excluded, dict)
            and all(combo.get(k) == v for k, v in excluded.items())
            for excluded in exclude
        )
    ]
    jobs = [dict(combo) for combo in originals]
    added: list[dict[str, Any]] = []
    for included in include:
        if not isinstance(included, dict):
            continue
        fits = [
            job
            for original, job in zip(originals, jobs, strict=True)
            if all(
                original[k] == v for k, v in included.items() if k in original
            )
        ]
        for job in fits:
            job.update(included)
        if not fits:
            added.append(dict(included))
    return jobs + added


def _holds(condition: Any, version: str) -> bool:
    """Whether a job's or step's ``if`` holds for ``version``.

    No condition holds; a literal ``true`` or ``false`` (bare or as an
    expression) is itself; any condition other than a comparison of
    ``matrix.python-version`` with a literal is not known to hold.
    """
    if condition is None:
        return True
    if isinstance(condition, bool):
        return condition
    literal = _LITERAL_CONDITION.fullmatch(str(condition))
    if literal is not None:
        return (literal.group(1) or literal.group(2)) == "true"
    match = _VERSION_CONDITION.fullmatch(str(condition))
    if match is None:
        return False
    equal = version == match.group(2)
    return equal if match.group(1) == "==" else not equal


def _shell_commands(script: str) -> list[list[str]]:
    """The simple commands of a ``run`` script, in order, as word lists.

    Each line is split as a POSIX shell splits words: quotes group, ``#``
    starts a comment, and newlines, ``;``, ``&``, ``&&``, ``|``, ``||`` and
    parentheses end a command. A command's leading keywords (``then``) and
    assignments (``CI=1``) are dropped, so its first word is its name. A
    line that does not split (an unclosed quote) yields no command.
    """

    def word(match: re.Match[str]) -> str:
        matrix = _MATRIX_VERSION.fullmatch(match.group(0))
        return _MATRIX_WORD if matrix else _OTHER_WORD

    script = _EXPRESSION.sub(word, script.replace("\\\n", " "))
    commands: list[list[str]] = []
    for line in script.splitlines():
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError:
            continue
        words: list[str] = []
        for token in [*tokens, ";"]:
            if token and set(token) <= _OPERATOR_CHARS:
                while words and (
                    words[0] in _COMMAND_PREFIXES
                    or _ASSIGNMENT.fullmatch(words[0])
                ):
                    words.pop(0)
                if words:
                    commands.append(words)
                words = []
            else:
                words.append(token)
    return commands


def _command_name(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def _action_selects_matrix(step: dict[str, Any]) -> bool:
    """Whether an action's ``python-version`` input is the matrix value.

    ``actions/setup-python`` (and ``astral-sh/setup-uv``) select the job's
    interpreter so.
    """
    inputs = step.get("with")
    if "uses" not in step or not isinstance(inputs, dict):
        return False
    value = str(inputs.get("python-version", "")).strip()
    return bool(_MATRIX_VERSION.fullmatch(value))


def _command_selects_matrix(words: list[str]) -> bool:
    """Whether a ``uv`` command selects the matrix's interpreter.

    It passes the matrix value as ``--python`` (or ``-p``), as
    ``uv sync --python ${{ matrix.python-version }}`` does.
    """
    if _command_name(words[0]) not in ("uv", "uvx"):
        return False
    for option, value in zip(words, [*words[1:], ""], strict=True):
        if option in ("--python", "-p") and value == _MATRIX_WORD:
            return True
        if option == f"--python={_MATRIX_WORD}":
            return True
    return False


def _python_module(arguments: list[str]) -> str | None:
    """The module an interpreter given ``arguments`` runs with ``-m``."""
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-m":
            return arguments[index + 1] if index + 1 < len(arguments) else None
        if argument.startswith("-m"):
            return argument[2:]
        if argument == "-c" or not argument.startswith("-"):
            return None  # code or a script runs instead
        index += 2 if argument in ("-W", "-X") else 1
    return None


def _runs_tests(words: list[str]) -> bool:
    """Whether a simple command runs the test suite.

    Its name is ``pytest``, ``tox`` or ``nox``; or it is an interpreter
    running one of them with ``-m``; or it is ``uv run`` of such a command
    (or ``-m`` module). The name elsewhere in a command, as an argument of
    ``pip install`` or ``echo``, runs nothing.
    """
    name = _command_name(words[0])
    if name in _TEST_RUNNERS:
        return True
    if _PYTHON.fullmatch(name):
        return _python_module(words[1:]) in _TEST_RUNNERS
    if name != "uv" or words[1:2] != ["run"]:
        return False
    rest = words[2:]
    index = 0
    while index < len(rest) and rest[index].startswith("-"):
        option = rest[index]
        if option == "--":
            index += 1
            break
        if option in ("-m", "--module"):
            return index + 1 < len(rest) and rest[index + 1] in _TEST_RUNNERS
        index += 2 if option in _UV_RUN_VALUE_OPTIONS else 1
    return index < len(rest) and _runs_tests(rest[index:])


def _runs_on_the_matrix_interpreter(
    steps: list[dict[str, Any]], version: str
) -> bool:
    """Whether ``steps`` select the matrix interpreter, then run the suite.

    Only steps whose ``if`` holds for ``version`` count. The selection is an
    action's ``python-version`` input or a ``uv`` command's ``--python``
    option of ``${{ matrix.python-version }}``; mentioning the value
    elsewhere (``echo``) selects nothing. The suite runs from a ``run``
    command at or after the selection (``_runs_tests``).
    """
    selected = False
    for step in steps:
        if not _holds(step.get("if"), version):
            continue
        selected = selected or _action_selects_matrix(step)
        run = step.get("run")
        for words in _shell_commands(run) if isinstance(run, str) else []:
            selected = selected or _command_selects_matrix(words)
            if selected and _runs_tests(words):
                return True
    return False


def _job_test_versions(job: dict[str, Any]) -> set[str]:
    """The matrix versions for which a job runs the test suite.

    A combination counts when the job's ``if`` holds for it and its enabled
    steps select the matrix interpreter, then run the suite
    (``_runs_on_the_matrix_interpreter``).
    """
    matrix = (job.get("strategy") or {}).get("matrix")
    steps = [step for step in job.get("steps") or [] if isinstance(step, dict)]
    versions: set[str] = set()
    for combo in _matrix_combinations(matrix):
        if "python-version" not in combo:
            continue
        version = str(combo["python-version"])
        if not _holds(job.get("if"), version):
            continue
        if _runs_on_the_matrix_interpreter(steps, version):
            versions.add(version)
    return versions


def _ci_versions(workflows: list[str] | None = None) -> set[str]:
    """The interpreter versions a CI workflow runs the test suite on.

    ``workflows`` are workflow texts; by default the repository's. Each is
    parsed as YAML (comments are not configuration) and read job by job.
    """
    import yaml

    if workflows is None:
        workflows = [
            path.read_text(encoding="utf-8")
            for path in sorted(WORKFLOWS.glob("*.y*ml"))
        ]
    versions: set[str] = set()
    for text in workflows:
        document = yaml.safe_load(text) or {}
        for job in (document.get("jobs") or {}).values():
            if isinstance(job, dict):
                versions |= _job_test_versions(job)
    return versions


CI_SELF_TEST = """
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        # python-version: ["3.15"]  (a comment is not a matrix)
        python-version:
          - "3.11"
          - "3.12"
          - "3.13"
        os: [ubuntu-latest, macos-latest]
        exclude:
          - python-version: "3.13"
        include:
          - python-version: "3.14"
            os: ubuntu-latest
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: python -m pytest -q
  lint:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10"]
    steps:
      - uses: actions/setup-python@v5
        with: {python-version: "${{ matrix.python-version }}"}
      - run: ruff check .
  gated:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.8"]
    steps:
      - run: uv sync --python ${{ matrix.python-version }}
      - if: matrix.python-version == '3.9'
        run: uv run pytest -q
  unwired:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.7"]
    steps:
      - run: pytest -q
"""


# @spec ING-EP-022
def test_ci_evidence_is_the_parsed_matrix_of_a_job_that_runs_the_tests():
    # 3.15 is only in a comment, 3.13 is excluded, 3.14 comes from include,
    # 3.10 runs no tests, 3.8 skips its test step and 3.7's interpreter is
    # never set from the matrix.
    assert _ci_versions([CI_SELF_TEST]) == {"3.11", "3.12", "3.14", "3.9"}
    assert _ci_versions(['jobs: {}\n# python-version: ["3.12"]\n']) == set()


SETUP_PYTHON = {
    "uses": "actions/setup-python@v5",
    "with": {"python-version": "${{ matrix.python-version }}"},
}


def _workflow(
    *steps: dict[str, Any], matrix: dict[str, Any] | None = None
) -> str:
    """A workflow of one job over ``matrix`` (by default 3.12)."""
    import yaml

    job = {
        "runs-on": "ubuntu-latest",
        "strategy": {"matrix": matrix or {"python-version": ["3.12"]}},
        "steps": list(steps),
    }
    return yaml.safe_dump({"jobs": {"job": job}})


#: ``run`` scripts with a test command in command position.
RUNS_THE_SUITE = {
    "bare": "pytest -q",
    "after &&": "cd tests && pytest -q",
    "after ;": "set -e; pytest",
    "after |": "true | pytest -q",
    "after then": "if [ -d tests ]; then pytest -q; fi",
    "after an assignment": "CI=1 pytest",
    "on a later line": "pip install pytest\npytest -q",
    "python -m": "python -m pytest -q",
    "uv run": "uv run pytest -q",
    "uv run python -m": "uv run --frozen python -m pytest",
    "uv run with a value option": "uv run --with pytest-cov pytest",
    "tox": "tox -e py",
    "nox": "nox -s tests",
}

#: Jobs that do not run the suite on the matrix's interpreter.
NOT_EVIDENCE: dict[str, list[dict[str, Any]]] = {
    "setup if: false": [{**SETUP_PYTHON, "if": False}, {"run": "pytest -q"}],
    "setup if: ${{ false }}": [
        {**SETUP_PYTHON, "if": "${{ false }}"},
        {"run": "pytest -q"},
    ],
    "echo selects no interpreter": [
        {"run": "echo ${{ matrix.python-version }}"},
        {"run": "pytest -q"},
    ],
    "pytest only echoed": [
        SETUP_PYTHON,
        {"run": 'echo "pytest on ${{ matrix.python-version }}"'},
    ],
    "pytest in a comment": [
        SETUP_PYTHON,
        {"run": "# pytest -q\nruff check .  # && pytest"},
    ],
    "pytest installed, not run": [
        SETUP_PYTHON,
        {"run": "pip install ruff pytest\nruff check ."},
    ],
    "pytest a uv run dependency": [
        SETUP_PYTHON,
        {"run": "uv run --with pytest ruff check ."},
    ],
}


# @spec ING-EP-022
@pytest.mark.parametrize("case", sorted(RUNS_THE_SUITE))
def test_ci_evidence_counts_a_test_command_in_command_position(case) -> None:
    run = {"run": RUNS_THE_SUITE[case]}
    assert _ci_versions([_workflow(SETUP_PYTHON, run)]) == {"3.12"}
    selected_by_uv = {"run": "uv sync --python ${{ matrix.python-version }}"}
    assert _ci_versions([_workflow(selected_by_uv, run)]) == {"3.12"}


# @spec ING-EP-022
@pytest.mark.parametrize("case", sorted(NOT_EVIDENCE))
def test_ci_evidence_needs_an_enabled_selection_and_a_test_command(
    case,
) -> None:
    assert _ci_versions([_workflow(*NOT_EVIDENCE[case])]) == set()


# @spec ING-EP-022
def test_ci_matrix_include_follows_github_semantics() -> None:
    include_only = {
        "include": [{"python-version": "3.11"}, {"python-version": "3.12"}]
    }
    assert _matrix_combinations(include_only) == include_only["include"]
    run = {"run": "pytest"}
    workflow = _workflow(SETUP_PYTHON, run, matrix=include_only)
    assert _ci_versions([workflow]) == {"3.11", "3.12"}
    # A partial entry after a standalone one: never matched against the
    # standalone job, so its missing ``os`` is never read.
    partial = {
        "os": ["linux", "mac"],
        "python-version": ["3.11"],
        "include": [
            {"python-version": "3.14"},
            {"os": "linux", "extra": True},
        ],
    }
    assert _matrix_combinations(partial) == [
        {"os": "linux", "python-version": "3.11", "extra": True},
        {"os": "mac", "python-version": "3.11"},
        {"python-version": "3.14"},
    ]
    # Exclude precedes include; an entry never overwrites an original value
    # but may overwrite an added one; a standalone job is never merged into.
    matrix = {
        "os": ["linux", "mac"],
        "python-version": ["3.11", "3.12"],
        "exclude": [{"os": "mac", "python-version": "3.11"}],
        "include": [
            {"tier": "a"},
            {"os": "linux", "tier": "b"},
            {"python-version": "3.12", "tag": "x"},
            {"python-version": "3.14"},
            {"python-version": "3.14", "os": "linux"},
        ],
    }
    assert _matrix_combinations(matrix) == [
        {"os": "linux", "python-version": "3.11", "tier": "b"},
        {"os": "linux", "python-version": "3.12", "tier": "b", "tag": "x"},
        {"os": "mac", "python-version": "3.12", "tier": "a", "tag": "x"},
        {"python-version": "3.14"},
        {"python-version": "3.14", "os": "linux"},
    ]


def _readme_support_table() -> dict[str, str]:
    """``Python`` to ``Status`` from the README's support table."""
    lines = README.read_text(encoding="utf-8").splitlines()
    heading = [
        i
        for i, line in enumerate(lines)
        if re.fullmatch(r"#+\s+Supported Python versions\s*", line)
    ]
    assert len(heading) == 1, "one 'Supported Python versions' section"
    rows: dict[str, str] = {}
    for line in lines[heading[0] + 1 :]:
        if line.startswith("#"):
            break
        cells = [
            cell.strip(" `") for cell in line.strip().strip("|").split("|")
        ]
        if len(cells) == 2 and re.fullmatch(r"3\.\d+", cells[0]):
            assert cells[0] not in rows, f"{cells[0]} listed twice"
            rows[cells[0]] = cells[1]
    return rows


# @spec ING-EP-022
def test_unexecuted_versions_are_recorded_as_declared_but_unverified() -> None:
    declared = _declared_versions()
    executed = _ci_versions()
    table = _readme_support_table()
    assert list(table) == declared, (
        "the README's support table lists every declared version once, "
        f"in order: {declared}"
    )
    wrong = {
        version: status
        for version, status in table.items()
        if status
        != ("supported" if version in executed else "declared-but-unverified")
    }
    assert not wrong, (
        f"a version is 'supported' only when a CI gate runs it {executed}: "
        f"{wrong}"
    )


# ------------------------------------------------ structural boundaries


FORBIDDEN_ENGINE_NAMES = (
    "AsyncLLM",
    "AsyncLLMEngine",
    "LLMEngine",
    "EngineCoreClient",
    "MPClient",
    "build_async_engine_client",
    "collective_rpc",
)
FORBIDDEN_ENGINE_MODULES = (
    "vllm.v1.engine.async_llm",
    "vllm.v1.engine.llm_engine",
    "vllm.v1.engine.core_client",
    "vllm.v1.engine.core",
    "vllm.engine.arg_utils",
    "vllm.entrypoints.launchers",
)


def _imported_modules(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _names(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.alias):
            found.add(node.name.rsplit(".", 1)[-1])
    return found


# @spec ING-EP-020
def test_frontend_creates_no_engine_and_opens_no_engine_path() -> None:
    offenders: list[str] = []
    for name, tree in _trees().items():
        for module in _imported_modules(tree):
            if module.startswith(FORBIDDEN_ENGINE_MODULES):
                offenders.append(f"{name}: imports {module}")
        for used in _names(tree) & set(FORBIDDEN_ENGINE_NAMES):
            offenders.append(f"{name}: uses {used}")
    assert not offenders, "\n".join(offenders)


# @spec ING-VEH-005
def test_frontend_defines_no_batching_mechanism() -> None:
    """No class, function or method of the frontend is a batching mechanism.

    The check reads definitions in the syntax tree; comments and strings
    (a docstring saying the engine batches) are not definitions.
    """
    offenders: list[str] = []
    for name, tree in _trees().items():
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ) and re.search("batch", node.name, re.I):
                offenders.append(f"{name}:{node.lineno}: {node.name}")
    assert not offenders, "\n".join(offenders)


# @spec ING-VEH-008
def test_frontend_renders_no_prompt_and_drives_no_generation() -> None:
    forbidden = {
        "renderer",
        "input_processor",
        "apply_chat_template",
        "get_tokenizer",
        "SamplingParams",
        "generate",
        "encode_prompt",
    }
    offenders: list[str] = []
    for name, tree in _trees().items():
        for used in sorted(_names(tree) & forbidden):
            offenders.append(f"{name}: uses {used}")
    assert not offenders, "\n".join(offenders)


# @spec ING-VEH-002
def test_servicer_is_registered_through_the_generated_aio_call() -> None:
    trees = _trees()
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for tree in trees.values()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "add_RivaSpeechRecognitionServicer_to_server" in calls
    offenders: list[str] = []
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "server"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "grpc"
            ):
                offenders.append(f"{name}:{node.lineno}: grpc.server (sync)")
    assert not offenders, "\n".join(offenders)


SESSIONS = 8


# @spec ING-VEH-002
async def test_concurrent_streams_commit_no_thread_per_session(
    started, fake_factory, clock, helpers
) -> None:
    """Open streams do not each hold an OS thread for their duration.

    A thread the process uses for shared work is allowed; one that each
    open stream holds shows up as growth with the number of open streams.
    """
    plugin, _ = started
    calls = [(await helpers.stream_with_lease(plugin, fake_factory, clock))[0]]
    await clock.settle()
    baseline = threading.active_count()
    for _ in range(SESSIONS):
        call, _ = await helpers.stream_with_lease(plugin, fake_factory, clock)
        calls.append(call)
    await clock.settle()
    grown = threading.active_count() - baseline
    assert grown < SESSIONS, (
        f"{grown} more threads while {SESSIONS} more streams are open"
    )
    for call in calls:
        call.half_close()
    for call in calls:
        await call.wait()


# @spec ING-VEH-007
def test_no_per_session_value_is_stored_on_the_shared_factory() -> None:
    """No module stores a per-session value on the shared factory."""
    offenders: list[str] = []
    for name, tree in _trees().items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "factory"
                for target in node.targets
            ):
                offenders.append(
                    f"{name}:{node.lineno}: assigns onto the shared factory"
                )
    assert not offenders, "\n".join(offenders)


# ------------------------------------------------------- where vLLM is imported

#: The lifecycle methods core calls on the plugin (ING-EP-108).
LIFECYCLE_METHODS = frozenset(
    {"attach_router", "init_state", "start", "ready", "drain", "stop"}
)
#: The class the lifecycle methods belong to.
PLUGIN_CLASS = "RivaFrontendPlugin"
#: The path-scoped authentication middleware: a class named for both.
AUTH_MIDDLEWARE = re.compile(r"Auth\w*Middleware$")
#: The complete-file path: the module of the frontend's file route and of
#: ``Recognize``'s call to core's transcription service, and the
#: ``Recognize`` servicer method itself.
COMPLETE_FILE_MODULES = frozenset({"files.py"})
COMPLETE_FILE_METHODS = frozenset({"Recognize"})
_DYNAMIC_IMPORTS = frozenset({"import_module", "__import__"})


def _is_vllm(module: str | None) -> bool:
    return module is not None and (
        module == "vllm" or module.startswith("vllm.")
    )


def _vllm_import(node: ast.AST) -> str | None:
    """The vLLM module ``node`` imports, or ``None``.

    Covers ``import vllm...``, ``from vllm... import ...`` and a call of
    ``importlib.import_module`` or ``__import__`` with a literal name.
    """
    if isinstance(node, ast.Import):
        for alias in node.names:
            if _is_vllm(alias.name):
                return alias.name
    elif isinstance(node, ast.ImportFrom):
        if node.level == 0 and _is_vllm(node.module):
            return node.module
    elif isinstance(node, ast.Call):
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if (
            name in _DYNAMIC_IMPORTS
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and _is_vllm(node.args[0].value)
        ):
            return node.args[0].value
    return None


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _allowed_scope(module: str, scope: list[tuple[str, str]]) -> bool:
    if module in COMPLETE_FILE_MODULES:
        return True
    for index, (kind, name) in enumerate(scope):
        if kind == "class" and AUTH_MIDDLEWARE.search(name):
            return True
        if kind == "def" and name in COMPLETE_FILE_METHODS:
            return True
        if (
            kind == "class"
            and name == PLUGIN_CLASS
            and index + 1 < len(scope)
            and scope[index + 1][0] == "def"
            and scope[index + 1][1] in LIFECYCLE_METHODS
        ):
            return True
    return False


def _misplaced_vllm_imports(module: str, tree: ast.Module) -> list[str]:
    """Every vLLM import of ``tree`` outside the places ING-EP-108 allows.

    An import under ``if TYPE_CHECKING:`` does not run and is not counted.
    Each offender is ``module:line: imports X in <scope>``.
    """
    offenders: list[str] = []

    def visit(node: ast.AST, scope: list[tuple[str, str]]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking(child.test):
                for other in child.orelse:
                    visit_one(other, scope)
                continue
            visit_one(child, scope)

    def visit_one(child: ast.AST, scope: list[tuple[str, str]]) -> None:
        inner = scope
        if isinstance(child, ast.ClassDef):
            inner = [*scope, ("class", child.name)]
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inner = [*scope, ("def", child.name)]
        imported = _vllm_import(child)
        if imported is not None and not _allowed_scope(module, scope):
            where = ".".join(name for _, name in scope) or "module level"
            offenders.append(
                f"{module}:{child.lineno}: imports {imported} in {where}"
            )
        visit(child, inner)

    visit(tree, [])
    return offenders


# @spec ING-EP-108
def test_vllm_is_imported_only_inside_lifecycle_auth_and_file_path() -> None:
    """Imports of vLLM sit only where ING-EP-108 allows; report the rest.

    The allowed places are the lifecycle methods, the authentication
    middleware and the complete-file path.
    """
    offenders: list[str] = []
    for name, tree in _trees().items():
        offenders.extend(_misplaced_vllm_imports(name, tree))
    assert not offenders, "\n".join(offenders)


_IMPORT_LAYOUT = """
import importlib
from typing import TYPE_CHECKING

import vllm.envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def helper():
    from vllm.entrypoints import utils
    importlib.import_module("vllm.plugins")


class RivaFrontendPlugin:
    async def init_state(self, engine_client, state, args):
        from vllm.entrypoints.serve.middleware import authenticate

        def nested():
            import vllm.envs

    def other(self):
        import vllm


class RivaAuthMiddleware:
    async def __call__(self, scope, receive, send):
        from vllm.entrypoints.serve.middleware import authenticate


class Servicer:
    async def Recognize(self, request, context):
        from vllm.entrypoints.speech_to_text import base
"""


# @spec ING-EP-108
def test_the_vllm_import_check_reports_each_misplaced_import() -> None:
    offenders = _misplaced_vllm_imports("layout.py", ast.parse(_IMPORT_LAYOUT))
    assert [o.split(": ", 1)[1] for o in offenders] == [
        "imports vllm.envs in module level",
        "imports vllm.entrypoints in helper",
        "imports vllm.plugins in helper",
        "imports vllm in RivaFrontendPlugin.other",
    ], offenders
    assert _misplaced_vllm_imports("files.py", ast.parse(_IMPORT_LAYOUT)) == []


# ------------------------------------------------- which tests need RFC 1

TESTS_ROOT = PROJECT_ROOT / "tests"
#: The one test that needs RFC 1 (ING-EP-071).
DEADLINE_PIN = (
    "vehicle/test_deadlines_capacity.py",
    "test_copied_defaults_are_pinned_against_the_core_build",
)


def _marker_names(node: ast.AST) -> set[str]:
    """``X`` of every ``pytest.mark.X`` (or ``mark.X``) under ``node``."""
    found: set[str] = set()
    for inner in ast.walk(node):
        if (
            isinstance(inner, ast.Attribute)
            and isinstance(inner.value, ast.Attribute)
            and inner.value.attr == "mark"
        ) or (
            isinstance(inner, ast.Attribute)
            and isinstance(inner.value, ast.Name)
            and inner.value.id == "mark"
        ):
            found.add(inner.attr)
    return found


def _engine_marked(source: str) -> list[tuple[str, bool]]:
    """``(owner, also_gpu)`` for each place ``source`` applies ``engine``.

    The owner is ``<module>`` for a module ``pytestmark``, the class or
    test name for a class ``pytestmark`` or a decorator, and the enclosing
    test (or ``line N``) for a mark applied anywhere else (a
    ``pytest.param(..., marks=...)`` or ``add_marker``).
    """
    tree = ast.parse(source)
    found: list[tuple[str, bool]] = []
    claimed: set[int] = set()

    def note(owner: str, nodes: list[ast.AST]) -> None:
        names: set[str] = set()
        for node in nodes:
            names |= _marker_names(node)
            claimed.update(id(inner) for inner in ast.walk(node))
        if "engine" in names:
            found.append((owner, "gpu" in names))

    def pytestmark(body: list[ast.stmt]) -> list[ast.AST]:
        return [
            stmt.value
            for stmt in body
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "pytestmark"
                for t in stmt.targets
            )
        ]

    note("<module>", pytestmark(tree.body))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            note(node.name, [*node.decorator_list, *pytestmark(node.body)])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            note(node.name, list(node.decorator_list))
    for node in ast.walk(tree):
        if (
            id(node) not in claimed
            and isinstance(node, ast.Attribute)
            and node.attr == "engine"
            and "engine" in _marker_names(node)
        ):
            found.append((f"line {node.lineno}", False))
    return found


# @spec ING-EP-072
def test_only_the_deadline_pin_needs_rfc1() -> None:
    """Every CPU test runs on an RFC 2 build without RFC 1.

    ``engine`` (RFC 1 + RFC 2) may mark only the stream-deadline pin of
    ING-EP-071; a test also marked ``gpu`` is not a CPU test.
    """
    offenders: list[str] = []
    pinned = False
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        relative = path.relative_to(TESTS_ROOT).as_posix()
        for owner, gpu in _engine_marked(path.read_text(encoding="utf-8")):
            if (relative, owner) == DEADLINE_PIN:
                pinned = True
            elif not gpu:
                offenders.append(f"{relative}: {owner} is marked engine")
    assert pinned, "the stream-deadline pin runs only on RFC 1 + RFC 2"
    assert not offenders, "\n".join(offenders)


_MARKED = """
import pytest
from pytest import mark

pytestmark = pytest.mark.engine


class TestGroup:
    pytestmark = [pytest.mark.engine]


@pytest.mark.engine
def test_decorated():
    pass


@mark.engine
@mark.gpu
def test_gpu():
    pass


@pytest.mark.parametrize("x", [pytest.param(1, marks=pytest.mark.engine)])
def test_param(x):
    pass


def test_unmarked(request):
    request.node.add_marker(pytest.mark.engine)
"""


# @spec ING-EP-072
def test_the_engine_mark_scan_finds_every_form() -> None:
    assert sorted(_engine_marked(_MARKED)) == sorted(
        [
            ("<module>", False),
            ("TestGroup", False),
            ("test_decorated", False),
            ("test_gpu", True),
            ("test_param", False),
            (f"line {_line_of(_MARKED, 'add_marker')}", False),
        ]
    )


def _line_of(source: str, needle: str) -> int:
    return next(
        number
        for number, line in enumerate(source.splitlines(), start=1)
        if needle in line
    )

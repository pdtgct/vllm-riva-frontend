"""Shared fixtures and test tiers.

* unmarked: CPU tests over the fakes in ``tests/support``, against the vLLM
  core tree on ``PYTHONPATH``. They need RFC 2 only (ING-EP-072) and fail,
  not skip, when it is missing.
* ``engine``: CPU tests that need RFC 1 as well (the RFC 1 + RFC 2 build);
  only the stream-deadline pin (ING-EP-071) belongs here. Skipped when RFC
  1's stream-admission settings do not import.
* ``real_factory``: the seam conformance suite against the model plugin's
  real factory. Skipped when the model plugin is not installed.
* ``gpu``: an installed engine on a CUDA device. Skipped without one.
"""

from __future__ import annotations

import importlib
import signal
import sys
import threading
from pathlib import Path

import pytest

_TESTS = Path(__file__).parent
sys.path.insert(0, str(_TESTS.parents[0] / "src"))
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


#: RFC 2 is required by every tier; ``engine`` additionally needs RFC 1.
RFC2 = _importable("vllm.plugins.endpoint_plugins.interface")
ENGINE = RFC2 and _importable("vllm.v1.core.stream_admission")
REAL_FACTORY = _importable("vllm_nemotron_asr_streaming.session")

#: Real seconds the ``started`` fixture's teardown gives the plugin's stop
#: before it fails the test as a settlement hang (generous: a conforming
#: stop under the fakes ends in well under a second).
STOP_BOUND = 30.0


#: Real seconds a CPU test (setup, call and teardown together) may run
#: before it fails as a hang. The plugin's settlement has no time limit by
#: design (ING-EP-093), so a settlement that never ends would otherwise
#: stall the run instead of failing the test. An expired test leaves tasks
#: behind that would corrupt later tests, so expiry also stops the session.
#: GPU and real-factory tests are not bounded.
TEST_REAL_TIME_BOUND = 120


class _RealTimeBoundExceeded(BaseException):
    """Raised by the alarm; a BaseException so no handler swallows it."""


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    bounded = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
        and "gpu" not in item.keywords
        and "real_factory" not in item.keywords
    )
    if not bounded:
        yield
        return

    def _expire(signum: int, frame: object) -> None:
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        item.session.shouldstop = (
            f"{item.nodeid} exceeded the real-time bound; later results "
            "would be unreliable"
        )
        raise _RealTimeBoundExceeded(
            f"{item.nodeid} ran past {TEST_REAL_TIME_BOUND} s of real time "
            "(a settlement or wait that never ends)"
        )

    previous = signal.signal(signal.SIGALRM, _expire)
    signal.alarm(TEST_REAL_TIME_BOUND)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "engine: imports the RFC 1 + RFC 2 vLLM tree"
    )
    config.addinivalue_line(
        "markers",
        "real_factory: seam conformance against the model plugin's factory",
    )
    config.addinivalue_line("markers", "gpu: needs an installed engine on CUDA")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    cuda: bool | None = None
    for item in items:
        if "engine" in item.keywords and not ENGINE:
            item.add_marker(
                pytest.mark.skip(reason="needs the RFC 1 + RFC 2 vLLM build")
            )
        if "real_factory" in item.keywords and not REAL_FACTORY:
            item.add_marker(
                pytest.mark.skip(reason="model plugin not installed")
            )
        if "gpu" in item.keywords:
            if cuda is None:
                cuda = _cuda_available()
            if not cuda:
                item.add_marker(pytest.mark.skip(reason="no CUDA device"))


# -------------------------------------------------------------- fixtures


@pytest.fixture
def clock():
    """A ``FakeClock`` starting at 1000 s of virtual time."""
    from support.clock import FakeClock

    return FakeClock()


@pytest.fixture
async def fake_factory(clock):
    """A ``FakeSessionFactory`` with the default cadences and locales.

    It serves ``560ms`` and ``1120ms`` with locales ``en-US``, ``de-DE``
    and ``auto`` (default ``auto``), on ``clock``, the plugin's clock, so its
    ``drain(deadline)`` waits in the same time the plugin's deadlines use.
    It models generation, cleanup and holding tasks (``support.fake_session``);
    whatever is still running at the end of the test is cancelled.
    """
    from support.fake_session import FakeSessionFactory

    factory = FakeSessionFactory(clock=clock)
    try:
        yield factory
    finally:
        await factory.aclose()


@pytest.fixture
def fake_seam(fake_factory, clock):
    """The seam conformance hooks (``support.seam.FakeSeam``) over it."""
    from support.seam import FakeSeam

    return FakeSeam(fake_factory, clock)


@pytest.fixture
def riva_options():
    """The ``riva_frontend`` options a test adds to the defaults.

    ``fake_engine`` adds ``grpc_bind`` on a free loopback port; a module or
    test overrides this fixture (or parametrizes it) to set others.
    """
    return {}


@pytest.fixture
def fake_engine(riva_options):
    """A ``FakeEngineClient`` with both speech tasks and ``riva_options``.

    It reports 17 pages (16 slots) and 16 active requests; ``grpc_bind`` is a
    free loopback port, since the design refuses port 0.
    """
    from support.fake_engine import FakeEngineClient
    from support.grpc_harness import free_port

    options = {"grpc_bind": f"127.0.0.1:{free_port()}", **riva_options}
    return FakeEngineClient(riva_frontend=options)


@pytest.fixture
def fake_models(fake_engine):
    from support.fake_transcription import FakeServingModels

    return FakeServingModels(fake_engine.model_config.served_model_name)


@pytest.fixture
def fake_transcription(fake_models):
    from support.fake_transcription import FakeTranscriptionServing

    return FakeTranscriptionServing(fake_models)


@pytest.fixture
def core_state(fake_engine, fake_models, fake_transcription):
    """A bare core ``State`` with no app.

    It is what ``init_app_state`` leaves before ``init_endpoint_plugins_state``
    runs.
    """
    from support.state import make_state

    return make_state(
        fake_engine, models=fake_models, transcription=fake_transcription
    )


@pytest.fixture
def entry_points(fake_factory):
    """The session-factory discovery seam, publishing ``fake_factory``."""
    from support.fake_session import fake_entry_points

    return fake_entry_points(fake=fake_factory.constructor)


@pytest.fixture
def plugin(clock, entry_points):
    """The frontend plugin under test, on ``clock`` and ``entry_points``."""
    from vllm_riva_frontend.lifecycle import RivaFrontendPlugin

    return RivaFrontendPlugin(clock=clock, entry_points=entry_points)


@pytest.fixture
def owner_parts(clock, fake_factory):
    """The constructor arguments of a directly built ``StreamOwner``.

    The layout's arguments over ``fake_factory`` and ``clock``, plus
    ``kind`` (the ``OwnerKind`` ``make_owner`` registers).
    """
    from vllm_riva_frontend.admission import OwnerKind
    from vllm_riva_frontend.config import parse_config
    from vllm_riva_frontend.errors import catalog
    from vllm_riva_frontend.lifecycle import OwnerRegistry

    registry = OwnerRegistry()
    return {
        "lease_opener": fake_factory.open,
        "surface": "riva",
        "config": parse_config({"riva_frontend": {}}),
        "clock": clock,
        "registry": registry,
        "catalog": catalog(),
        "kind": OwnerKind.stream,
    }


@pytest.fixture
def make_owner(owner_parts):
    """Build a ``StreamOwner`` registered for the current task."""
    import asyncio

    def make(**overrides):
        from vllm_riva_frontend.lease import StreamOwner

        parts = dict(owner_parts)
        kind = parts.pop("kind")
        registry = parts["registry"]
        record = registry.register(kind, asyncio.current_task())
        parts.update(overrides)
        return StreamOwner(record=record, **parts)

    return make


@pytest.fixture
def plugins():
    """The endpoint plugins ``test_app`` attaches.

    A leaf that serves the frontend overrides this with ``[plugin]``; the
    default attaches none.
    """
    return []


@pytest.fixture
def supported_tasks():
    return ("transcription", "realtime")


@pytest.fixture
def test_app(plugins, supported_tasks):
    """A core-like FastAPI app with ``plugins`` attached (``support.app``)."""
    from support.app import build_app

    return build_app(plugins, supported_tasks=supported_tasks)


@pytest.fixture
async def app_state(test_app, fake_engine, fake_models, fake_transcription):
    """``test_app.state`` after core's state init and Phase B."""
    from support.app import init_state

    return await init_state(
        test_app,
        fake_engine,
        models=fake_models,
        transcription=fake_transcription,
    )


@pytest.fixture
async def started(app_state, plugin):
    """``plugin`` after core's state init and ``start``; stopped after.

    The teardown never masks the test's own result, except that a stop
    which does not finish within ``STOP_BOUND`` seconds of real time fails
    the test: a settlement that hangs (a scripted call left blocked, a
    factory task that never ends) is a defect to report, not a stalled run.
    """
    import asyncio
    import contextlib

    from support.app import start_plugins, stop_plugins

    await start_plugins(app_state)
    try:
        yield plugin
    finally:
        stop = asyncio.ensure_future(stop_plugins(app_state))
        done, _ = await asyncio.wait({stop}, timeout=STOP_BOUND)
        if not done:
            stop.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait({stop}, timeout=STOP_BOUND)
            pytest.fail(
                f"plugin stop did not finish within {STOP_BOUND} s of real"
                " time: settlement hung",
                pytrace=False,
            )
        with contextlib.suppress(Exception):
            stop.result()

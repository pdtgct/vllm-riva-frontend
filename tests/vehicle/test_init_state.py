"""``init_state``: check order, identity, factory last, no side effects."""

from __future__ import annotations

import asyncio
import logging
import math
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import APIRouter
from support.app import PLUGIN_PREFIX


@pytest.fixture(autouse=True)
def _no_ambient_api_key(monkeypatch) -> None:
    """``VLLM_API_KEY`` from the shell must not configure a key here."""
    monkeypatch.delenv("VLLM_API_KEY", raising=False)


# --------------------------------------------------------------- order


async def _expect_failure(helpers, plugin, engine, **kw) -> list[str]:
    from vllm_riva_frontend.lifecycle import StartupError

    with pytest.raises(StartupError) as info:
        await helpers.initialize(plugin, engine, **kw)
    return list(info.value.failures)


async def _init(
    clock: Any,
    constructor: Any,
    *,
    options: dict[str, Any] | None = None,
    args: Any = None,
    models: Any = None,
    auth_registrations: tuple[dict[str, Any], ...] = (),
    extra_plugins: tuple[Any, ...] = (),
    **engine_kw: Any,
) -> Any:
    """Attach, fill the state and run ``init_state`` with these inputs.

    Returns the plugin; ``init_state``'s exception propagates.
    """
    from support.app import build_app
    from support.fake_session import fake_entry_points
    from support.state import make_args

    from vllm_riva_frontend.lifecycle import RivaFrontendPlugin

    from .conftest import engine_with, prepare_state, run_init

    plugin = RivaFrontendPlugin(
        clock=clock, entry_points=fake_entry_points(nemotron=constructor)
    )
    engine = engine_with(options=options, **engine_kw)
    app = build_app(
        [plugin, *extra_plugins],
        args=args if args is not None else make_args(),
        auth_registrations=auth_registrations,
    )
    await run_init(app, engine, prepare_state(app, engine, models=models))
    return plugin


async def _init_failures(clock: Any, constructor: Any, **kw: Any) -> list[str]:
    from vllm_riva_frontend.lifecycle import StartupError

    with pytest.raises(StartupError) as info:
        await _init(clock, constructor, **kw)
    return list(info.value.failures)


def _args(**overrides: Any) -> Any:
    from support.state import make_args

    return make_args(**overrides)


#: Signals the authentication decision compares (ING-EP-086).
_AUTH_WORDS = ("api_key", "api-key", "authenticat", "AuthenticationMiddleware")
#: A registration of core's authentication with no configured key: the
#: authentication step fails (ING-EP-086).
_STRAY_REGISTRATION = ({"tokens": ["registered-only"]},)


def _mentions_auth(failures: list[str]) -> bool:
    return any(
        word.lower() in text.lower()
        for text in failures
        for word in _AUTH_WORDS
    )


#: One case per step of ING-EP-068: the inputs make that step and the next
#: one fail; ``first`` must be named, ``later`` must not be, and the session
#: factory (the last step) is never constructed.
ORDER_CASES = {
    "configuration-before-identity": dict(
        inputs=dict(
            options={"surprise": 1, "another": 2},
            models=SimpleNamespace(model_name=lambda: ""),
        ),
        first=("surprise", "another"),
        later=(),
    ),
    "identity-before-deadlines": dict(
        inputs=dict(
            models=SimpleNamespace(model_name=lambda: ""),
            stream_admission={"never_fed_timeout": -1},
        ),
        first=(),
        later=("never_fed_timeout",),
    ),
    "deadlines-before-capacity": dict(
        inputs=dict(
            stream_admission={"draining_timeout": 0}, num_gpu_blocks=None
        ),
        first=("draining_timeout",),
        later=("num_gpu_blocks",),
    ),
    "capacity-before-complete-file-service": dict(
        inputs=dict(num_gpu_blocks=None, options={"max_concurrent_files": 50}),
        first=("num_gpu_blocks",),
        later=("max_concurrent_files",),
    ),
    "complete-file-service-before-api-server-count": dict(
        inputs=dict(
            options={"max_concurrent_files": 50},
            args=_args(api_server_count=2),
        ),
        first=("max_concurrent_files",),
        later=("api_server_count",),
    ),
    "api-server-count-before-authentication": dict(
        inputs=dict(
            args=_args(api_server_count=2),
            auth_registrations=_STRAY_REGISTRATION,
        ),
        first=("api_server_count",),
        later=("<authentication>",),
    ),
    "authentication-before-route-table": dict(
        inputs=dict(
            auth_registrations=_STRAY_REGISTRATION,
            route_override=f"{PLUGIN_PREFIX}/metadata",
        ),
        first=("<authentication>",),
        later=(f"{PLUGIN_PREFIX}/metadata",),
    ),
    "route-table-before-session-factory": dict(
        inputs=dict(route_override=f"{PLUGIN_PREFIX}/version"),
        first=(f"{PLUGIN_PREFIX}/version",),
        later=(),
    ),
}


# @spec ING-EP-068
@pytest.mark.parametrize("case", list(ORDER_CASES))
async def test_check_order_reports_the_first_failing_step(
    clock, fake_factory, helpers, case
) -> None:
    from support.fake_transcription import FakeServingModels

    spec = ORDER_CASES[case]
    inputs = dict(spec["inputs"])
    override = inputs.pop("route_override", None)
    if override is not None:
        inputs["extra_plugins"] = (_Overrider(override),)
    counted = FakeServingModels("served")
    if "models" not in inputs:
        inputs["models"] = counted
    constructor = helpers.counting(fake_factory.constructor)
    failures = await _init_failures(clock, constructor, **inputs)
    for needle in spec["first"]:
        if needle == "<authentication>":
            assert _mentions_auth(failures), failures
        else:
            assert helpers.mentions(failures, needle), (needle, failures)
    for needle in spec["later"]:
        if needle == "<authentication>":
            assert not _mentions_auth(failures), failures
        else:
            assert not helpers.mentions(failures, needle), (needle, failures)
    if case == "configuration-before-identity":
        assert inputs["models"] is not counted
    elif inputs["models"] is counted:
        assert counted.model_name_calls == 1
    assert constructor.calls == 0
    assert fake_factory.names("factory") == []


# @spec ING-EP-068
async def test_configuration_failure_reads_no_identity(
    clock, fake_factory
) -> None:
    from support.fake_transcription import FakeServingModels

    models = FakeServingModels("served")
    failures = await _init_failures(
        clock,
        fake_factory.constructor,
        options={"surprise": 1},
        models=models,
    )
    assert any("surprise" in text for text in failures)
    assert models.model_name_calls == 0


# @spec ING-EP-081
@pytest.mark.parametrize(
    "count, passes",
    [(None, True), ("absent", True), (1, True), (0, False), (2, False)],
    ids=["none", "absent", "one", "headless-zero", "two"],
)
async def test_api_server_count_must_count_one(
    clock, fake_factory, count, passes
) -> None:
    from support.state import ABSENT

    args = _args(api_server_count=ABSENT if count == "absent" else count)
    if passes:
        plugin = await _init(clock, fake_factory.constructor, args=args)
        assert plugin.factory is fake_factory
    else:
        failures = await _init_failures(
            clock, fake_factory.constructor, args=args
        )
        assert any("api_server_count" in text for text in failures), failures
        assert fake_factory.names("factory") == []


def _declining(engine_client, model_config):
    return None


def _raising(engine_client, model_config):
    raise RuntimeError("constructor failed")


#: ING-EP-107: an invalid configuration, a failed capacity check, or a
#: failed discovery or selection each fails startup, naming the cause.
STARTUP_FAILURES = {
    "unknown-option": dict(options={"surprise": 1}, named=("surprise",)),
    "null-option": dict(
        options={"preconfiguration_timeout": None},
        named=("preconfiguration_timeout",),
    ),
    "no-state-pages": dict(num_gpu_blocks=0, named=("num_gpu_blocks",)),
    "unknown-state-pages": dict(num_gpu_blocks=None, named=("num_gpu_blocks",)),
    "unreadable-active-limit": dict(
        unreadable_limit=True, named=("max_num_seqs",)
    ),
    "no-accepting-constructor": dict(
        factories=("declining", "declining"), named=("first", "second")
    ),
    "two-accepting-constructors": dict(
        factories=("accepting", "accepting"), named=("first", "second")
    ),
    "constructor-raises": dict(
        factories=("accepting", "raising"), named=("second",)
    ),
    "named-constructor-absent": dict(
        factories=("accepting", "accepting"),
        options={"session_factory": "missing"},
        named=("missing",),
    ),
    "named-constructor-declines": dict(
        factories=("accepting", "declining"),
        options={"session_factory": "second"},
        named=("second",),
    ),
}


# @spec ING-EP-107
@pytest.mark.parametrize("case", list(STARTUP_FAILURES))
async def test_init_state_fails_on_config_capacity_or_discovery(
    clock, helpers, case
) -> None:
    from support.fake_session import FakeSessionFactory

    from vllm_riva_frontend.lifecycle import StartupError

    spec = dict(STARTUP_FAILURES[case])
    named = spec.pop("named")
    kinds = spec.pop("factories", ("accepting",))
    unreadable = spec.pop("unreadable_limit", False)
    built: list[FakeSessionFactory] = []

    def constructor(kind: str) -> Any:
        if kind == "declining":
            return _declining
        if kind == "raising":
            return _raising
        built.append(FakeSessionFactory())
        return built[-1].constructor

    names = ("first", "second") if len(kinds) == 2 else ("nemotron",)
    plugin = helpers.make_frontend(
        clock,
        factories={
            name: constructor(kind)
            for name, kind in zip(names, kinds, strict=True)
        },
    )
    engine = helpers.engine_with(**spec)
    if unreadable:
        del engine.vllm_config.scheduler_config.max_num_seqs
        del engine.vllm_config.scheduler_config.max_num_active_seqs
    with pytest.raises(StartupError) as info:
        await helpers.initialize(plugin, engine)
    for needle in named:
        assert helpers.mentions(info.value.failures, needle), (
            needle,
            info.value.failures,
        )


# @spec ING-EP-068, ING-VEH-026
async def test_configuration_is_checked_before_identity(
    clock, fake_factory, helpers
) -> None:
    from support.fake_transcription import FakeServingModels

    models = FakeServingModels("served")
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with(options={"surprise": 1})
    failures = await _expect_failure(helpers, plugin, engine, models=models)
    assert helpers.mentions(failures, "surprise")
    assert models.model_name_calls == 0


# @spec ING-EP-068
async def test_identity_is_checked_before_engine_deadlines(
    clock, fake_factory, helpers
) -> None:
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with(stream_admission={"never_fed_timeout": -1})
    models = SimpleNamespace(model_name=lambda: "")
    failures = await _expect_failure(helpers, plugin, engine, models=models)
    assert not helpers.mentions(failures, "never_fed_timeout")


# @spec ING-EP-068
async def test_deadlines_are_checked_before_capacity(
    clock, fake_factory, helpers
) -> None:
    failures = await helpers.startup_failures(
        clock,
        fake_factory,
        stream_admission={"draining_timeout": 0},
        data_parallel_size=2,
    )
    assert helpers.mentions(failures, "draining_timeout")
    assert not helpers.mentions(failures, "data_parallel")


# @spec ING-EP-068
async def test_capacity_is_checked_before_complete_file_service(
    clock, fake_factory, helpers
) -> None:
    failures = await helpers.startup_failures(
        clock,
        fake_factory,
        options={"max_concurrent_files": 50},
        num_gpu_blocks=None,
    )
    assert helpers.mentions(failures, "num_gpu_blocks")
    assert not helpers.mentions(failures, "max_concurrent_files")


# @spec ING-EP-068, ING-EP-070
async def test_route_table_is_checked_before_the_factory(
    clock, fake_factory, helpers
) -> None:
    constructor = helpers.counting(fake_factory.constructor)
    plugin = helpers.make_frontend(clock, factories={"nemotron": constructor})
    engine = helpers.engine_with()
    await _expect_failure(
        helpers,
        plugin,
        engine,
        extra_plugins=[_Overrider(f"{PLUGIN_PREFIX}/metadata")],
    )
    assert constructor.calls == 0


# @spec ING-EP-068, ING-EP-107
@pytest.mark.parametrize(
    "engine_kw",
    [
        {"options": {"surprise": 1}},
        {"stream_admission": {"never_fed_timeout": 0}},
        {"num_gpu_blocks": 0},
        {"data_parallel_size": 4},
        {"options": {"max_concurrent_files": 99}},
    ],
)
async def test_the_factory_is_constructed_only_after_every_other_check(
    clock, fake_factory, helpers, engine_kw
) -> None:
    constructor = helpers.counting(fake_factory.constructor)
    plugin = helpers.make_frontend(clock, factories={"nemotron": constructor})
    engine = helpers.engine_with(**engine_kw)
    await _expect_failure(helpers, plugin, engine)
    assert constructor.calls == 0
    assert fake_factory.teardown_calls == 0


# @spec ING-EP-107
async def test_init_state_succeeds_when_every_check_holds(
    initialized, fake_factory, fake_engine
) -> None:
    plugin, _ = initialized
    assert plugin.factory is fake_factory
    assert fake_factory.constructed_with == (
        fake_engine,
        fake_engine.model_config,
    )
    assert plugin.registry is not None
    assert plugin.ingress is not None
    assert plugin.catalog is not None
    assert plugin.servicer is not None


# @spec ING-EP-107
@pytest.mark.parametrize("accepting", [0, 2])
async def test_startup_fails_unless_exactly_one_factory_accepts(
    clock, fake_factory, helpers, accepting
) -> None:
    from support.fake_session import FakeSessionFactory

    other = FakeSessionFactory()
    factories = {
        "first": fake_factory.constructor if accepting else (lambda e, m: None),
        "second": other.constructor if accepting else (lambda e, m: None),
    }
    plugin = helpers.make_frontend(clock, factories=factories)
    failures = await _expect_failure(helpers, plugin, helpers.engine_with())
    if accepting:
        assert helpers.mentions(failures, "first")
        assert helpers.mentions(failures, "second")


# @spec ING-EP-051
async def test_session_factory_option_selects_among_several_accepting(
    clock, fake_factory, helpers
) -> None:
    from support.fake_session import FakeSessionFactory

    other = FakeSessionFactory()
    plugin = helpers.make_frontend(
        clock,
        factories={
            "first": other.constructor,
            "second": fake_factory.constructor,
        },
    )
    engine = helpers.engine_with(options={"session_factory": "second"})
    await helpers.initialize(plugin, engine)
    assert plugin.factory is fake_factory


# @spec ING-EP-021
async def test_capability_failure_of_the_factory_constructor_fails_startup(
    clock, helpers
) -> None:
    def constructor(engine_client, model_config):
        raise AttributeError("engine client has no admit_stream")

    plugin = helpers.make_frontend(clock, factories={"nemotron": constructor})
    failures = await _expect_failure(helpers, plugin, helpers.engine_with())
    assert helpers.mentions(failures, "nemotron")


# @spec ING-EP-021
async def test_no_vllm_version_is_checked(
    monkeypatch, clock, fake_factory, helpers
) -> None:
    import vllm

    monkeypatch.setattr(vllm, "__version__", "0.0.0.dev0", raising=False)
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    await helpers.initialize(plugin, helpers.engine_with())
    assert plugin.factory is fake_factory


def _settled(factory: Any) -> bool:
    """Whether ``factory`` went through exactly one factory settlement.

    ING-EP-096: ``drain(math.inf)``, then ``teardown_all``, then one more
    ``drain`` at a finite deadline (now).
    """
    return (
        factory.names("factory") == ["drain", "teardown_all", "drain"]
        and factory.drains[0] == math.inf
        and math.isfinite(factory.drains[1])
    )


# @spec ING-EP-069
@pytest.mark.parametrize(
    "named, discarded",
    [("second", ["first", "third"]), ("first", ["second", "third"])],
)
async def test_discarded_factories_are_settled_when_discarded(
    clock, helpers, named, discarded
) -> None:
    from support.fake_session import FakeSessionFactory

    factories = {
        name: FakeSessionFactory() for name in ("first", "second", "third")
    }
    plugin = helpers.make_frontend(
        clock,
        factories={
            name: factory.constructor for name, factory in factories.items()
        },
    )
    engine = helpers.engine_with(options={"session_factory": named})
    await helpers.initialize(plugin, engine)
    assert plugin.factory is factories[named]
    for name in discarded:
        assert _settled(factories[name]), (name, factories[name].names())
    assert factories[named].names("factory") == []


# @spec ING-EP-069
async def test_several_accepting_without_a_name_settles_each_and_fails(
    clock, helpers
) -> None:
    from support.fake_session import FakeSessionFactory

    first, second = FakeSessionFactory(), FakeSessionFactory()
    plugin = helpers.make_frontend(
        clock,
        factories={"first": first.constructor, "second": second.constructor},
    )
    await _expect_failure(helpers, plugin, helpers.engine_with())
    assert _settled(first), first.names()
    assert _settled(second), second.names()


# @spec ING-EP-069, ING-EP-097
@pytest.mark.parametrize(
    "case",
    ["unserved-cadence", "later-constructor-raises", "discard-then-fail"],
)
async def test_failed_init_state_settles_every_constructed_factory(
    clock, helpers, case
) -> None:
    """A failed ``init_state`` settles each constructed factory, once."""
    from support.fake_session import FakeSessionFactory

    def raising(engine_client, model_config):
        raise RuntimeError("constructor failed")

    if case == "unserved-cadence":
        # The factory does not serve the 560ms streaming cadence: the
        # cadence check after construction fails.
        built = [FakeSessionFactory(served_cadences=("1120ms",))]
        factories = {"nemotron": built[0].constructor}
        options: dict[str, Any] = {}
    elif case == "later-constructor-raises":
        built = [FakeSessionFactory()]
        factories = {"first": built[0].constructor, "second": raising}
        options = {}
    else:
        # "first" is discarded (and settled then); the named "second"
        # fails the cadence check and is settled when startup fails.
        built = [
            FakeSessionFactory(),
            FakeSessionFactory(served_cadences=("1120ms",)),
        ]
        factories = {
            "first": built[0].constructor,
            "second": built[1].constructor,
        }
        options = {"session_factory": "second"}
    plugin = helpers.make_frontend(clock, factories=factories)
    await _expect_failure(helpers, plugin, helpers.engine_with(options=options))
    for factory in built:
        assert _settled(factory), factory.names()
        assert factory.running_tasks() == set()


# @spec ING-EP-025
async def test_init_state_binds_no_listener_and_starts_no_task(
    frontend,
    test_app,
    fake_engine,
    fake_models,
    fake_transcription,
    grpc_port,
    helpers,
) -> None:
    state = helpers.prepare_state(
        test_app,
        fake_engine,
        models=fake_models,
        transcription=fake_transcription,
    )
    before = asyncio.all_tasks()
    await helpers.run_init(test_app, fake_engine, state)
    await asyncio.sleep(0)
    assert asyncio.all_tasks() - before == set()
    assert not helpers.port_open(grpc_port)
    assert getattr(frontend, "server", None) is None
    assert not frontend.admission_open


# @spec ING-EP-027
async def test_zero_shutdown_timeout_logs_that_no_drain_will_run(
    clock, fake_factory, helpers, caplog
) -> None:
    # ``--shutdown-timeout`` is read where core's launcher reads it:
    # ``engine_client.vllm_config.shutdown_timeout``.
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with(shutdown_timeout=0.0)
    app = helpers.build(plugin)
    state = helpers.prepare_state(app, engine)
    with caplog.at_level(logging.WARNING):
        await helpers.run_init(app, engine, state)
    warnings = [
        r.getMessage().lower()
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ]
    assert any("drain" in m and "stop" in m for m in warnings), warnings


# @spec ING-EP-027
async def test_the_args_namespace_is_not_the_shutdown_timeout_source(
    clock, fake_factory, helpers, caplog
) -> None:
    from support.state import make_args

    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with()  # vllm_config.shutdown_timeout = 60
    app = helpers.build(plugin, args=make_args(shutdown_timeout=0))
    state = helpers.prepare_state(app, engine)
    with caplog.at_level(logging.WARNING):
        await helpers.run_init(app, engine, state)
    assert not any(
        "drain" in r.getMessage().lower() and r.levelno >= logging.WARNING
        for r in caplog.records
    )


# @spec ING-EP-027
async def test_nonzero_shutdown_timeout_logs_no_drain_warning(
    initialized, caplog
) -> None:
    assert not any(
        "drain" in r.getMessage().lower() and r.levelno >= logging.WARNING
        for r in caplog.records
    )


# ------------------------------------------------------- served identity


# @spec ING-VEH-026
async def test_identity_comes_from_the_serving_registry_once(
    clock, fake_factory, helpers
) -> None:
    from support.fake_transcription import FakeServingModels
    from support.state import make_args

    models = FakeServingModels("registry-primary", "registry-alias")
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with(
        model="org/checkpoint", served_model_name="engine-name"
    )
    app = helpers.build(
        plugin, args=make_args(served_model_name=["cli-alias", "other"])
    )
    state = helpers.prepare_state(app, engine, models=models)
    await helpers.run_init(app, engine, state)
    assert plugin.served_model == "registry-primary"
    assert models.model_name_calls == 1


# @spec ING-VEH-026
async def test_identity_is_immutable_for_the_plugin_lifetime(
    started, fake_models
) -> None:
    from riva.client.proto import riva_asr_pb2
    from support.grpc_harness import FakeServicerContext

    plugin, _ = started
    first = plugin.served_model
    fake_models.base_model_paths[0].name = "renamed"
    response = await plugin.servicer.GetRivaSpeechRecognitionConfig(
        riva_asr_pb2.RivaSpeechRecognitionConfigRequest(), FakeServicerContext()
    )
    assert plugin.served_model == first
    assert {c.model_name for c in response.model_config} == {first}
    assert fake_models.model_name_calls == 1


class _Raising:
    def model_name(self) -> str:
        raise RuntimeError("registry lookup failed")


# @spec ING-VEH-026
@pytest.mark.parametrize(
    "models",
    [
        "missing",
        SimpleNamespace(),
        SimpleNamespace(model_name="not-callable"),
        _Raising(),
        SimpleNamespace(model_name=lambda: 42),
        SimpleNamespace(model_name=lambda: ""),
        SimpleNamespace(model_name=lambda: None),
    ],
    ids=[
        "no-registry",
        "no-accessor",
        "not-callable",
        "raises",
        "non-string",
        "empty",
        "none",
    ],
)
async def test_unusable_identity_fails_before_listener_or_lease(
    clock, fake_factory, helpers, grpc_port, models: Any
) -> None:
    from vllm_riva_frontend.lifecycle import StartupError

    constructor = helpers.counting(fake_factory.constructor)
    plugin = helpers.make_frontend(clock, factories={"nemotron": constructor})
    engine = helpers.engine_with(grpc_port)
    app = helpers.build(plugin)
    state = helpers.prepare_state(app, engine)
    if models == "missing":
        del state.openai_serving_models
    else:
        state.openai_serving_models = models
    with pytest.raises(StartupError):
        await helpers.run_init(app, engine, state)
    assert constructor.calls == 0
    assert fake_factory.leases == []
    assert not helpers.port_open(grpc_port)


# ------------------------------------------------- route table check


class _Overrider:
    """A later endpoint plugin that overrides one ``GET`` path."""

    name = "overrider"
    required_tasks = None

    def __init__(self, path: str, method: str = "GET") -> None:
        self.path = path
        self.method = method

    def attach_router(self, app) -> None:
        router = APIRouter()

        async def handler() -> dict[str, str]:
            return {"owner": "overrider"}

        router.add_api_route(self.path, handler, methods=[self.method])
        app.include_router(router)

    async def init_state(self, engine_client, state, args) -> None:
        return None


# @spec ING-EP-070
@pytest.mark.parametrize(
    "path, method",
    [
        (f"{PLUGIN_PREFIX}/metadata", "GET"),
        (f"{PLUGIN_PREFIX}/health/ready", "GET"),
        (f"{PLUGIN_PREFIX}/realtime/transcription_sessions", "POST"),
        (f"{PLUGIN_PREFIX}/audio/transcriptions", "POST"),
    ],
)
async def test_a_later_plugin_overriding_a_frontend_path_fails_startup(
    clock, fake_factory, helpers, path, method
) -> None:
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    failures = await _expect_failure(
        helpers,
        plugin,
        helpers.engine_with(),
        extra_plugins=[_Overrider(path, method)],
    )
    assert helpers.mentions(failures, path)


# @spec ING-EP-070
async def test_a_later_plugin_on_other_paths_does_not_fail_startup(
    clock, fake_factory, helpers
) -> None:
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    await helpers.initialize(
        plugin, helpers.engine_with(), extra_plugins=[_Overrider("/v1/other")]
    )
    assert plugin.factory is fake_factory

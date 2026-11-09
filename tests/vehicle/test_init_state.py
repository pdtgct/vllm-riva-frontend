"""``init_state``: check order, identity, factory last, no side effects."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import APIRouter

# --------------------------------------------------------------- order


async def _expect_failure(helpers, plugin, engine, **kw) -> list[str]:
    from vllm_riva_frontend.lifecycle import StartupError

    with pytest.raises(StartupError) as info:
        await helpers.initialize(plugin, engine, **kw)
    return list(info.value.failures)


# @spec ING-EP-023, ING-VEH-026
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


# @spec ING-EP-023
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


# @spec ING-EP-023
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


# @spec ING-EP-023
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


# @spec ING-EP-023, ING-EP-031
async def test_route_table_is_checked_before_the_factory(
    clock, fake_factory, helpers
) -> None:
    constructor = helpers.counting(fake_factory.constructor)
    plugin = helpers.make_frontend(clock, factories={"nemotron": constructor})
    engine = helpers.engine_with()
    await _expect_failure(
        helpers, plugin, engine, extra_plugins=[_Overrider("/v1/metadata")]
    )
    assert constructor.calls == 0


# @spec ING-EP-023, ING-EP-003
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


# @spec ING-EP-003
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


# @spec ING-EP-003
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
@pytest.mark.engine
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


# @spec ING-EP-024
async def test_failure_after_factory_construction_tears_the_factory_down(
    clock, helpers
) -> None:
    from support.fake_session import FakeSessionFactory

    # The factory does not serve the 560ms streaming cadence: the cadence
    # check after construction fails.
    factory = FakeSessionFactory(served_cadences=("1120ms",))
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": factory.constructor}
    )
    await _expect_failure(helpers, plugin, helpers.engine_with())
    assert factory.teardown_calls == 1
    assert factory.names("factory") == ["teardown_all"]


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


# @spec ING-EP-031
@pytest.mark.parametrize(
    "path, method",
    [
        ("/v1/metadata", "GET"),
        ("/v1/health/ready", "GET"),
        ("/v1/realtime/transcription_sessions", "POST"),
        ("/v1/audio/transcriptions", "POST"),
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


# @spec ING-EP-031
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

"""Factory discovery, selection, the cadence check and the locale rule."""

from __future__ import annotations

import pytest
from support.fake_session import FakeSessionFactory, fake_entry_points

from .conftest import failure_text, startup_failure


def _discover(engine, entry_points, selected=None):
    from vllm_riva_frontend.lease import discover_session_factory

    return discover_session_factory(
        engine, selected=selected, entry_points=entry_points
    )


# ---------------------------------------------------------------- discovery


# @spec ING-CORE-012
def test_every_constructor_gets_the_engine_client_and_its_model_config(
    fake_engine,
):
    from vllm_riva_frontend.lease import SESSION_FACTORIES_GROUP

    assert SESSION_FACTORIES_GROUP == "vllm_riva_frontend.session_factories"
    accepting = FakeSessionFactory()
    seen: list[tuple[object, object]] = []

    def declining(engine_client, model_config):
        seen.append((engine_client, model_config))
        return None

    groups: list[object] = []
    points = fake_entry_points(other=declining, ours=accepting.constructor)

    def entry_points(**params):
        groups.append(params.get("group"))
        return points(**params)

    factory = _discover(fake_engine, entry_points)

    assert factory is accepting
    assert groups and set(groups) == {SESSION_FACTORIES_GROUP}
    assert seen == [(fake_engine, fake_engine.model_config)]
    assert seen[0][0] is fake_engine
    assert seen[0][1] is fake_engine.model_config
    assert accepting.constructed_with[0] is fake_engine
    assert accepting.constructed_with[1] is fake_engine.model_config


# @spec ING-CORE-012
def test_a_raising_constructor_fails_startup_with_its_message(fake_engine):
    from vllm_riva_frontend.lifecycle import StartupError

    def broken(engine_client, model_config):
        raise RuntimeError("engine client has no wait_stream_retired")

    with pytest.raises(StartupError) as info:
        _discover(
            fake_engine,
            fake_entry_points(
                broken=broken, ok=FakeSessionFactory().constructor
            ),
        )
    assert "wait_stream_retired" in failure_text(info.value)


# @spec ING-CORE-017
@pytest.mark.parametrize("accepting", [0, 2])
def test_zero_or_several_accepting_factories_fail_listing_names(
    fake_engine, accepting
):
    from vllm_riva_frontend.lifecycle import StartupError

    constructors = {
        f"factory_{i}": FakeSessionFactory().constructor
        for i in range(accepting)
    }
    constructors["declines"] = lambda engine_client, model_config: None

    with pytest.raises(StartupError) as info:
        _discover(fake_engine, fake_entry_points(**constructors))
    text = failure_text(info.value)
    for name in constructors:
        assert name in text


# @spec ING-CORE-017
def test_a_named_factory_is_the_only_one_used(fake_engine):
    first, second = FakeSessionFactory(), FakeSessionFactory()

    factory = _discover(
        fake_engine,
        fake_entry_points(first=first.constructor, second=second.constructor),
        selected="second",
    )

    assert factory is second


# @spec ING-CORE-017
def test_a_named_factory_that_is_absent_or_declines_fails_startup(
    fake_engine,
):
    from vllm_riva_frontend.lifecycle import StartupError

    points = fake_entry_points(
        ok=FakeSessionFactory().constructor,
        declines=lambda engine_client, model_config: None,
    )
    with pytest.raises(StartupError) as absent:
        _discover(fake_engine, points, selected="missing")
    assert "missing" in failure_text(absent.value)
    with pytest.raises(StartupError) as declined:
        _discover(fake_engine, points, selected="declines")
    assert "declines" in failure_text(declined.value)


# @spec ING-CORE-017, ING-CORE-012
async def test_init_state_reads_the_session_factory_option(clock):
    from support.app import build_app, init_state
    from support.fake_engine import FakeEngineClient
    from support.grpc_harness import free_port

    from vllm_riva_frontend.lifecycle import RivaFrontendPlugin

    first, second = FakeSessionFactory(), FakeSessionFactory()
    plugin = RivaFrontendPlugin(
        clock=clock,
        entry_points=fake_entry_points(
            first=first.constructor, second=second.constructor
        ),
    )
    engine = FakeEngineClient(
        riva_frontend={
            "grpc_bind": f"127.0.0.1:{free_port()}",
            "session_factory": "first",
        }
    )
    app = build_app([plugin])
    await init_state(app, engine)

    assert plugin.factory is first


# ------------------------------------------------------------------ cadence


# @spec ING-CORE-014
async def test_startup_fails_when_the_factory_does_not_serve_560ms(clock):
    from support.app import build_app, init_state
    from support.fake_engine import FakeEngineClient
    from support.grpc_harness import free_port

    from vllm_riva_frontend.lease import STREAM_CADENCE
    from vllm_riva_frontend.lifecycle import RivaFrontendPlugin

    assert STREAM_CADENCE == "560ms"
    factory = FakeSessionFactory(served_cadences=("320ms", "1120ms"))
    plugin = RivaFrontendPlugin(
        clock=clock, entry_points=fake_entry_points(fake=factory.constructor)
    )
    engine = FakeEngineClient(
        riva_frontend={"grpc_bind": f"127.0.0.1:{free_port()}"}
    )
    app = build_app([plugin])

    with pytest.raises(BaseException) as info:
        await init_state(app, engine)
    assert startup_failure(info.value) is not None
    assert "560ms" in failure_text(info.value)


# @spec ING-CORE-014
async def test_every_stream_opens_at_560ms(riva, fake_factory):
    from support.audio import pcm16_bytes, pcm16_sine

    call = riva()
    call.configure()
    call.audio(pcm16_bytes(pcm16_sine(0.1)))
    call.half_close()
    await call.wait()

    opens = [c for c in fake_factory.calls if c.name == "open"]
    assert [c.kwargs["cadence"] for c in opens] == ["560ms"]


# ------------------------------------------------------------------ locale


# @spec ING-CORE-015
def test_resolve_locale_validates_defaults_and_requires_a_name():
    from vllm_riva_frontend.errors import FrontendError
    from vllm_riva_frontend.lease import resolve_locale

    factory = FakeSessionFactory(
        locales=("en-US", "de-DE"), default_locale="en-US"
    )
    assert resolve_locale("de-DE", factory, field="language_code") == "de-DE"
    assert resolve_locale("", factory, field="language_code") == "en-US"
    assert resolve_locale(None, factory, field="language_code") == "en-US"

    with pytest.raises(FrontendError) as unknown:
        resolve_locale("xx-XX", factory, field="language_code")
    assert unknown.value.code == "unknown_locale"

    no_default = FakeSessionFactory(locales=("en-US",), default_locale=None)
    for missing in ("", None):
        with pytest.raises(FrontendError) as info:
            resolve_locale(missing, no_default, field="language_code")
        assert info.value.code == "invalid_config_field"
        assert "language_code" in tuple(info.value.fields)


# @spec ING-CORE-015
async def test_riva_stream_without_language_opens_at_the_default_locale(
    riva, fake_factory
):
    call = riva()
    call.configure(language_code="")
    call.half_close()
    await call.wait()

    opens = [c for c in fake_factory.calls if c.name == "open"]
    assert [c.kwargs["locale"] for c in opens] == [fake_factory.default_locale]
    call.assert_ok()


class TestNoDefaultLocale:
    @pytest.fixture
    def fake_factory(self, clock):
        return FakeSessionFactory(
            locales=("en-US",), default_locale=None, final_text="x", clock=clock
        )

    # @spec ING-CORE-015
    async def test_riva_stream_without_language_is_invalid_config_field(
        self, riva, fake_factory
    ):
        import grpc

        call = riva()
        call.configure(language_code="")
        await call.wait()

        call.assert_status(
            grpc.StatusCode.INVALID_ARGUMENT,
            "invalid_config_field",
            "language_code",
        )
        assert [c.name for c in fake_factory.calls if c.name == "open"] == []

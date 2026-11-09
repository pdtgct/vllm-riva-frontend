"""The ingress cap, the file sub-cap, engine refusals and bounded ingress."""

from __future__ import annotations

import asyncio
import logging

import grpc
import pytest
from support.audio import pcm16_bytes, pcm16_sine
from support.fake_session import Block, Raise, refused
from support.realtime import (
    CREATED,
    ERROR,
    UPDATED,
    RealtimeClient,
    error_authority,
    error_code,
    error_fields,
)
from support.riva import config_request, recognize, recognize_request

# ------------------------------------------------------------ IngressCap


def cap(max_sessions: int = 4, max_file_owners: int = 2):
    """An opened ``IngressCap`` (constructed closed; ``start`` opens it)."""
    from vllm_riva_frontend.admission import IngressCap

    ingress = IngressCap(
        max_sessions=max_sessions, max_file_owners=max_file_owners
    )
    ingress.open()
    return ingress


# @spec ING-EP-035, ING-ADM-011
def test_a_new_cap_is_closed_until_opened() -> None:
    from vllm_riva_frontend.admission import IngressCap, OwnerKind

    ingress = IngressCap(max_sessions=4, max_file_owners=2)
    assert not ingress.is_open
    assert ingress.try_admit(OwnerKind.stream) is None
    assert ingress.count == 0
    assert ingress.refusal(OwnerKind.stream).code == "service_unavailable"
    ingress.open()
    assert ingress.is_open
    assert ingress.try_admit(OwnerKind.stream) is not None


# @spec ING-ADM-017
def test_four_owner_kinds_count_and_two_are_file_kinds() -> None:
    from vllm_riva_frontend.admission import FILE_KINDS, OwnerKind

    ingress = cap(max_sessions=8, max_file_owners=2)
    kinds = [
        OwnerKind.stream,
        OwnerKind.realtime,
        OwnerKind.recognize,
        OwnerKind.file_route,
    ]
    admissions = [ingress.try_admit(kind) for kind in kinds]
    assert all(a is not None for a in admissions)
    assert ingress.count == 4
    assert ingress.file_count == 2
    assert set(FILE_KINDS) == {OwnerKind.recognize, OwnerKind.file_route}


# @spec ING-ADM-011, ING-ADM-017
def test_full_cap_refuses_busy_with_load_shed_authority() -> None:
    from vllm_riva_frontend.admission import OwnerKind

    ingress = cap(max_sessions=2, max_file_owners=1)
    assert ingress.try_admit(OwnerKind.stream) is not None
    assert ingress.try_admit(OwnerKind.realtime) is not None
    assert ingress.try_admit(OwnerKind.stream) is None
    assert ingress.count == 2
    error = ingress.refusal(OwnerKind.stream)
    assert error.code == "busy"
    assert error.authority == "load_shed"


# @spec ING-ADM-011
def test_closed_cap_refuses_service_unavailable() -> None:
    from vllm_riva_frontend.admission import OwnerKind

    ingress = cap()
    assert ingress.is_open
    ingress.close()
    assert not ingress.is_open
    assert ingress.try_admit(OwnerKind.stream) is None
    assert ingress.count == 0
    assert ingress.refusal(OwnerKind.stream).code == "service_unavailable"


# @spec ING-ADM-013, ING-ADM-016
def test_admission_release_is_idempotent_and_frees_the_count() -> None:
    from vllm_riva_frontend.admission import OwnerKind

    ingress = cap(max_sessions=2, max_file_owners=1)
    admission = ingress.try_admit(OwnerKind.recognize)
    assert (ingress.count, ingress.file_count) == (1, 1)
    admission.release()
    admission.release()
    assert (ingress.count, ingress.file_count) == (0, 0)


# @spec ING-ADM-010
def test_file_sub_cap_refuses_files_while_streams_keep_headroom() -> None:
    from vllm_riva_frontend.admission import OwnerKind

    ingress = cap(max_sessions=4, max_file_owners=2)
    assert ingress.try_admit(OwnerKind.recognize) is not None
    assert ingress.try_admit(OwnerKind.file_route) is not None
    assert ingress.try_admit(OwnerKind.file_route) is None
    error = ingress.refusal(OwnerKind.file_route)
    assert (error.code, error.authority) == ("busy", "load_shed")
    assert ingress.try_admit(OwnerKind.stream) is not None
    assert ingress.try_admit(OwnerKind.realtime) is not None
    assert ingress.count == 4


# @spec ING-ADM-012
def test_admission_below_the_cap_touches_no_engine_or_factory(
    fake_factory, fake_engine
) -> None:
    from vllm_riva_frontend.admission import OwnerKind

    ingress = cap()
    ingress.try_admit(OwnerKind.stream)
    assert fake_factory.calls == []
    assert fake_engine.aborted == []
    assert fake_engine.health_checks == 0


# @spec ING-ADM-010, ING-EP-050
@pytest.mark.parametrize(
    "options, named",
    [
        (
            {"max_concurrent_files": 3, "max_file_owners": 2},
            "max_file_owners",
        ),
        (
            {
                "load_shed_max_sessions": 6,
                "max_concurrent_files": 2,
                "max_file_owners": 6,
            },
            "max_file_owners",
        ),
    ],
)
async def test_sub_cap_relations_fail_startup(
    clock, fake_factory, helpers, options, named
) -> None:
    failures = await helpers.startup_failures(
        clock, fake_factory, options=options
    )
    assert helpers.mentions(failures, named)


# ------------------------------------------------------------ derived caps


def _derive(capacity, *, service=True, **options):
    from vllm_riva_frontend.config import (
        Capacity,
        derive_file_caps,
        parse_config,
    )

    config = parse_config({"riva_frontend": options})
    return derive_file_caps(config, Capacity(*capacity), service)


# @spec ING-ADM-014
@pytest.mark.parametrize(
    "capacity, expected",
    [
        ((16, 16), (4, 6, True)),
        ((64, 64), (4, 6, True)),
        ((16, 8), (2, 4, True)),
        ((5, 32), (1, 3, True)),
        ((2, 2), (1, 3, True)),
        ((1, 16), (0, 0, False)),
        ((16, 1), (0, 0, False)),
    ],
)
def test_default_file_caps_are_derived_from_capacity(
    capacity, expected
) -> None:
    assert _derive(capacity) == expected


# @spec ING-ADM-014
def test_absent_transcription_service_disables_files() -> None:
    assert _derive((16, 16), service=False) == (0, 0, False)


# @spec ING-ADM-014
def test_configured_file_caps_are_kept() -> None:
    concurrent, owners, enabled = _derive(
        (16, 16), max_concurrent_files=2, max_file_owners=5
    )
    assert (concurrent, owners, enabled) == (2, 5, True)


# @spec ING-ADM-014
async def test_explicit_file_cap_on_a_tiny_deployment_fails_startup(
    clock, fake_factory, helpers
) -> None:
    failures = await helpers.startup_failures(
        clock,
        fake_factory,
        options={"max_concurrent_files": 1},
        num_gpu_blocks=2,
    )
    assert helpers.mentions(failures, "max_concurrent_files")


# @spec ING-ADM-014
async def test_tiny_deployment_disables_files_with_a_warning(
    clock, fake_factory, helpers, caplog
) -> None:
    # Another import may have raised the root logger's level above WARNING.
    caplog.set_level(logging.WARNING)
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with(num_gpu_blocks=2)
    await helpers.initialize(plugin, engine)
    assert plugin.files_enabled is False
    assert plugin.file_gate is None
    assert (
        plugin.config.max_concurrent_files,
        plugin.config.max_file_owners,
    ) == (
        0,
        0,
    )
    assert any(r.levelname == "WARNING" for r in caplog.records)


# @spec ING-ADM-014
async def test_without_transcription_service_no_file_owner_is_created(
    clock, fake_factory, helpers
) -> None:
    from support.fake_transcription import FakeServingModels

    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    _, state = await helpers.initialize(
        plugin,
        helpers.engine_with(),
        tasks=("realtime",),
        models=FakeServingModels(helpers.SERVED),
    )
    await plugin.start(state)
    try:
        assert plugin.files_enabled is False
        assert plugin.file_gate is None
        outcome = await recognize(
            plugin.servicer, recognize_request(pcm16_bytes(pcm16_sine(0.1)))
        )
        assert outcome.status is grpc.StatusCode.UNIMPLEMENTED
        assert outcome.code == "unsupported_capability"
        assert not plugin.registry.records
        assert plugin.ingress.count == 0
    finally:
        await helpers.drive(clock, plugin.stop(state))


# ---------------------------------------------- the cap on every surface


async def _fill(plugin, count: int):
    from vllm_riva_frontend.admission import OwnerKind

    return [plugin.ingress.try_admit(OwnerKind.stream) for _ in range(count)]


# @spec ING-ADM-017, ING-ADM-011
async def test_full_cap_refuses_a_stream_before_its_first_read(
    small_cap, fake_factory, helpers
) -> None:
    plugin, _, _, _ = small_cap
    await _fill(plugin, 3)
    run = helpers.stream(plugin, plugin.clock, config_request())
    await asyncio.wait_for(run.task, 2.0)
    assert run.status is grpc.StatusCode.RESOURCE_EXHAUSTED
    assert run.details.startswith("busy")
    assert run.requests.pulls == 0
    assert "open" not in fake_factory.names()
    assert not plugin.registry.records


# @spec ING-ADM-017, ING-ADM-011
async def test_full_cap_refuses_recognize_before_decode(
    small_cap, fake_factory, helpers
) -> None:
    plugin, _, _, transcription = small_cap
    await _fill(plugin, 3)
    outcome = await recognize(
        plugin.servicer, recognize_request(b"not audio at all")
    )
    assert outcome.status is grpc.StatusCode.RESOURCE_EXHAUSTED
    assert outcome.details.startswith("busy: authority: load_shed")
    assert transcription.calls == []
    assert not plugin.registry.records


# @spec ING-ADM-017, ING-ADM-011
async def test_full_cap_refuses_the_file_route_before_reading_the_body(
    small_cap, helpers
) -> None:
    from support.asgi import asgi_http

    from vllm_riva_frontend.errors import catalog

    plugin, app, _, transcription = small_cap
    await _fill(plugin, 3)
    body, headers = helpers.file_request_body()
    never = asyncio.Event()
    response = await asgi_http(
        app,
        "POST",
        "/v1/audio/transcriptions",
        headers=headers,
        body=body,
        hold_body=never,
        timeout=3.0,
    )
    assert response.error is None, "answered without waiting for the body"
    assert response.status == catalog().http_status("busy")
    assert transcription.calls == []
    assert not plugin.registry.records


# @spec ING-ADM-017, ING-ADM-011
async def test_full_cap_refuses_a_realtime_session_without_open(
    small_cap, fake_factory
) -> None:
    plugin, app, _, _ = small_cap
    await _fill(plugin, 3)
    client = RealtimeClient(app, plugin.clock)
    assert await client.connect()
    before = await client.assert_closes(1013, timeout=3.0)
    assert [error_code(e) for e in before] == ["busy"], before
    assert error_authority(before[0]) == "load_shed"
    assert "open" not in fake_factory.names()
    assert plugin.ingress.count == 3


# @spec ING-ADM-017
async def test_excluded_work_is_served_at_a_full_cap(small_cap) -> None:
    from riva.client.proto import riva_asr_pb2
    from support.app import http_client
    from support.asgi import AsgiWebSocket
    from support.grpc_harness import FakeServicerContext

    plugin, app, _, _ = small_cap
    await _fill(plugin, 3)
    reply = await plugin.servicer.GetRivaSpeechRecognitionConfig(
        riva_asr_pb2.RivaSpeechRecognitionConfigRequest(), FakeServicerContext()
    )
    assert len(reply.model_config) >= 1
    async with http_client(app) as client:
        boot = await client.post("/v1/realtime/transcription_sessions")
        assert boot.status_code == 200
        for path in ("/v1/health/live", "/v1/version", "/v1/metadata"):
            assert (await client.get(path)).status_code == 200, path
    ws = AsgiWebSocket(app, "/v1/realtime")
    assert await ws.connect()
    assert (await ws.receive_json())["type"] == "core.realtime"
    assert plugin.ingress.count == 3


# @spec ING-ADM-012
async def test_open_sessions_continue_while_new_ones_are_refused(
    small_cap, fake_factory, clock, helpers
) -> None:
    plugin, _, _, _ = small_cap
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    await _fill(plugin, 2)
    refused_run = helpers.stream(plugin, plugin.clock, config_request())
    await asyncio.wait_for(refused_run.task, 2.0)
    assert refused_run.status is grpc.StatusCode.RESOURCE_EXHAUSTED
    run.requests.push(helpers.audio_message())
    run.requests.close()
    await helpers.settle_until(clock, lambda: run.done)
    assert run.status is grpc.StatusCode.OK
    assert lease.names().count("feed") >= 2
    assert lease.names()[-3:] == ["flush", "finish", "release"]


# ------------------------------------------------------- engine refusals


# @spec ING-ADM-008
@pytest.mark.parametrize("reason", ["capacity", "active_limit"])
def test_engine_refusal_projects_busy_with_provider_authority(reason) -> None:
    from vllm_riva_frontend.errors import project_seam_error

    error = project_seam_error(refused(reason))
    assert error.code == "busy"
    assert error.authority == "provider"


# @spec ING-ADM-008
@pytest.mark.parametrize("reason", ["capacity", "active_limit"])
async def test_engine_refusal_consumes_no_audio_and_sends_no_ack(
    started, fake_factory, clock, helpers, reason
) -> None:
    plugin, _ = started
    fake_factory.script("open", Raise(refused(reason)))
    run = helpers.stream(
        plugin, plugin.clock, config_request(), helpers.audio_message()
    )
    await helpers.settle_until(clock, lambda: run.done)
    assert run.status is grpc.StatusCode.RESOURCE_EXHAUSTED
    assert run.details.startswith("busy")
    assert run.responses == [], "no invented success acknowledgement"
    assert fake_factory.leases == []
    assert not plugin.registry.records
    assert plugin.ingress.count == 0


# -------------------------------------------------------- bounded ingress


# @spec ING-ADM-016
async def test_first_message_is_bounded_by_preconfiguration_timeout(
    started, fake_factory, clock, helpers
) -> None:
    plugin, _ = started
    run = helpers.stream(plugin, plugin.clock)
    await helpers.settle_until(clock, lambda: bool(plugin.registry.records))
    await clock.advance(plugin.config.preconfiguration_timeout - 1)
    assert not run.done
    await clock.advance(2)
    await helpers.settle_until(clock, lambda: run.done)
    run.assert_status(
        grpc.StatusCode.DEADLINE_EXCEEDED, "configuration_timeout"
    )
    assert "open" not in fake_factory.names()
    assert not plugin.registry.records
    assert plugin.ingress.count == 0


# @spec ING-ADM-016
async def test_client_disconnect_releases_lease_record_and_count(
    started, fake_factory, clock, helpers
) -> None:
    plugin, _ = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    run.context.cancel()
    await helpers.settle_until(clock, lambda: run.done)
    assert lease.names()[-2:] == ["abort", "release"]
    assert not plugin.registry.records
    assert plugin.ingress.count == 0


# @spec ING-ADM-016
async def test_file_body_read_is_bounded_and_releases_its_owner(
    started, test_app, clock, helpers
) -> None:
    from support.asgi import asgi_http

    plugin, _ = started
    body, headers = helpers.file_request_body()
    never = asyncio.Event()
    request = asyncio.ensure_future(
        asgi_http(
            test_app,
            "POST",
            "/v1/audio/transcriptions",
            headers=headers,
            body=body,
            hold_body=never,
            timeout=5.0,
        )
    )
    await helpers.settle_until(clock, lambda: bool(plugin.registry.records))
    await clock.advance(plugin.config.http_body_read_timeout + 1)
    response = await asyncio.wait_for(request, 5.0)
    assert response.status is not None and response.status >= 400
    await helpers.settle_until(clock, lambda: not plugin.registry.records)
    assert plugin.ingress.count == 0
    assert plugin.ingress.file_count == 0


# --------------------------------------------------- selectors and locale


# @spec ING-EP-026
@pytest.mark.parametrize(
    "model, accepted",
    [("primary", True), ("", True), ("secondary", False), ("org/ckpt", False)],
)
async def test_riva_accepts_only_the_primary_identity_as_selector(
    clock, fake_factory, helpers, model, accepted
) -> None:
    from support.fake_transcription import FakeServingModels

    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with(model="org/ckpt")
    _, state = await helpers.initialize(
        plugin, engine, models=FakeServingModels("primary", "secondary")
    )
    await plugin.start(state)
    try:
        run = helpers.stream(
            plugin,
            plugin.clock,
            config_request(model=model),
            helpers.audio_message(),
        )
        if accepted:
            await helpers.settle_until(clock, lambda: bool(fake_factory.leases))
            run.requests.close()
        await helpers.settle_until(clock, lambda: run.done)
        if accepted:
            assert run.status is grpc.StatusCode.OK
        else:
            assert run.status is grpc.StatusCode.INVALID_ARGUMENT
            assert "model" in run.details
            assert "open" not in fake_factory.names()
    finally:
        await helpers.drive(clock, plugin.stop(state))


# @spec ING-EP-026
@pytest.mark.parametrize("model", ["secondary", "org/ckpt"])
async def test_realtime_rejects_secondary_and_checkpoint_selectors(
    clock, fake_factory, helpers, model
) -> None:
    from support.fake_transcription import FakeServingModels

    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    app, state = await helpers.initialize(
        plugin,
        helpers.engine_with(model="org/ckpt"),
        models=FakeServingModels("primary", "secondary"),
    )
    await plugin.start(state)
    try:
        client = RealtimeClient(app, clock)
        assert await client.connect()
        assert (await client.next())["type"] == CREATED
        await client.send(helpers.realtime_update(model=model))
        error = await client.until_type(ERROR)
        assert error_code(error) == "invalid_config_field", error
        assert error_fields(error) == ["model"], error
        assert "open" not in fake_factory.names()
        await client.disconnect()
    finally:
        await helpers.drive(clock, plugin.stop(state))


# @spec ING-VEH-007
async def test_streams_pass_cadence_and_locale_only_to_open(
    started, fake_factory, clock, helpers
) -> None:
    plugin, _ = started
    config_before = repr(plugin.config)
    default_before = fake_factory.default_locale
    first, _ = await helpers.stream_with_lease(
        plugin, fake_factory, clock, locale="en-US"
    )
    second, _ = await helpers.stream_with_lease(
        plugin, fake_factory, clock, locale="de-DE"
    )
    opens = [c for c in fake_factory.calls if c.name == "open"]
    assert [c.kwargs for c in opens] == [
        {"cadence": "560ms", "locale": "en-US"},
        {"cadence": "560ms", "locale": "de-DE"},
    ]
    assert [c.args for c in opens] == [(), ()]
    assert repr(plugin.config) == config_before
    assert fake_factory.default_locale == default_before
    for run in (first, second):
        run.requests.close()
    await helpers.settle_until(clock, lambda: first.done and second.done)


# @spec ING-VEH-007
async def test_riva_has_no_mid_session_configuration_update(
    started, fake_factory, clock, helpers
) -> None:
    plugin, _ = started
    run, lease = await helpers.stream_with_lease(plugin, fake_factory, clock)
    run.requests.push(config_request(language_code="de-DE"))
    await helpers.settle_until(clock, lambda: run.done)
    run.assert_status(grpc.StatusCode.FAILED_PRECONDITION, "protocol_order")
    assert "update_locale" not in lease.names()
    assert lease.locale == "en-US"


# @spec ING-VEH-007
async def test_realtime_locale_change_goes_through_update_locale(
    started, test_app, fake_factory, clock, helpers
) -> None:
    plugin, _ = started
    client = RealtimeClient(test_app, clock)
    first = helpers.realtime_update(language="en-US")["session"]
    await client.open_session(first)
    ws = client.ws
    assert len(fake_factory.leases) == 1
    lease = fake_factory.leases[0]
    block = Block()
    lease.script("update_locale", block)
    await client.send(helpers.realtime_update(language="de-DE"))
    await helpers.settle_until(clock, lambda: block.entered.is_set())
    acknowledged = [
        m
        for m in ws.received
        if m["type"] == "websocket.send" and "updated" in (m.get("text") or "")
    ]
    assert len(acknowledged) == 1, "no acknowledgement before update_locale"
    block.release()
    updated = await client.until_type(UPDATED)
    language = updated["session"]["input_audio_transcription"]["language"]
    assert language == "de-DE"
    assert lease.names().count("update_locale") == 1
    assert [c.name for c in fake_factory.calls].count("open") == 1
    await client.disconnect()

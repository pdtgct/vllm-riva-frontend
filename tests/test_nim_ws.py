"""GPU-free raw-ASGI tests for the NIM Realtime compatibility adapter."""

import asyncio
import base64
import json
from dataclasses import dataclass, field, replace

import pytest

from vllm_riva_frontend.admission import LoadShedGate
from vllm_riva_frontend.lease import DirectLeaseOwner
from vllm_riva_frontend.nim_ws import (
    DispatchDecision,
    bootstrap_session,
    dispatch_decision,
    dispatch_nim_realtime,
    project_event,
)


def _scope(query_string: bytes) -> dict[str, object]:
    return {
        "type": "websocket",
        "path": "/v1/realtime",
        "query_string": query_string,
    }


@dataclass
class _Lease:
    calls: list[str] = field(default_factory=list)

    async def feed(self, samples: object, *, on_accepted: object) -> list[str]:
        self.calls.append(f"feed:{len(samples)}")  # type: ignore[arg-type]
        on_accepted(len(samples))  # type: ignore[operator]
        return ["hello"]

    async def update_locale(self, locale: str) -> object:
        self.calls.append(f"locale:{locale}")
        return None

    async def flush(self) -> str:
        self.calls.append("flush")
        return "hello world"

    async def finish(self) -> None:
        self.calls.append("finish")

    async def abort(self) -> None:
        self.calls.append("abort")

    async def release(self) -> None:
        self.calls.append("release")


@dataclass
class _Factory:
    lease: _Lease = field(default_factory=_Lease)
    opens: list[tuple[str, str]] = field(default_factory=list)

    async def open(self, *, cadence: str, locale: str) -> _Lease:
        self.opens.append((cadence, locale))
        return self.lease


@dataclass(frozen=True)
class _Config:
    load_shed_max_sessions: int = 2
    session_cleanup_timeout: float = 1.0
    ws_event_envelope_max_bytes: int = 4096
    max_riff_header_bytes: int = 256
    preconfiguration_timeout: float = 1.0
    session_idle_timeout: float = 1.0
    max_session_duration: float | None = None


@dataclass(frozen=True)
class _ModelConfig(_Config):
    model_name: str = "nemotron"
    locales: frozenset[str] = frozenset({"auto", "en-US"})


def _event(event: dict[str, object]) -> dict[str, object]:
    return {"type": "websocket.receive", "text": json.dumps(event)}


def _session(*, fmt: str = "pcm16") -> dict[str, object]:
    return {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": fmt,
            "input_audio_params": {"sample_rate_hz": 16000, "num_channels": 1},
            "input_audio_transcription": {"language": "auto"},
        },
    }


async def _drive(
    events: list[dict[str, object]],
    *,
    config: _Config | None = None,
) -> tuple[list[dict[str, object]], _Factory]:
    queue = list(events)
    sent: list[dict[str, object]] = []
    factory = _Factory()

    async def receive() -> dict[str, object]:
        return queue.pop(0) if queue else {"type": "websocket.disconnect"}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    async def native(scope: object, receive: object, send: object) -> None:
        raise AssertionError("NIM intent must not reach native route")

    app = dispatch_nim_realtime(
        app=native, factory=factory, config=config or _Config()
    )
    await app(_scope(b"intent=transcription"), receive, send)
    return sent, factory


def _json_events(sent: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        json.loads(message["text"])
        for message in sent
        if message["type"] == "websocket.send"
    ]


# @spec ING-NIMWS-002
def test_only_exact_transcription_intent_is_claimed() -> None:
    assert (
        dispatch_decision(_scope(b"intent=transcription"))
        is DispatchDecision.CLAIM
    )
    assert dispatch_decision(_scope(b"")) is DispatchDecision.PASS
    assert (
        dispatch_decision(_scope(b"intent=transcription&x=1"))
        is DispatchDecision.PASS
    )
    assert dispatch_decision(_scope(b"intent=%ff")) is DispatchDecision.DENY


# @spec ING-NIMWS-001, ING-CORE-008
def test_bootstrap_is_echoable_and_never_claims_admission() -> None:
    session = bootstrap_session("nemotron")
    assert session["object"] == "realtime.transcription_session"
    assert session["input_audio_transcription"]["model"] == "nemotron"
    assert "admitted" not in session


# @spec ING-VEH-013, ING-NIMWS-008
def test_canonical_checkpoint_is_rejected_when_realtime_alias_is_set() -> None:
    update = _session()
    session = update["session"]
    assert isinstance(session, dict)
    transcription = session["input_audio_transcription"]
    assert isinstance(transcription, dict)
    transcription["model"] = "nvidia/nemotron-asr"

    sent, factory = asyncio.run(_drive([_event(update)], config=_ModelConfig()))
    events = _json_events(sent)
    assert events[0]["type"] == "conversation.created"
    assert events[1]["type"] == "error"
    assert events[1]["error"]["code"] == "invalid_config_field"
    assert events[1]["error"]["param"] == "input_audio_transcription.model"
    assert factory.opens == []


# @spec ING-NIMWS-003, ING-CORE-007, ING-NIMWS-004, ING-LIFE-002
def test_raw_asgi_update_append_done_emits_suffix_and_completes() -> None:
    audio = base64.b64encode(b"\x01\x00").decode()
    sent, factory = asyncio.run(
        _drive(
            [
                _event(_session()),
                _event({"type": "input_audio_buffer.append", "audio": audio}),
                _event({"type": "input_audio_buffer.done"}),
            ]
        )
    )
    events = _json_events(sent)
    assert events[0]["type"] == "conversation.created"
    assert [event["type"] for event in events] == [
        "conversation.created",
        "transcription_session.updated",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.completed",
    ]
    assert [
        event["delta"] for event in events if event["type"].endswith(".delta")
    ] == ["hello", " world"]
    assert factory.opens == [("560ms", "auto")]
    assert factory.lease.calls == ["feed:1", "flush", "finish", "release"]
    assert sent[-1] == {"type": "websocket.close", "code": 1000}


# @spec ING-LIFE-001, ING-NIMWS-004
def test_raw_asgi_rejects_audio_before_open_without_creating_a_lease() -> None:
    sent, factory = asyncio.run(
        _drive([_event({"type": "input_audio_buffer.append", "audio": ""})])
    )
    events = _json_events(sent)
    assert events[-1]["error"]["code"] == "protocol_order"
    assert factory.opens == []
    assert sent[-1] == {"type": "websocket.close", "code": 1008}


# @spec ING-NIMWS-004
def test_deeply_invalid_session_field_type_is_structural() -> None:
    event = _session()
    event["session"]["input_audio_params"]["sample_rate_hz"] = "16000"  # type: ignore[index]
    sent, factory = asyncio.run(_drive([_event(event)]))
    error = _json_events(sent)[-1]["error"]
    assert error["code"] == "invalid_event"
    assert error["param"] == "input_audio_params.sample_rate_hz"
    assert factory.opens == []
    assert sent[-1] == {"type": "websocket.close", "code": 1008}


# @spec ING-NIMWS-006
def test_clear_is_fifo_local_and_does_not_rewind_an_already_fed_lease() -> None:
    audio = base64.b64encode(b"\x01\x00").decode()
    sent, factory = asyncio.run(
        _drive(
            [
                _event(_session()),
                _event({"type": "input_audio_buffer.append", "audio": audio}),
                _event({"type": "input_audio_buffer.clear"}),
                _event({"type": "input_audio_buffer.done"}),
            ]
        )
    )
    events = _json_events(sent)
    assert any(
        event["type"] == "input_audio_buffer.cleared" for event in events
    )
    assert factory.lease.calls[:1] == ["feed:1"]


# @spec ING-NIMWS-007, ING-FE-001
def test_invalid_format_fails_before_open() -> None:
    bad = _session(fmt="g711_ulaw")
    bad["session"]["input_audio_params"]["sample_rate_hz"] = 16000  # type: ignore[index]
    sent, factory = asyncio.run(_drive([_event(bad)]))
    assert _json_events(sent)[-1]["error"]["code"] == "unsupported_format"
    assert factory.opens == []


# @spec ING-NIMWS-009
def test_none_mode_resolves_riff_before_feeding_audio() -> None:
    fmt = (
        b"fmt "
        + (16).to_bytes(4, "little")
        + b"\x01\x00\x01\x00\x80>\x00\x00\x00\xfa\x00\x00\x02\x00\x10\x00"
    )
    wav = (
        b"RIFF"
        + (40).to_bytes(4, "little")
        + b"WAVE"
        + fmt
        + b"data\x02\x00\x00\x00\x01\x00"
    )
    sent, factory = asyncio.run(
        _drive(
            [
                _event(_session(fmt="none")),
                _event(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(wav).decode(),
                    }
                ),
                _event({"type": "input_audio_buffer.done"}),
            ]
        )
    )
    assert any(event["type"].endswith(".delta") for event in _json_events(sent))
    assert factory.lease.calls[0] == "feed:1"


# @spec ING-CORE-005
def test_pure_projection_uses_true_suffix() -> None:
    assert (
        project_event(
            {"type": "hypothesis", "previous": "good", "current": "good day"}
        )["delta"]
        == " day"
    )


# @spec ING-VEH-016, ING-VEH-019
def test_closed_host_admission_rejects_before_owner_creation() -> None:
    class Closed:
        def is_open(self) -> bool:
            return False

    async def case() -> tuple[list[dict[str, object]], _Factory]:
        sent: list[dict[str, object]] = []
        factory = _Factory()

        async def receive() -> dict[str, object]:
            return {"type": "websocket.disconnect"}

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        async def native(scope: object, receive: object, send: object) -> None:
            raise AssertionError("claimed NIM scope must not pass through")

        app = dispatch_nim_realtime(
            app=native,
            factory=factory,
            config=_Config(),
            admission=Closed(),
        )
        await app(_scope(b"intent=transcription"), receive, send)
        return sent, factory

    sent, factory = asyncio.run(case())
    assert _json_events(sent)[0]["error"]["code"] == "service_unavailable"
    assert sent[-1] == {"type": "websocket.close", "code": 1013}
    assert factory.opens == []


# @spec ING-NIMWS-002, ING-ERR-001
def test_structurally_invalid_non_text_event_errors_then_closes_1008() -> None:
    sent, _ = asyncio.run(
        _drive([{"type": "websocket.receive", "bytes": b"{}"}])
    )
    assert _json_events(sent)[-1]["error"]["code"] == "invalid_event"
    assert sent[-1] == {"type": "websocket.close", "code": 1008}


# @spec ING-NIMWS-008
def test_unknown_first_update_field_is_rejected_without_opening() -> None:
    event = _session()
    event["session"]["unrecognized"] = True  # type: ignore[index]
    sent, factory = asyncio.run(_drive([_event(event)]))
    assert _json_events(sent)[-1]["error"]["code"] == "invalid_config_field"
    assert factory.opens == []


# @spec ING-LIFE-014
def test_corrective_preconfiguration_events_do_not_extend_deadline() -> None:
    async def delayed_disconnect() -> tuple[list[dict[str, object]], _Factory]:
        queue = [
            _event(
                {
                    "type": "transcription_session.update",
                    "session": {"bad": True},
                }
            )
        ]
        sent: list[dict[str, object]] = []
        factory = _Factory()

        async def receive() -> dict[str, object]:
            if queue:
                return queue.pop(0)
            await asyncio.sleep(0.03)
            return {"type": "websocket.disconnect"}

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        async def native(scope: object, receive: object, send: object) -> None:
            raise AssertionError

        app = dispatch_nim_realtime(
            app=native,
            factory=factory,
            config=replace(_Config(), preconfiguration_timeout=0.01),
        )
        await app(_scope(b"intent=transcription"), receive, send)
        return sent, factory

    sent, factory = asyncio.run(delayed_disconnect())
    assert _json_events(sent)[-1]["error"]["code"] == "configuration_timeout"
    assert factory.opens == []


# @spec ING-LIFE-005
def test_non_audio_commit_does_not_extend_idle_deadline() -> None:
    async def delayed_disconnect() -> tuple[list[dict[str, object]], _Factory]:
        queue = [
            _event(_session()),
            _event({"type": "input_audio_buffer.commit"}),
        ]
        sent: list[dict[str, object]] = []
        factory = _Factory()

        async def receive() -> dict[str, object]:
            if queue:
                return queue.pop(0)
            await asyncio.sleep(0.03)
            return {"type": "websocket.disconnect"}

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        async def native(scope: object, receive: object, send: object) -> None:
            raise AssertionError

        app = dispatch_nim_realtime(
            app=native,
            factory=factory,
            config=replace(_Config(), session_idle_timeout=0.01),
        )
        await app(_scope(b"intent=transcription"), receive, send)
        return sent, factory

    sent, factory = asyncio.run(delayed_disconnect())
    assert _json_events(sent)[-1]["error"]["code"] == "idle_timeout"
    assert factory.lease.calls[-2:] == ["abort", "release"]


# @spec ING-ADM-006
def test_shared_gate_rejects_before_first_receive() -> None:
    async def case() -> list[dict[str, object]]:
        gate = LoadShedGate(1)
        held = await gate.register("nim_realtime_transcription")
        assert held is not None
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            raise AssertionError("rejected connection must not receive")

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        async def native(scope: object, receive: object, send: object) -> None:
            raise AssertionError

        app = dispatch_nim_realtime(
            app=native, factory=_Factory(), config=_Config(), gate=gate
        )
        await app(_scope(b"intent=transcription"), receive, send)
        await held.release()
        return sent

    sent = asyncio.run(case())
    assert _json_events(sent)[0]["error"]["code"] == "busy"


# @spec ING-VEH-017
def test_owner_registration_precedes_receive_and_releases() -> None:
    async def case() -> list[str]:
        order: list[str] = []

        class Tracking:
            async def release(self) -> None:
                order.append("release")

        async def register() -> Tracking:
            order.append("register")
            return Tracking()

        async def receive() -> dict[str, object]:
            assert order == ["register"]
            return {"type": "websocket.disconnect"}

        async def send(message: dict[str, object]) -> None:
            del message

        async def native(scope: object, receive: object, send: object) -> None:
            raise AssertionError

        app = dispatch_nim_realtime(
            app=native,
            factory=_Factory(),
            config=_Config(),
            gate=LoadShedGate(1),
            owner_register=register,
        )
        await app(_scope(b"intent=transcription"), receive, send)
        return order

    assert asyncio.run(case()) == ["register", "release"]


# @spec ING-LIFE-010, ING-LIFE-012, ING-VEH-017
def test_owner_task_cancellation_aborts_before_releasing_registration() -> None:
    async def case() -> tuple[list[str], int]:
        factory = _Factory()
        gate = LoadShedGate(1)
        queue = [_event(_session())]
        blocked = asyncio.Event()

        async def receive() -> dict[str, object]:
            if queue:
                return queue.pop(0)
            await blocked.wait()
            raise AssertionError("cancelled receive resumed")

        async def send(message: dict[str, object]) -> None:
            del message

        async def native(scope: object, receive: object, send: object) -> None:
            raise AssertionError

        app = dispatch_nim_realtime(
            app=native,
            factory=factory,
            config=_Config(),
            gate=gate,
        )
        task = asyncio.create_task(
            app(_scope(b"intent=transcription"), receive, send)
        )
        while not factory.opens:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return factory.lease.calls, gate.active

    calls, active = asyncio.run(case())
    assert calls == ["abort", "release"]
    assert active == 0


# @spec ING-LIFE-005
def test_max_duration_bounds_a_blocking_receive() -> None:
    async def case() -> list[dict[str, object]]:
        queue = [_event(_session())]
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            if queue:
                return queue.pop(0)
            await asyncio.sleep(0.03)
            return {"type": "websocket.disconnect"}

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        async def native(scope: object, receive: object, send: object) -> None:
            raise AssertionError

        app = dispatch_nim_realtime(
            app=native,
            factory=_Factory(),
            config=replace(
                _Config(), session_idle_timeout=1.0, max_session_duration=0.01
            ),
        )
        await app(_scope(b"intent=transcription"), receive, send)
        return sent

    assert (
        _json_events(asyncio.run(case()))[-1]["error"]["code"]
        == "request_timeout"
    )


# @spec ING-LIFE-013
def test_lifecycle_can_supply_lease_owner_factory() -> None:
    async def case() -> list[float]:
        calls: list[float] = []

        def factory(
            lease_factory: object, *, cleanup_timeout: float
        ) -> DirectLeaseOwner:
            calls.append(cleanup_timeout)
            return DirectLeaseOwner(
                lease_factory, cleanup_timeout=cleanup_timeout
            )  # type: ignore[arg-type]

        queue = [_event(_session()), {"type": "websocket.disconnect"}]

        async def receive() -> dict[str, object]:
            return queue.pop(0)

        async def send(message: dict[str, object]) -> None:
            del message

        async def native(scope: object, receive: object, send: object) -> None:
            raise AssertionError

        app = dispatch_nim_realtime(
            app=native,
            factory=_Factory(),
            config=_Config(),
            owner_factory=factory,
        )
        await app(_scope(b"intent=transcription"), receive, send)
        return calls

    assert asyncio.run(case()) == [1.0]


# @spec ING-LIFE-008, ING-LIFE-009
def test_later_immutable_rejection_precedes_honored_locale_ack() -> None:
    first = _session()
    later = _session()
    later["session"]["input_audio_format"] = "g711_ulaw"  # type: ignore[index]
    later["session"]["input_audio_transcription"]["language"] = "fr-FR"  # type: ignore[index]
    sent, factory = asyncio.run(_drive([_event(first), _event(later)]))
    events = _json_events(sent)
    assert events[2]["error"]["code"] == "config_change_rejected"
    assert events[3]["type"] == "transcription_session.updated"
    assert factory.lease.calls == ["locale:fr-FR", "abort", "release"]


# @spec ING-NIMWS-008
def test_initial_provenance_is_an_echo_key() -> None:
    event = _session()
    event["session"]["provenance"] = {"precision_policy": "fp32"}  # type: ignore[index]
    sent, factory = asyncio.run(_drive([_event(event)]))
    assert factory.opens == [("560ms", "auto")]
    assert _json_events(sent)[1]["type"] == "transcription_session.updated"

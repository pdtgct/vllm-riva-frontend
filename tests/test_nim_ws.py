"""GPU-free raw-ASGI tests for the NIM Realtime compatibility adapter."""

import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass, field, replace

import pytest

import vllm_riva_frontend.nim_ws as nim_ws
from vllm_riva_frontend.admission import LoadShedGate
from vllm_riva_frontend.config import (
    DeploymentMetadata,
    build_deployment_provenance,
)
from vllm_riva_frontend.lease import DirectLeaseOwner
from vllm_riva_frontend.nim_ws import (
    DispatchDecision,
    bootstrap_session,
    dispatch_decision,
    dispatch_nim_realtime,
    project_event,
)

#: See test_operational_plugin.py's identically-named constants; kept as a
#: local copy so this file's negative tests stand alone.
_FORBIDDEN_PROVENANCE_KEY = "selectedModelProfileId"
_HASH_SHAPED = re.compile(r"^[0-9A-Fa-f]{8,64}$")
_NGC_SCHEME = "ngc://"


def _walk(node: object) -> list[tuple[str, object]]:
    """Return every (key, value) pair reachable at any depth of a JSON tree."""
    pairs: list[tuple[str, object]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            pairs.append((key, value))
            pairs.extend(_walk(value))
    elif isinstance(node, list):
        for item in node:
            pairs.extend(_walk(item))
    return pairs


def _assert_no_nim_identity_leak(body: object) -> None:
    """Fail on a NIM registry/profile identity anywhere in a JSON tree."""
    for key, value in _walk(body):
        assert key != _FORBIDDEN_PROVENANCE_KEY, (
            f"forbidden key present at some depth: {key!r}"
        )
        if not isinstance(value, str):
            continue
        assert _FORBIDDEN_PROVENANCE_KEY not in value, (
            f"forbidden identifier present in a value: {value!r}"
        )
        assert _NGC_SCHEME not in value, (
            f"an ngc:// reference is present in a value: {value!r}"
        )
        assert not _HASH_SHAPED.fullmatch(value), (
            f"a NIM-profile-hash-shaped value is present: {value!r}"
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


class _HostAdmissionLease:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self._events.append("admission:release")


class _HostAdmission:
    def __init__(self, *, opened: bool, events: list[str]) -> None:
        self._opened = opened
        self._events = events
        self.leases: list[_HostAdmissionLease] = []

    def is_open(self) -> bool:
        return self._opened

    def try_acquire(self) -> _HostAdmissionLease | None:
        self._events.append("admission:try")
        if not self._opened:
            return None
        lease = _HostAdmissionLease(self._events)
        self.leases.append(lease)
        return lease


@dataclass(frozen=True)
class _Config:
    load_shed_max_sessions: int = 2
    session_cleanup_timeout: float = 1.0
    ws_event_envelope_max_bytes: int = 4096
    max_riff_header_bytes: int = 256
    preconfiguration_timeout: float = 1.0
    session_idle_timeout: float = 1.0
    max_session_duration: float | None = None
    provenance: dict[str, object] | None = None


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


def _private_connection(
    *,
    config: _Config | None = None,
) -> nim_ws._Connection:
    admission_lease = nim_ws.try_acquire_admission(None)
    assert admission_lease is not None
    return nim_ws._Connection(
        _Factory(),
        config or _ModelConfig(),
        LoadShedGate(2),
        None,
        lambda factory, *, cleanup_timeout: DirectLeaseOwner(
            factory, cleanup_timeout=cleanup_timeout
        ),
        admission_lease,
    )


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


# @spec ING-VEH-013, ING-NIMWS-004, ING-NIMWS-008
def test_accepted_alias_selector_still_opens_the_bound_lease() -> None:
    """An accepted alias must delegate, not vanish before the factory.

    Complements the rejection above: the exact-alias case must not be a
    silent no-op either -- it must reach the one bound session factory
    and the accepted identity must be the one echoed back to the client.
    """
    update = _session()
    session = update["session"]
    assert isinstance(session, dict)
    transcription = session["input_audio_transcription"]
    assert isinstance(transcription, dict)
    transcription["model"] = "nemotron"

    sent, factory = asyncio.run(_drive([_event(update)], config=_ModelConfig()))
    events = _json_events(sent)
    updated = next(
        event
        for event in events
        if event["type"] == "transcription_session.updated"
    )
    assert (
        updated["session"]["input_audio_transcription"]["model"] == "nemotron"
    )
    assert factory.opens == [("560ms", "auto")]


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


# @spec ING-SHIM-001, ING-NIMWS-004
def test_deployment_owned_session_provenance_never_leaks_nim_identity() -> None:
    """The server-authored provenance path cannot leak a NIM identity.

    ``transcription_session.updated`` echoes deployment-owned provenance
    the same as /v1/metadata does.  There is no fixture that reproduces a
    leak against the current design here either, because the config's
    ``provenance`` is always ``build_deployment_provenance``'s output --
    this pins the real construction path clean under the same
    list-aware, ngc://- and hash-aware walker used for /v1/metadata.
    Client-supplied ``session.provenance`` is a distinct, intentionally
    unfiltered path -- see test_initial_provenance_is_an_echo_key.
    """
    metadata = DeploymentMetadata(
        image="sha256:test",
        pin="vllm==0.24.0",
        precision_policy="nemotron-asr-fp32-v1",
    )
    provenance = build_deployment_provenance(
        metadata, resampler_identifier="scipy-poly-v1"
    )
    config = replace(_Config(), provenance=provenance)

    sent, factory = asyncio.run(_drive([_event(_session())], config=config))
    del factory
    updated = next(
        event
        for event in _json_events(sent)
        if event["type"] == "transcription_session.updated"
    )
    session = updated["session"]
    _assert_no_nim_identity_leak(session)
    assert session["provenance"] == provenance


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


# @spec ING-NIMWS-004, ING-NIMWS-008
def test_structural_validation_names_every_nested_type_failure() -> None:
    invalid = nim_ws._structural_session_fields(
        {
            "id": 1,
            "object": [],
            "provenance": "not-a-map",
            "modalities": ["text", 1],
            "input_audio_format": 3,
            "input_audio_params": {
                "sample_rate_hz": "16000",
                "num_channels": True,
            },
            "input_audio_transcription": {
                "language": 1,
                "model": [],
                "prompt": {},
            },
            "recognition_config": {
                "max_alternatives": "1",
                "enable_word_time_offsets": "false",
            },
            "speaker_diarization": {"enable_speaker_diarization": "false"},
            "word_boosting": {
                "enable_word_boosting": "false",
                "word_boosting_list": ["ok", 1],
            },
            "endpointing_config": [],
        }
    )

    assert set(invalid) == {
        "id",
        "object",
        "provenance",
        "modalities",
        "input_audio_format",
        "input_audio_params.sample_rate_hz",
        "input_audio_params.num_channels",
        "input_audio_transcription.language",
        "input_audio_transcription.model",
        "input_audio_transcription.prompt",
        "recognition_config.max_alternatives",
        "recognition_config.enable_word_time_offsets",
        "speaker_diarization.enable_speaker_diarization",
        "word_boosting.enable_word_boosting",
        "word_boosting.word_boosting_list",
        "endpointing_config",
    }


# @spec ING-NIMWS-004, ING-NIMWS-008
def test_semantic_validation_collects_all_unsupported_fields() -> None:
    rejected = nim_ws._session_rejections(
        {
            "unknown": True,
            "modalities": ["audio"],
            "input_audio_params": {"extra": 1},
            "input_audio_transcription": {
                "extra": True,
                "prompt": "bias me",
            },
            "speaker_diarization": {"enable_speaker_diarization": True},
            "word_boosting": {"enable_word_boosting": True},
            "endpointing_config": {"start_history": 1},
            "recognition_config": {
                "unknown": True,
                "max_alternatives": 2,
                "enable_word_time_offsets": True,
                "enable_profanity_filter": True,
            },
        }
    )

    assert ("invalid_config_field", "unknown") in rejected
    assert ("invalid_config_field", "modalities") in rejected
    assert (
        "unsupported_capability",
        "input_audio_transcription.prompt",
    ) in rejected
    assert ("unsupported_capability", "speaker_diarization") in rejected
    assert ("unsupported_capability", "word_boosting") in rejected
    assert ("unsupported_capability", "endpointing_config") in rejected
    assert (
        "invalid_config_field",
        "recognition_config.max_alternatives",
    ) in rejected
    assert (
        "unsupported_capability",
        "recognition_config.enable_word_time_offsets",
    ) in rejected


# @spec ING-CORE-005, ING-NIMWS-004, ING-NIMWS-005, ING-NIMWS-006
@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"type": "connect"}, "conversation.created"),
        (
            {"type": "input_audio_buffer.done"},
            "conversation.item.input_audio_transcription.completed",
        ),
        ({"type": "input_audio_buffer.clear"}, "input_audio_buffer.cleared"),
        (
            {"type": "transcription_session.update", "rate": "16000"},
            "error",
        ),
        (
            {
                "type": "transcription_session.update",
                "format": "g711_ulaw",
                "rate": 16000,
            },
            None,
        ),
        (
            {
                "type": "transcription_session.update",
                "locale": "en-US",
            },
            "transcription_session.updated",
        ),
        ({"type": "idle_timeout"}, None),
        ({"type": "unknown"}, "error"),
    ],
)
def test_projection_covers_every_sans_io_event_family(
    event: dict[str, object], expected: str | None
) -> None:
    projected = project_event(event)
    if expected is not None:
        assert projected["type"] == expected
    else:
        assert "code" in projected


# @spec ING-NIMWS-001, ING-NIMWS-002, ING-ERR-001
def test_bootstrap_query_and_error_helpers_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_session("")
    assert (
        dispatch_decision(
            {
                "type": "websocket",
                "path": "/v1/realtime",
                "query_string": "intent=transcription",
            }
        )
        is DispatchDecision.DENY
    )
    assert dispatch_decision(_scope(b"intent=%0")) is DispatchDecision.DENY
    assert dispatch_decision(_scope(b"intent=%ZZ")) is DispatchDecision.DENY
    assert nim_ws._suffix("old", "replacement") == "replacement"
    error = nim_ws._error("not-in-catalog", "bad", ["a", "b"])
    assert error["type"] == "error"
    assert error["error"]["param"] == "a,b"


# @spec ING-NIMWS-004, ING-NIMWS-008
def test_private_connection_update_edges_project_named_failures() -> None:
    async def exercise() -> list[dict[str, object]]:
        sent: list[dict[str, object]] = []

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        missing = _private_connection()
        await missing._update({"type": "transcription_session.update"}, send)

        unknown_locale = _private_connection()
        event = _session()
        event["session"]["input_audio_transcription"]["language"] = "xx-XX"  # type: ignore[index]
        await unknown_locale._update(event, send)

        configured = _private_connection()
        configured._configured = True
        configured._configured_at = time.monotonic()
        configured._last_accepted_audio_at = configured._configured_at
        configured._initial_session = bootstrap_session("nemotron")
        configured._owner = _Lease()  # type: ignore[assignment]
        await configured._update(_session(), send)

        locale_update = _session()
        locale_update["session"]["input_audio_transcription"]["language"] = (  # type: ignore[index]
            "en-US"
        )
        await configured._update(locale_update, send)

        immutable = _session()
        immutable["session"]["input_audio_params"]["sample_rate_hz"] = 8000  # type: ignore[index]
        await configured._update(immutable, send)

        missing_owner = _private_connection()
        missing_owner._configured = True
        missing_owner._initial_session = bootstrap_session("nemotron")
        await missing_owner._update(_session(), send)
        return sent

    events = _json_events(asyncio.run(exercise()))
    codes = [event["error"]["code"] for event in events if "error" in event]
    assert "invalid_event" in codes
    assert "unknown_locale" in codes
    assert "config_change_rejected" in codes
    assert "internal" in codes


# @spec ING-NIMWS-003, ING-NIMWS-004, ING-ERR-001
def test_private_connection_append_and_unknown_event_edges() -> None:
    async def exercise() -> list[dict[str, object]]:
        sent: list[dict[str, object]] = []

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        connection = _private_connection()
        connection._configured = True
        await connection._append(
            {"type": "input_audio_buffer.append", "audio": 7}, send
        )
        await connection._append(
            {"type": "input_audio_buffer.append", "audio": "%%%"}, send
        )
        await connection._handle({"type": "unknown"}, send)

        missing_audio_state = _private_connection()
        missing_audio_state._configured = True
        await missing_audio_state._append(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(b"\x00\x00").decode(),
            },
            send,
        )
        return sent

    events = _json_events(asyncio.run(exercise()))
    codes = [event["error"]["code"] for event in events if "error" in event]
    assert codes.count("invalid_audio") == 2
    assert "invalid_event" in codes
    assert "internal" in codes


# @spec ING-NIMWS-009, ING-FE-001
def test_private_connection_deferred_riff_rejects_declared_mismatch() -> None:
    async def exercise() -> list[dict[str, object]]:
        sent: list[dict[str, object]] = []

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        connection = _private_connection()
        await connection._resolve_riff(
            nim_ws.RiffFormat("LINEAR_PCM", 8000, 2, 44, 2),
            send,
        )
        return sent

    events = _json_events(asyncio.run(exercise()))
    assert events[-1]["error"]["code"] == "unsupported_format"
    assert (
        events[-1]["error"]["param"]
        == "input_audio_params.sample_rate_hz,input_audio_params.num_channels"
    )


# @spec ING-NIMWS-003, ING-NIMWS-004, ING-LIFE-010
def test_private_connection_terminal_failure_families() -> None:
    class NoneComplete:
        async def complete(self) -> None:
            return None

        async def cancel(self) -> None:
            return None

    class ErrorComplete(NoneComplete):
        async def complete(self) -> None:
            raise RuntimeError("complete failed")

    async def exercise() -> list[dict[str, object]]:
        sent: list[dict[str, object]] = []

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        unconfigured = _private_connection()
        await unconfigured._done(send)

        truncated = _private_connection()
        truncated._configured = True
        truncated._owner = NoneComplete()  # type: ignore[assignment]
        truncated._riff_remaining = 1
        await truncated._done(send)

        expired = _private_connection(
            config=replace(_Config(), max_session_duration=0.001)
        )
        expired._configured = True
        expired._owner = NoneComplete()  # type: ignore[assignment]
        expired._configured_at = time.monotonic() - 1
        await expired._done(send)

        missing_tail = _private_connection()
        missing_tail._configured = True
        missing_tail._owner = NoneComplete()  # type: ignore[assignment]
        missing_tail._configured_at = time.monotonic()
        await missing_tail._done(send)

        failed_tail = _private_connection()
        failed_tail._configured = True
        failed_tail._owner = ErrorComplete()  # type: ignore[assignment]
        failed_tail._configured_at = time.monotonic()
        await failed_tail._done(send)
        return sent

    events = _json_events(asyncio.run(exercise()))
    codes = [event["error"]["code"] for event in events if "error" in event]
    assert "protocol_order" in codes
    assert "invalid_audio" in codes
    assert "request_timeout" in codes
    assert codes.count("internal") == 2


# @spec ING-VEH-016, ING-VEH-019
def test_closed_host_admission_rejects_before_owner_creation() -> None:
    events: list[str] = []
    admission = _HostAdmission(opened=False, events=events)

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
            admission=admission,
        )
        await app(_scope(b"intent=transcription"), receive, send)
        return sent, factory

    sent, factory = asyncio.run(case())
    assert sent == [{"type": "websocket.close"}]
    assert events == ["admission:try"]
    assert admission.leases == []
    assert factory.opens == []


# @spec ING-VEH-019
def test_claimed_websocket_holds_host_admission_until_cleanup() -> None:
    events: list[str] = []
    admission = _HostAdmission(opened=True, events=events)
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
        admission=admission,
    )
    asyncio.run(app(_scope(b"intent=transcription"), receive, send))

    assert sent[0] == {"type": "websocket.accept"}
    assert events == ["admission:try", "admission:release"]
    assert len(admission.leases) == 1
    assert admission.leases[0].released


# @spec ING-VEH-019, ING-VEH-022
def test_websocket_accept_failure_cannot_leak_host_admission() -> None:
    events: list[str] = []
    admission = _HostAdmission(opened=True, events=events)

    async def receive() -> dict[str, object]:
        raise AssertionError

    async def send(message: dict[str, object]) -> None:
        assert message == {"type": "websocket.accept"}
        raise RuntimeError("socket failed")

    async def native(scope: object, receive: object, send: object) -> None:
        raise AssertionError("claimed NIM scope must not pass through")

    app = dispatch_nim_realtime(
        app=native,
        factory=_Factory(),
        config=_Config(),
        admission=admission,
    )
    with pytest.raises(RuntimeError, match="socket failed"):
        asyncio.run(app(_scope(b"intent=transcription"), receive, send))

    assert events == ["admission:try", "admission:release"]
    assert admission.leases[0].released


# @spec ING-VEH-019, ING-NIMWS-002
def test_passthrough_websocket_does_not_acquire_host_admission() -> None:
    events: list[str] = []
    admission = _HostAdmission(opened=True, events=events)
    native_scopes: list[object] = []

    async def receive() -> dict[str, object]:
        raise AssertionError

    async def send(message: dict[str, object]) -> None:
        raise AssertionError

    async def native(scope: object, receive: object, send: object) -> None:
        del receive, send
        native_scopes.append(scope)

    app = dispatch_nim_realtime(
        app=native,
        factory=_Factory(),
        config=_Config(),
        admission=admission,
    )
    asyncio.run(app(_scope(b""), receive, send))

    assert native_scopes == [_scope(b"")]
    assert events == []
    assert admission.leases == []


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
    updated = _json_events(sent)[1]
    assert updated["type"] == "transcription_session.updated"
    assert updated["session"]["provenance"] == {"precision_policy": "fp32"}


# @spec ING-NIMWS-008
def test_client_supplied_provenance_is_echoed_verbatim_not_filtered() -> None:
    """The echo-key contract, not a leak: this scope choice is deliberate.

    Only the deployment-owned provenance path (built exclusively by
    ``build_deployment_provenance``, see
    test_deployment_owned_session_provenance_never_leaks_nim_identity) is
    covered by the no-NIM-identity guarantee.  A client that echoes its
    own ``session.provenance`` -- including content that would be
    forbidden if this deployment had authored it -- gets it back
    unmodified, per ING-NIMWS-008; rewriting a client's own session
    field would break the documented echo contract and canonical-client
    round-trip conformance.
    """
    event = _session()
    client_supplied = {
        "selectedModelProfileId": "client-owns-this-value",
        "modelUrl": "ngc://nim/model",
        "profileHash": "deadbeef",
    }
    event["session"]["provenance"] = client_supplied  # type: ignore[index]

    sent, factory = asyncio.run(_drive([_event(event)]))

    assert factory.opens == [("560ms", "auto")]
    updated = _json_events(sent)[1]
    assert updated["type"] == "transcription_session.updated"
    assert updated["session"]["provenance"] == client_supplied

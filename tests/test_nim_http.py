"""Speech NIM HTTP tests over raw ASGI receive chunks and direct leases."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import vllm_riva_frontend.nim_http as nim_http
from vllm_riva_frontend.admission import LoadShedGate
from vllm_riva_frontend.lease import DirectLeaseOwner, SessionFactory
from vllm_riva_frontend.nim_http import (
    HttpFailure,
    HttpTranscriptionConfig,
    MultipartLimits,
    NimHttpTranscriptionEndpoint,
    classify_request_limit,
    parse_transcription_multipart,
)


class Receive:
    def __init__(
        self, chunks: list[bytes], *, disconnect: bool = False
    ) -> None:
        self.messages = [
            {
                "type": "http.request",
                "body": chunk,
                "more_body": index + 1 < len(chunks),
            }
            for index, chunk in enumerate(chunks)
        ]
        if disconnect:
            self.messages[-1] = {"type": "http.disconnect"}

    async def __call__(self) -> dict[str, object]:
        return self.messages.pop(0)


class Send:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def __call__(self, message: dict[str, object]) -> None:
        self.messages.append(message)


class Lease:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.samples: list[object] = []

    async def feed(self, samples, *, on_accepted):
        self.events.append("feed")
        self.samples.append(samples)
        on_accepted(len(samples))
        return []

    async def flush(self) -> str:
        self.events.append("flush")
        return "transcript"

    async def finish(self) -> None:
        self.events.append("finish")

    async def release(self) -> None:
        self.events.append("release")

    async def abort(self) -> None:
        self.events.append("abort")

    async def update_locale(self, locale: str) -> None:
        del locale


class Factory:
    def __init__(self, lease: Lease | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.lease = lease or Lease()

    async def open(self, *, cadence: str, locale: str) -> Lease:
        self.calls.append((cadence, locale))
        return self.lease


def _limits() -> MultipartLimits:
    return MultipartLimits(4096, 4608, 256, 64, 4, 256, 128)


def _headers(boundary: bytes = b"x") -> list[tuple[bytes, bytes]]:
    return [(b"content-type", b"multipart/form-data; boundary=" + boundary)]


def _multipart(parts: list[tuple[str, bytes]], boundary: bytes = b"x") -> bytes:
    chunks: list[bytes] = []
    for name, value in parts:
        chunks.extend(
            [
                b"--" + boundary + b"\r\n",
                b'Content-Disposition: form-data; name="'
                + name.encode()
                + b'"\r\n\r\n',
                value,
                b"\r\n",
            ]
        )
    chunks.append(b"--" + boundary + b"--\r\n")
    return b"".join(chunks)


def _wav(samples: bytes = b"\x00\x00\x01\x00") -> bytes:
    fmt = (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
    fmt += (16000).to_bytes(4, "little") + (32000).to_bytes(4, "little")
    fmt += (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
    payload = b"WAVEfmt " + (16).to_bytes(4, "little") + fmt
    payload += b"data" + len(samples).to_bytes(4, "little") + samples
    return b"RIFF" + (len(payload) + 4).to_bytes(4, "little") + payload


def _config() -> HttpTranscriptionConfig:
    return HttpTranscriptionConfig(_limits(), 256, 10.0, 16000, 1.0, 1.0, 1.0)


def _endpoint(
    factory: Factory,
    *,
    max_sessions: int = 1,
    admission=None,
    owner_register=None,
    owner_factory=DirectLeaseOwner,
    config: HttpTranscriptionConfig | None = None,
) -> NimHttpTranscriptionEndpoint:
    return NimHttpTranscriptionEndpoint(
        factory=factory,
        load_shed=LoadShedGate(max_sessions),
        config=config or _config(),
        served_model="nemotron",
        locales=frozenset({"en-US"}),
        admission=admission,
        owner_register=owner_register,
        owner_factory=owner_factory,
    )


# @spec ING-HTTP-004, ING-HTTP-007, ING-HTTP-011
def test_byte_fragmentation_is_raw_asgi_equivalent() -> None:
    body = _multipart([("file", b"abc"), ("model", b"nemotron")])
    parsed = asyncio.run(
        parse_transcription_multipart(
            receive=Receive(
                [body[index : index + 1] for index in range(len(body))]
            ),
            headers=_headers(),
            limits=_limits(),
        )
    )
    assert parsed == {"file": b"abc", "model": "nemotron"}


# @spec ING-HTTP-005, ING-HTTP-007, ING-HTTP-010
def test_multipart_structure_disconnect_and_content_length_fail_closed() -> (
    None
):
    duplicate = _multipart([("file", b"a"), ("file", b"b")])
    duplicate_result = asyncio.run(
        parse_transcription_multipart(
            receive=Receive([duplicate]), headers=_headers(), limits=_limits()
        )
    )
    assert isinstance(duplicate_result, HttpFailure)
    assert duplicate_result.status == 400
    assert duplicate_result.body["error"]["message"].startswith(
        "malformed_request:"
    )

    disconnected = asyncio.run(
        parse_transcription_multipart(
            receive=Receive([b"--x\r\n"], disconnect=True),
            headers=_headers(),
            limits=_limits(),
        )
    )
    assert isinstance(disconnected, HttpFailure)
    assert disconnected.status == 400

    overlong = asyncio.run(
        parse_transcription_multipart(
            receive=Receive([]),
            headers=_headers() + [(b"content-length", b"99999")],
            limits=_limits(),
        )
    )
    assert isinstance(overlong, HttpFailure)
    assert overlong.status == 413

    mismatched = asyncio.run(
        parse_transcription_multipart(
            receive=Receive([_multipart([("file", b"a")])]),
            headers=_headers() + [(b"content-length", b"1")],
            limits=_limits(),
        )
    )
    assert isinstance(mismatched, HttpFailure)
    assert mismatched.status == 400


# @spec ING-HTTP-005, ING-HTTP-007, ING-HTTP-010
def test_multipart_limits_are_413_audio_limit_is_400() -> None:
    boundary_failure = asyncio.run(
        parse_transcription_multipart(
            receive=Receive([]),
            headers=_headers(b"x" * 65),
            limits=_limits(),
        )
    )
    assert isinstance(boundary_failure, HttpFailure)
    assert boundary_failure.status == 413
    assert boundary_failure.body["error"]["type"] == "RequestTooLargeError"

    semantic = classify_request_limit(
        encoded_audio_bytes=4097,
        multipart_envelope_bytes=4097,
        limits=_limits(),
    )
    assert semantic == HttpFailure(
        400, {"detail": "request_too_large: audio too long"}
    )


# @spec ING-HTTP-001, ING-HTTP-002, ING-HTTP-003, ING-HTTP-008, ING-HTTP-011
def test_endpoint_drives_one_direct_lease_to_release() -> None:
    factory = Factory()
    owner_events: list[str] = []
    endpoint: NimHttpTranscriptionEndpoint

    class Tracking:
        async def release(self) -> None:
            assert factory.lease.events[-3:] == ["flush", "finish", "release"]
            assert endpoint._load_shed.active == 1
            owner_events.append("release")

    async def register() -> Tracking:
        owner_events.append("register")
        return Tracking()

    endpoint = _endpoint(factory, owner_register=register)
    body = _multipart(
        [("file", _wav()), ("language", b"en-US"), ("response_format", b"json")]
    )
    response = asyncio.run(
        endpoint.handle(
            scope={"type": "http", "method": "POST", "headers": _headers()},
            receive=Receive([body]),
        )
    )

    assert response.status == 200
    assert response.content_type == b"application/json"
    assert json.loads(response.body) == {"text": "transcript"}
    assert factory.calls == [("1120ms", "en-US")]
    assert factory.lease.events[-3:] == ["flush", "finish", "release"]
    assert "abort" not in factory.lease.events
    assert owner_events == ["register", "release"]


# @spec ING-HTTP-001, ING-HTTP-009
def test_raw_asgi_mount_writes_one_complete_response() -> None:
    factory = Factory()
    endpoint = _endpoint(factory)
    body = _multipart([("file", _wav()), ("model", b"nemotron")])
    send = Send()

    asyncio.run(
        endpoint(
            {"type": "http", "method": "POST", "headers": _headers()},
            Receive([body]),
            send,
        )
    )

    assert send.messages == [
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        },
        {
            "type": "http.response.body",
            "body": b'{"text": "transcript"}',
            "more_body": False,
        },
    ]


# @spec ING-LIFE-013
def test_injected_owner_factory_receives_the_cleanup_bound() -> None:
    factory = Factory()
    constructed: list[tuple[SessionFactory, float]] = []

    def owner_factory(
        session_factory: SessionFactory, *, cleanup_timeout: float
    ) -> DirectLeaseOwner:
        constructed.append((session_factory, cleanup_timeout))
        return DirectLeaseOwner(
            session_factory, cleanup_timeout=cleanup_timeout
        )

    body = _multipart([("file", _wav()), ("model", b"nemotron")])
    response = asyncio.run(
        _endpoint(factory, owner_factory=owner_factory).handle(
            scope={"type": "http", "method": "POST", "headers": _headers()},
            receive=Receive([body]),
        )
    )

    assert response.status == 200
    assert constructed == [(factory, 1.0)]


# @spec ING-VEH-017
def test_owner_registration_precedes_http_body_and_releases() -> None:
    order: list[str] = []

    class Tracking:
        async def release(self) -> None:
            order.append("release")

    async def register() -> Tracking:
        order.append("register")
        return Tracking()

    async def receive() -> dict[str, object]:
        assert order == ["register"]
        return {"type": "http.disconnect"}

    response = asyncio.run(
        _endpoint(Factory(), owner_register=register).handle(
            scope={"type": "http", "method": "POST", "headers": _headers()},
            receive=receive,
        )
    )

    assert response.status == 400
    assert order == ["register", "release"]


# @spec ING-VEH-017, ING-VEH-019, ING-LIFE-006
def test_not_ready_rejection_does_not_register_an_owner() -> None:
    class ClosedAdmission:
        def is_open(self) -> bool:
            return False

    async def register() -> object:
        raise AssertionError("not-ready request must not register an owner")

    response = asyncio.run(
        _endpoint(
            Factory(), admission=ClosedAdmission(), owner_register=register
        ).handle(
            scope={"type": "http", "method": "POST", "headers": _headers()},
            receive=Receive([]),
        )
    )

    assert response.status == 503


# @spec ING-FE-005, ING-FE-006, ING-VEH-004
def test_pre_submit_cap_splits_complete_audio_without_cadence_math(
    monkeypatch,
) -> None:
    frontend_events: list[str] = []

    class MultiPieceFrontend:
        def __init__(self, *, encoding: str, sample_rate_hz: int) -> None:
            del encoding, sample_rate_hz

        def push(self, data: bytes) -> list[float]:
            del data
            frontend_events.append("push")
            return [0.0, 1.0, 2.0, 3.0, 4.0]

        def flush(self) -> list[float]:
            frontend_events.append("flush")
            return []

    class ValidatedFactory(Factory):
        async def open(self, *, cadence: str, locale: str) -> Lease:
            assert frontend_events == ["push", "flush"]
            return await super().open(cadence=cadence, locale=locale)

    monkeypatch.setattr(nim_http, "StreamingAudioFrontend", MultiPieceFrontend)
    factory = ValidatedFactory()
    body = _multipart([("file", _wav()), ("model", b"nemotron")])
    response = asyncio.run(
        _endpoint(
            factory,
            config=replace(_config(), pre_submit_max_samples=2),
        ).handle(
            scope={"type": "http", "method": "POST", "headers": _headers()},
            receive=Receive([body]),
        )
    )

    assert response.status == 200
    assert factory.calls == [("1120ms", "auto")]
    assert factory.lease.samples == [
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0],
    ]
    assert factory.lease.events == [
        "feed",
        "feed",
        "feed",
        "flush",
        "finish",
        "release",
    ]


# @spec ING-VEH-004
def test_invalid_acceptance_credit_is_internal_and_aborts_cleanly(
    monkeypatch,
) -> None:
    class TwoPieceFrontend:
        def __init__(self, *, encoding: str, sample_rate_hz: int) -> None:
            del encoding, sample_rate_hz

        def push(self, data: bytes) -> list[float]:
            del data
            return [0.0, 0.0]

        def flush(self) -> list[float]:
            return [0.0, 0.0]

    class PartialLease(Lease):
        async def feed(self, samples, *, on_accepted):
            self.events.append(f"feed:{len(samples)}")
            on_accepted(1)
            return []

    monkeypatch.setattr(nim_http, "StreamingAudioFrontend", TwoPieceFrontend)
    factory = Factory(PartialLease())
    body = _multipart([("file", _wav()), ("model", b"nemotron")])
    response = asyncio.run(
        _endpoint(
            factory,
            config=replace(_config(), pre_submit_max_samples=2),
        ).handle(
            scope={"type": "http", "method": "POST", "headers": _headers()},
            receive=Receive([body]),
        )
    )

    assert response.status == 500
    assert factory.calls == [("1120ms", "auto")]
    assert factory.lease.events == ["feed:2", "abort", "release"]


# @spec ING-HTTP-008, ING-LIFE-012, ING-VEH-017
def test_request_timeout_cancels_owner_and_releases_lifecycle_token() -> None:
    class BlockingLease(Lease):
        async def feed(self, samples, *, on_accepted):
            del samples, on_accepted
            self.events.append("feed")
            await asyncio.Event().wait()
            return []

    owner_events: list[str] = []

    class Tracking:
        async def release(self) -> None:
            owner_events.append("release")

    async def register() -> Tracking:
        owner_events.append("register")
        return Tracking()

    factory = Factory(BlockingLease())
    body = _multipart([("file", _wav()), ("model", b"nemotron")])
    response = asyncio.run(
        _endpoint(
            factory,
            owner_register=register,
            config=replace(_config(), request_timeout=0.05),
        ).handle(
            scope={"type": "http", "method": "POST", "headers": _headers()},
            receive=Receive([body]),
        )
    )

    assert response.status == 504
    assert json.loads(response.body) == {
        "detail": "request_timeout: HTTP request timed out"
    }
    assert factory.lease.events == ["feed", "abort", "release"]
    assert owner_events == ["register", "release"]


# @spec ING-HTTP-002, ING-HTTP-008, ING-HTTP-010
def test_invalid_selector_or_format_never_opens_a_lease() -> None:
    factory = Factory()
    endpoint = _endpoint(factory)
    body = _multipart([("file", _wav()), ("language", b"auto")])
    response = asyncio.run(
        endpoint.handle(
            scope={"type": "http", "method": "POST", "headers": _headers()},
            receive=Receive([body]),
        )
    )

    assert response.status == 400
    assert json.loads(response.body) == {"detail": "unknown_locale: language"}
    assert factory.calls == []


# @spec ING-VEH-013, ING-HTTP-002, ING-HTTP-010
def test_canonical_checkpoint_is_rejected_when_http_alias_is_set() -> None:
    factory = Factory()
    body = _multipart([("file", _wav()), ("model", b"nvidia/nemotron-asr")])
    response = asyncio.run(
        _endpoint(factory).handle(
            scope={"type": "http", "method": "POST", "headers": _headers()},
            receive=Receive([body]),
        )
    )

    assert response.status == 400
    assert json.loads(response.body) == {
        "detail": "invalid_config_field: model"
    }
    assert factory.calls == []


# @spec ING-ADM-006, ING-HTTP-010
def test_load_shed_rejects_before_body_read_or_lease_open() -> None:
    factory = Factory()
    owner_register_calls = 0

    async def register_owner() -> object:
        nonlocal owner_register_calls
        owner_register_calls += 1
        raise AssertionError("rejected request must not register an owner")

    endpoint = _endpoint(factory, max_sessions=1, owner_register=register_owner)
    registration = asyncio.run(
        endpoint._load_shed.register("nim_http_transcription")
    )
    body = _multipart([("file", _wav()), ("model", b"nemotron")])
    receive = Receive([body])
    response = asyncio.run(
        endpoint.handle(
            scope={"type": "http", "method": "POST", "headers": _headers()},
            receive=receive,
        )
    )

    assert response.status == 503
    assert receive.messages
    assert factory.calls == []
    assert owner_register_calls == 0
    asyncio.run(registration.release())

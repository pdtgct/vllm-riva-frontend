"""Bounded in-memory multipart parsing and the route's ingress steps."""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import tracemalloc

import pytest
from support.fake_transcription import Raise
from support.http import (
    BOUNDARY,
    CORE_ROUTE,
    HttpExchange,
    MiB,
    assert_core_error,
    chunked,
    content_type,
    multipart_body,
    until,
)


def _limits(**overrides):
    from vllm_riva_frontend.multipart import MultipartLimits

    values = {
        "envelope_max_bytes": 4096,
        "content_type_max_bytes": 256,
        "boundary_max_bytes": 64,
        "max_parts": 8,
        "max_header_bytes": 256,
        "text_field_max_bytes": 128,
    }
    values.update(overrides)
    return MultipartLimits(**values)


def _receive(chunks, *, disconnect_at=None, counter=None):
    """An ASGI ``receive`` over ``chunks``; ``disconnect_at`` cuts in."""
    queue = list(chunks)
    state = {"index": 0}

    async def receive():
        index = state["index"]
        state["index"] += 1
        if counter is not None:
            counter.append(index)
        if disconnect_at is not None and index >= disconnect_at:
            return {"type": "http.disconnect"}
        if index >= len(queue):
            return {"type": "http.disconnect"}
        return {
            "type": "http.request",
            "body": queue[index],
            "more_body": index + 1 < len(queue),
        }

    return receive


async def _parse(chunks, limits=None, *, clock, boundary=BOUNDARY, **kwargs):
    from vllm_riva_frontend.multipart import parse_multipart

    return await parse_multipart(
        _receive(chunks, **kwargs),
        boundary,
        limits or _limits(),
        deadline=clock.monotonic() + 60,
        clock=clock,
    )


def _code(raised) -> str:
    return raised.value.code


SAMPLE = multipart_body(
    [("model", "nemotron"), ("language", "en")], file=b"\x00\x01abc\r\n--x"
)


# ---------------------------------------------------- fragmentation units


# @spec ING-HTTP-007, ING-FILE-004
async def test_every_fragmentation_parses_equivalently(clock):
    whole = await _parse([SAMPLE], clock=clock)
    assert whole.file == b"\x00\x01abc\r\n--x"
    assert whole.fields == [("model", "nemotron"), ("language", "en")]
    assert whole.filename == "audio.wav"
    assert whole.content_type == "audio/wav"
    for size in (1, 2, 3, 7, 13, len(BOUNDARY) + 1):
        parsed = await _parse(chunked(SAMPLE, size), clock=clock)
        assert parsed.fields == whole.fields, size
        assert bytes(parsed.file) == bytes(whole.file), size


# @spec ING-HTTP-007
async def test_split_inside_every_boundary_position_parses_equivalently(
    clock,
):
    whole = await _parse([SAMPLE], clock=clock)
    marker = SAMPLE.find(b"--" + BOUNDARY, 1)
    for cut in range(marker, marker + len(BOUNDARY) + 4):
        parsed = await _parse([SAMPLE[:cut], SAMPLE[cut:]], clock=clock)
        assert parsed.fields == whole.fields, cut
        assert bytes(parsed.file) == bytes(whole.file), cut


# @spec ING-HTTP-007, ING-FILE-039
@pytest.mark.parametrize(
    ("limits", "body"),
    [
        ({"max_parts": 2}, multipart_body([("model", "a"), ("language", "b")])),
        ({"text_field_max_bytes": 8}, multipart_body([("model", "a" * 9)])),
        ({"envelope_max_bytes": 64}, multipart_body([], file=b"x" * 128)),
        (
            {"max_header_bytes": 40},
            multipart_body([], filename="f" * 64 + ".wav"),
        ),
    ],
    ids=["parts", "text-field", "envelope", "part-headers"],
)
async def test_bounds_answer_request_too_large(clock, limits, body):
    from vllm_riva_frontend.errors import FrontendError

    with pytest.raises(FrontendError) as raised:
        await _parse(chunked(body, 5), _limits(**limits), clock=clock)
    assert _code(raised) == "request_too_large"


# @spec ING-HTTP-007, ING-FILE-037
async def test_body_bound_is_enforced_while_reading(clock):
    from vllm_riva_frontend.errors import FrontendError

    body = multipart_body([], file=b"y" * 64_000)
    chunks = chunked(body, 512)
    reads: list[int] = []
    with pytest.raises(FrontendError) as raised:
        await _parse(
            chunks, _limits(envelope_max_bytes=4096), clock=clock, counter=reads
        )
    assert _code(raised) == "request_too_large"
    # The parser stopped once the running count passed the bound: it did
    # not read (or hold) the whole encoded body first.
    assert len(reads) <= 4096 // 512 + 2 < len(chunks)


# @spec ING-HTTP-007, ING-FILE-005
@pytest.mark.parametrize(
    "body",
    [
        SAMPLE[:-10],  # truncated: no closing boundary
        b"garbage without any boundary",
        multipart_body([("model", "a")], file=None),  # missing file
        multipart_body([], file=b"a")[: -len(b"--" + BOUNDARY + b"--\r\n")]
        + multipart_body([], file=b"b"),  # second file part
    ],
    ids=["truncated", "no-framing", "missing-file", "second-file"],
)
async def test_malformed_framing_never_succeeds(clock, body):
    from vllm_riva_frontend.errors import FrontendError

    with pytest.raises(FrontendError) as raised:
        await _parse(chunked(body, 9), clock=clock)
    assert _code(raised) == "malformed_request"


# @spec ING-HTTP-007
async def test_disconnect_mid_body_never_becomes_success(clock):
    from vllm_riva_frontend.multipart import parse_multipart

    receive = _receive(chunked(SAMPLE, 8), disconnect_at=3)
    with pytest.raises(Exception) as raised:
        await parse_multipart(
            receive,
            BOUNDARY,
            _limits(),
            deadline=clock.monotonic() + 60,
            clock=clock,
        )
    assert not isinstance(raised.value, (AssertionError, ImportError))


# @spec ING-FILE-038
async def test_body_read_deadline_answers_request_timeout(clock):
    from vllm_riva_frontend.errors import FrontendError
    from vllm_riva_frontend.multipart import parse_multipart

    stall = asyncio.Event()
    first = SAMPLE[:20]

    sent = {"n": 0}

    async def receive():
        sent["n"] += 1
        if sent["n"] == 1:
            return {"type": "http.request", "body": first, "more_body": True}
        await stall.wait()
        return {"type": "http.disconnect"}

    task = asyncio.ensure_future(
        parse_multipart(
            receive,
            BOUNDARY,
            _limits(),
            deadline=clock.monotonic() + 120,
            clock=clock,
        )
    )
    await clock.advance(119)
    assert not task.done()
    await clock.advance(2)
    with pytest.raises(FrontendError) as raised:
        await asyncio.wait_for(task, 1)
    assert _code(raised) == "request_timeout"


# @spec ING-FILE-041
async def test_upload_object_is_in_memory_and_not_a_copy(clock):
    from vllm_riva_frontend.multipart import make_upload

    payload = os.urandom(8 * MiB)
    form = await _parse(
        [multipart_body([], file=payload)],
        _limits(envelope_max_bytes=16 * MiB),
        clock=clock,
    )
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        upload = make_upload(form)
        grown = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert isinstance(upload.file, io.BytesIO)
    assert grown < MiB, f"make_upload copied the part ({grown} bytes)"
    assert await upload.read() == payload


# ---------------------------------------------------------- route ingress


# @spec ING-FILE-005
@pytest.mark.parametrize(
    "value",
    [
        "application/json",
        "application/x-www-form-urlencoded",
        "multipart/form-data",
        "multipart/form-data; charset=utf-8",
        "multipart/form-data; boundary=a; boundary=b",
        "multipart/form-data; boundary=bad space",
        "multipart/form-data; boundary=",
        "multipart/form-data; boundary=café",
        "multipart/form-data; malformed",
    ],
)
async def test_bad_content_type_is_malformed_before_any_read(files_env, value):
    env = await files_env()
    exchange = await env.request(headers=[("content-type", value)]).finish()
    assert_core_error(exchange, 400, "malformed_request")
    assert exchange.chunks_delivered == 0
    assert env.calls == []


# @spec ING-FILE-005
async def test_quoted_boundary_is_accepted(files_env):
    env = await files_env()
    exchange = await env.request(
        headers=[
            (
                "content-type",
                f'multipart/form-data; boundary="{BOUNDARY.decode()}"',
            )
        ]
    ).finish()
    assert exchange.status == 200, exchange.body
    assert len(env.calls) == 1


# @spec ING-FILE-005
@pytest.mark.parametrize(
    "body",
    [
        multipart_body([("model", "a")], file=None),
        multipart_body([], file=b"a")[: -len(b"--" + BOUNDARY + b"--\r\n")]
        + multipart_body([], file=b"b"),
        multipart_body([], file=b"abc")[:-12],
    ],
    ids=["missing-file", "second-file", "truncated"],
)
async def test_bad_structure_is_malformed_on_the_route(files_env, body):
    env = await files_env()
    exchange = await env.request(body=body).finish()
    assert_core_error(exchange, 400, "malformed_request")
    assert env.calls == []


# @spec ING-FILE-039, ING-HTTP-007, ING-FILE-055
async def test_content_length_above_envelope_rejects_before_reading(
    files_env,
):
    env = await files_env()
    too_big = env.plugin.config.http_multipart_envelope_max_bytes + 1
    exchange = HttpExchange(
        env.app,
        [("content-type", content_type()), ("content-length", str(too_big))],
        [b"x"],
    )
    await exchange.finish()
    assert_core_error(exchange, 413, "request_too_large")
    assert exchange.chunks_delivered == 0
    assert env.calls == []


# @spec ING-FILE-039, ING-HTTP-007
async def test_chunked_body_above_envelope_is_413(files_env):
    env = await files_env(
        riva_frontend={"http_multipart_envelope_max_bytes": 25 * MiB + 262_144}
    )
    body = multipart_body([], file=b"z" * (25 * MiB + 300_000))
    exchange = await env.request(body=body, with_length=False).finish(
        timeout=30
    )
    assert_core_error(exchange, 413, "request_too_large")
    assert env.calls == []


# @spec ING-HTTP-007
async def test_missing_content_length_is_supported(files_env):
    env = await files_env()
    exchange = await env.request(with_length=False, chunk_size=7).finish()
    assert exchange.status == 200, exchange.body
    assert env.calls[0].audio_data == b"RIFF-not-really-audio"


# @spec ING-FILE-039
async def test_oversized_content_type_and_boundary_are_413(files_env):
    env = await files_env()
    limit = env.plugin.config.http_content_type_max_bytes
    padded = content_type() + "; x=" + "p" * limit
    exchange = await env.request(headers=[("content-type", padded)]).finish()
    assert_core_error(exchange, 413, "request_too_large")
    assert exchange.chunks_delivered == 0

    long_boundary = b"b" * (
        env.plugin.config.http_multipart_boundary_max_bytes + 1
    )
    exchange = await env.request(
        headers=[("content-type", content_type(long_boundary))],
        body=multipart_body([], boundary=long_boundary),
    ).finish()
    assert_core_error(exchange, 413, "request_too_large")
    assert exchange.chunks_delivered == 0
    assert env.calls == []


# @spec ING-FILE-038, ING-ADM-016
async def test_stalled_upload_answers_408(files_env):
    env = await files_env()
    exchange = env.request(chunk_size=16, stall_before=2).start()
    await until(lambda: exchange.chunks_delivered == 2, what="two chunks")
    await env.clock.advance(env.plugin.config.http_body_read_timeout - 1)
    assert exchange.status is None
    await env.clock.advance(2)
    await exchange.finish()
    assert_core_error(exchange, 408, "request_timeout")
    assert env.calls == []
    await until(lambda: not env.plugin.registry.records, what="unregistered")


# @spec ING-FILE-037
async def test_route_reads_incrementally_and_stops_at_the_bound(files_env):
    env = await files_env()
    envelope = env.plugin.config.http_multipart_envelope_max_bytes
    body = multipart_body([], file=b"q" * (envelope + 4 * MiB))
    exchange = await env.request(body=body, with_length=False).finish(
        timeout=30
    )
    assert_core_error(exchange, 413, "request_too_large")
    assert exchange.chunks_delivered < len(exchange.chunks)


# @spec ING-FILE-011
async def test_core_upload_limit_gives_core_error(files_env, monkeypatch):
    from support.app import build_app, init_state
    from support.fake_engine import FakeEngineClient

    monkeypatch.setenv("VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", "1")
    env = await files_env()
    oversized = b"o" * (MiB + 1)
    exchange = await env.request(file=oversized).finish(timeout=10)
    assert env.calls == []

    # Core's own route answers the same request with the same status and
    # body (core's read_upload_with_limit error through core's handlers).
    reference = build_app([], supported_tasks=("transcription",))
    await init_state(reference, FakeEngineClient())
    expected = HttpExchange(
        reference,
        [("content-type", content_type())],
        chunked(multipart_body([], file=oversized)),
        path=CORE_ROUTE,
    )
    await expected.finish(timeout=10)
    assert exchange.status == expected.status
    assert exchange.json() == expected.json()


# @spec ING-FILE-011
async def test_file_at_core_upload_limit_reaches_core(files_env, monkeypatch):
    monkeypatch.setenv("VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", "1")
    env = await files_env()
    exchange = await env.request(file=b"k" * MiB).finish(timeout=10)
    assert exchange.status == 200, exchange.body[:200]
    assert len(env.calls[0].audio_data) == MiB


# ------------------------------------------------------- no form, no spool


@pytest.fixture
def no_spool(monkeypatch, tmp_path):
    """Make every spooling helper fail and point temp files at ``tmp_path``."""
    import python_multipart
    import starlette.requests

    def forbidden(*args, **kwargs):
        raise AssertionError("the file route spooled or used a form parser")

    for name in (
        "SpooledTemporaryFile",
        "TemporaryFile",
        "NamedTemporaryFile",
        "mkstemp",
        "mkdtemp",
    ):
        monkeypatch.setattr(tempfile, name, forbidden)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(starlette.requests.Request, "form", forbidden)
    monkeypatch.setattr(python_multipart, "FormParser", forbidden)
    if hasattr(python_multipart, "multipart"):
        monkeypatch.setattr(
            python_multipart.multipart, "FormParser", forbidden, raising=False
        )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _written(root) -> list[str]:
    return sorted(
        os.path.join(d, f) for d, _, files in os.walk(root) for f in files
    )


# @spec ING-FILE-004, ING-HTTP-011
async def test_success_path_uses_no_form_parser_and_writes_nothing(
    files_env, no_spool
):
    env = await files_env()
    exchange = await env.request(
        [("model", env.models.model_name())],
        file=os.urandom(2 * MiB),
        chunk_size=1000,
    ).finish(timeout=10)
    assert exchange.status == 200, exchange.body[:200]
    assert _written(no_spool) == []


# @spec ING-FILE-004, ING-HTTP-011
@pytest.mark.parametrize(
    "filename",
    ["../../escape.wav", "/tmp/abs.wav", "con", "a\u0000b.wav"],
)
async def test_filename_is_untrusted_metadata(files_env, no_spool, filename):
    env = await files_env()
    body = multipart_body([], file=b"abc", filename=filename)
    exchange = await env.request(body=body).finish()
    assert exchange.status == 200, exchange.body
    assert _written(no_spool) == []
    assert not os.path.exists(os.path.join(no_spool, "escape.wav"))


# @spec ING-HTTP-011
async def test_rejection_disconnect_and_failure_paths_write_nothing(
    files_env, no_spool
):
    env = await files_env()
    # rejection (guard)
    await env.request([("temperature", "0.9")]).finish()
    # disconnect during the body read
    exchange = env.request(chunk_size=8, stall_before=3).start()
    await until(lambda: exchange.chunks_delivered == 3, what="partial body")
    exchange.disconnect()
    exchange.resume()
    await exchange.finish()
    assert exchange.status is None
    # failure raised by core
    env.service.script(Raise(RuntimeError("core failed")))
    await env.request().finish()
    assert _written(no_spool) == []


# @spec ING-FILE-007
async def test_disconnect_during_body_read_discards_and_calls_nothing(
    files_env,
):
    env = await files_env()
    exchange = env.request(chunk_size=8, stall_before=3).start()
    await until(lambda: exchange.chunks_delivered == 3, what="partial body")
    exchange.disconnect()
    exchange.resume()
    await exchange.finish()
    assert exchange.status is None
    assert env.calls == []
    assert env.engine.aborted == []
    await until(lambda: env.plugin.ingress.count == 0, what="released")
    assert not env.plugin.registry.records


def _raw_part(disposition: bytes) -> bytes:
    return (
        b"--"
        + BOUNDARY
        + b"\r\n"
        + disposition
        + b"\r\n\r\nvalue\r\n"
        + multipart_body([], file=b"abc")
    )


# @spec ING-FILE-005, ING-HTTP-007
@pytest.mark.parametrize(
    "disposition",
    [
        b"X-Other: 1",  # no Content-Disposition
        b'Content-Disposition: attachment; name="model"',
        b"Content-Disposition: form-data",
        b'Content-Disposition: form-data; name=""',
        b"Content-Disposition: \xff\xfe",
    ],
    ids=["missing", "attachment", "no-name", "empty-name", "non-ascii"],
)
async def test_bad_part_disposition_is_malformed(clock, disposition):
    from vllm_riva_frontend.errors import FrontendError

    with pytest.raises(FrontendError) as raised:
        await _parse(chunked(_raw_part(disposition), 11), clock=clock)
    assert _code(raised) == "malformed_request"

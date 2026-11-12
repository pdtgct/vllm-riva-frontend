"""The Riva Realtime WebSocket surface: route, bootstrap and adapter.

``RealtimeRoute`` is the ASGI app of the WebSocket route
``/plugins/riva/v1/realtime`` (``routes.REALTIME_PATH``) and
``bootstrap_endpoint`` the handler of the bootstrap ``POST``; ``attach_router``
registers both. The route serves only the exact query
``intent=transcription``, through ``RealtimeAdapter``, which translates the
dialect's events straight to one ``lease.StreamOwner`` and projects the
owner's results and its one terminal cause back as events and a close.

Event shapes follow the public client (``riva/client/realtime.py`` of
``nvidia-riva/python-clients``) and the OpenAI Realtime ``error`` object.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from types import MappingProxyType
from typing import Any, TypeVar
from urllib.parse import unquote_to_bytes

# Modules, not names: ``lifecycle`` imports this module while it loads.
from vllm_riva_frontend import admission, errors, frontend, lease, surfaces

T = TypeVar("T")
Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

UPDATE = "transcription_session.update"
APPEND = "input_audio_buffer.append"
COMMIT = "input_audio_buffer.commit"
CLEAR = "input_audio_buffer.clear"
DONE = "input_audio_buffer.done"

#: The client events of the dialect.
EVENT_TYPES = frozenset({UPDATE, APPEND, COMMIT, CLEAR, DONE})

_CREATED = "conversation.created"
_UPDATED = "transcription_session.updated"
_COMMITTED = "input_audio_buffer.committed"
_CLEARED = "input_audio_buffer.cleared"
_DELTA = "conversation.item.input_audio_transcription.delta"
_COMPLETED = "conversation.item.input_audio_transcription.completed"
_FAILED = "conversation.item.input_audio_transcription.failed"
_ERROR = "error"

#: Realtime audio names to the audio frontend's encodings.
_FORMATS = MappingProxyType(
    {"pcm16": "LINEAR_PCM", "g711_ulaw": "MULAW", "g711_alaw": "ALAW"}
)
#: The stock file-mode client's declaration of raw container bytes.
_DEFERRED = "none"
#: The audio leaf's neutral format fields by their Realtime names.
_REALTIME_FORMAT_FIELDS = MappingProxyType(
    {
        "encoding": "input_audio_format",
        "sample_rate_hz": "sample_rate_hz",
        "channels": "num_channels",
    }
)

_ECHO_KEYS = frozenset({"id", "object", "client_secret", "provenance"})
_SECTION_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "input_audio_params": frozenset({"sample_rate_hz", "num_channels"}),
        "input_audio_transcription": frozenset({"language", "model", "prompt"}),
        "recognition_config": frozenset(
            {
                "max_alternatives",
                "enable_automatic_punctuation",
                "enable_verbatim_transcripts",
                "enable_word_time_offsets",
                "enable_profanity_filter",
            }
        ),
        "speaker_diarization": frozenset(
            {"enable_speaker_diarization", "max_speaker_count"}
        ),
        "word_boosting": frozenset(
            {"enable_word_boosting", "word_boosting_list"}
        ),
        "endpointing_config": frozenset(
            {
                "start_history",
                "start_threshold",
                "stop_history",
                "stop_threshold",
                "stop_history_eou",
                "stop_threshold_eou",
            }
        ),
    }
)
_CONFIG_KEYS = frozenset(
    {"modalities", "input_audio_format", "custom_configuration", *_SECTION_KEYS}
)
_CAPABILITY_SECTIONS = (
    "speaker_diarization",
    "word_boosting",
    "endpointing_config",
)

#: Codes a terminal error event closes with when the catalog row has none.
_CLIENT_CODES = frozenset(
    {
        "invalid_config_field",
        "unsupported_capability",
        "unknown_locale",
        "unsupported_format",
        "malformed_request",
        "protocol_order",
        "invalid_event",
        "session_terminal",
    }
)
_SHUTDOWN_PHASES = frozenset({"shutdown", "drain"})


def _hex_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


# ==================================================================== route


def _strict_unquote(raw: bytes) -> str | None:
    """Percent-decode and UTF-8 decode, or ``None`` when either is malformed."""
    index = raw.find(b"%")
    while index != -1:
        escape = raw[index + 1 : index + 3]
        if len(escape) != 2 or not all(
            c in b"0123456789abcdefABCDEF" for c in escape
        ):
            return None
        index = raw.find(b"%", index + 3)
    try:
        return unquote_to_bytes(raw).decode("utf-8", "strict")
    except UnicodeDecodeError:
        return None


def _query_pairs(query: bytes) -> list[tuple[str, str | None]] | None:
    if not query:
        return []
    pairs: list[tuple[str, str | None]] = []
    for part in query.split(b"&"):
        key, separator, value = part.partition(b"=")
        name = _strict_unquote(key)
        text = _strict_unquote(value) if separator else ""
        if name is None or text is None:
            return None
        pairs.append((name, text if separator else None))
    return pairs


def _serves_query(scope: Mapping[str, Any]) -> bool:
    """Whether the query decodes strictly to just ``intent=transcription``."""
    query = scope.get("query_string", b"")
    if not isinstance(query, bytes | bytearray):
        return False
    return _query_pairs(bytes(query)) == [("intent", "transcription")]


class RealtimeRoute:
    """The ASGI app of the WebSocket route ``/plugins/riva/v1/realtime``.

    ``attach_router`` registers it under the plugin prefix, so it runs
    inside core's middleware (WebSocket metrics included); core's own
    ``/v1/realtime`` route is not touched.
    """

    def __init__(self, plugin: Any) -> None:
        """Bind the route to ``plugin``."""
        self.plugin = plugin

    # @spec ING-NIMWS-026, ING-NIMWS-027, ING-SHIM-022, ING-EP-095
    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Serve a connection with the exact query; refuse any other.

        The query check is the first step, before the ingress cap: any
        other query is closed with 1008 before accept (the ASGI server's
        HTTP 403), a refusal send that creates no owner and is one terminal
        attempt once its bound has passed; nothing is read and no
        ingress-cap count is taken.
        """
        if not _serves_query(scope):
            close: Message = {"type": "websocket.close", "code": 1008}
            with contextlib.suppress(Exception):
                await surfaces.bounded_send(
                    self.plugin, send(close), terminal=True
                )
            return
        adapter = RealtimeAdapter(
            plugin=self.plugin, scope=scope, receive=receive, send=send
        )
        await adapter.run()


def bootstrap_endpoint(plugin: Any) -> Callable[[], Awaitable[Any]]:
    """The handler of the Realtime bootstrap ``POST`` route."""
    from fastapi.responses import JSONResponse

    # @spec ING-NIMWS-025, ING-LIFE-045
    async def realtime_transcription_sessions() -> JSONResponse:
        """HTTP 200 and the default session object, in any plugin state.

        It creates no owner and starts no timer; a plugin that is not
        serving refuses on the WebSocket route instead (ING-NIMWS-028).
        """
        factory = plugin.factory
        default_locale = None if factory is None else factory.default_locale
        return JSONResponse(
            bootstrap_session(plugin.served_model or "", default_locale)
        )

    return realtime_transcription_sessions


# ================================================================ bootstrap


# @spec ING-NIMWS-025, ING-NIMWS-013, ING-NIMWS-020
def bootstrap_session(
    model_name: str, default_locale: str | None
) -> dict[str, Any]:
    """The default transcription-session object of the public reference.

    The stock client posts ``{}`` and echoes this back as its first update.
    """
    return {
        "id": _hex_id("sess"),
        "object": "realtime.transcription_session",
        "modalities": ["text"],
        "input_audio_format": "pcm16",
        "input_audio_params": {"sample_rate_hz": 16_000, "num_channels": 1},
        "input_audio_transcription": {
            "model": model_name,
            "language": default_locale or "",
        },
        "recognition_config": {"max_alternatives": 1},
        "speaker_diarization": {"enable_speaker_diarization": False},
        "word_boosting": {
            "enable_word_boosting": False,
            "word_boosting_list": [],
        },
        "endpointing_config": {},
        "client_secret": None,
    }


# =============================================================== projection


# @spec ING-CORE-019
def suffix(previous: str, current: str) -> str:
    """The incremental text between two cumulative hypotheses.

    A hypothesis that extends the previous one yields only the new text; a
    revised one is sent whole, since no suffix of it is true.
    """
    if current.startswith(previous):
        return current[len(previous) :]
    return current


def project_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """One server event on the wire: its ``type``, an ``event_id``, fields."""
    projected: dict[str, Any] = {
        "type": event["type"],
        "event_id": event.get("event_id") or _hex_id("event"),
    }
    projected.update(
        (key, value)
        for key, value in event.items()
        if key not in ("type", "event_id")
    )
    return projected


def _error_object(error: errors.FrontendError) -> dict[str, Any]:
    """The OpenAI-shaped ``error`` object with the catalog code."""
    body: dict[str, Any] = {"code": error.code, "message": error.message()}
    if error.fields:
        body["param"] = ",".join(error.fields)
    if error.code == "busy" and error.authority:
        body["authority"] = error.authority
    return body


# @spec ING-ERR-016, ING-ERR-017, ING-ERR-018, ING-ERR-019, ING-ERR-020
# @spec ING-ERR-022, ING-ERR-023, ING-ERR-024, ING-ERR-026, ING-ERR-048
# @spec ING-ERR-062, ING-ERR-063
def _ending(
    catalog: Any, error: errors.FrontendError, owner: Any
) -> tuple[str, int]:
    """The event and close of a terminal error (errors leaf, phase table).

    ``service_unavailable`` closes 1012 for the shutdown abort row (the
    owner was in provisional shutdown mode, or the drain deadline came
    first), 1013 for an engine refusal before configuration, and 1011 after
    a lease-call timeout or ``engine_dead``. A refusal of ``open`` (the
    factory phase) is an ``error`` event, never ``failed``: ``cadence`` and
    ``request_id_in_use`` close 1011 with ``internal``.
    """
    code = error.code
    cause = owner.cause if owner is not None else None
    phase = None if cause is None else cause.phase
    shutdown = phase in _SHUTDOWN_PHASES or (
        owner is not None
        and owner.shutdown_observed
        and owner.finalization != "client"
    )
    if code == "service_unavailable":
        if shutdown:
            return _ERROR, 1012
        if phase == "factory":
            return _ERROR, 1013
        return (_FAILED if phase == "engine" else _ERROR), 1011
    failing = code in ("internal", "invalid_audio") and phase != "factory"
    event = _FAILED if failing else _ERROR
    if code in ("invalid_audio", "unsupported_format"):
        return event, 1008
    if code == "internal":
        return event, 1011
    close = catalog.realtime_close(code)
    if close is None:
        close = 1008 if code in _CLIENT_CODES else 1011
    return event, int(close)


# =============================================================== validation


def _rejection(code: str, field: str, detail: str) -> errors.FrontendError:
    return errors.FrontendError(code, detail, fields=(field,))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


#: The JSON type of every known field inside a session section.
_FIELD_TYPES: Mapping[str, Callable[[Any], bool]] = MappingProxyType(
    {
        "sample_rate_hz": lambda v: _is_int(v),
        "num_channels": lambda v: _is_int(v),
        "language": lambda v: isinstance(v, str),
        "model": lambda v: isinstance(v, str),
        "prompt": lambda v: isinstance(v, str),
        "max_alternatives": lambda v: _is_int(v),
        "enable_automatic_punctuation": lambda v: isinstance(v, bool),
        "enable_verbatim_transcripts": lambda v: isinstance(v, bool),
        "enable_word_time_offsets": lambda v: isinstance(v, bool),
        "enable_profanity_filter": lambda v: isinstance(v, bool),
        "enable_speaker_diarization": lambda v: isinstance(v, bool),
        "max_speaker_count": lambda v: _is_int(v),
        "enable_word_boosting": lambda v: isinstance(v, bool),
        "word_boosting_list": lambda v: isinstance(v, list),
        **{
            name: _is_number
            for name in (
                "start_history",
                "start_threshold",
                "stop_history",
                "stop_threshold",
                "stop_history_eou",
                "stop_threshold_eou",
            )
        },
    }
)


def _structure_errors(session: Mapping[str, Any]) -> list[str]:
    """Known session fields whose JSON type is wrong (``invalid_event``)."""
    wrong: list[str] = []
    modalities = session.get("modalities", [])
    if not (
        isinstance(modalities, list)
        and all(isinstance(m, str) for m in modalities)
    ):
        wrong.append("modalities")
    if not isinstance(session.get("input_audio_format", ""), str):
        wrong.append("input_audio_format")
    for name in ("custom_configuration", *_SECTION_KEYS):
        value = session.get(name, {})
        if not isinstance(value, dict):
            wrong.append(name)
            continue
        if name == "custom_configuration":
            continue
        wrong.extend(
            key
            for key, item in value.items()
            if key in _SECTION_KEYS[name] and not _FIELD_TYPES[key](item)
        )
    return list(dict.fromkeys(wrong))


def _section(session: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = session.get(name)
    return value if isinstance(value, dict) else {}


def _format_rejections(
    session: Mapping[str, Any],
) -> list[errors.FrontendError]:
    """The joint format check of ``input_audio_format`` and its params."""
    name = session.get("input_audio_format", "pcm16")
    params = _section(session, "input_audio_params")
    rate = params.get("sample_rate_hz", 16_000)
    channels = params.get("num_channels", 1)
    if not (isinstance(name, str) and _is_int(rate) and _is_int(channels)):
        return []  # a structural error, reported as invalid_event
    if name == _DEFERRED:
        found: list[errors.FrontendError] = []
        if rate not in (0, 8_000, 16_000):
            found.append(
                _rejection(
                    "unsupported_format",
                    "sample_rate_hz",
                    f"no accepted cell has a sample rate of {rate} Hz",
                )
            )
        if channels not in (0, 1):
            found.append(
                _rejection(
                    "unsupported_format",
                    "num_channels",
                    "only mono audio is accepted",
                )
            )
        return found
    problem = frontend.validate_format(
        _FORMATS.get(name, name), rate, channels or 1
    )
    if problem is None:
        return []
    return [_format_error(problem)]


def _format_error(problem: frontend.FormatError) -> errors.FrontendError:
    return errors.FrontendError(
        "unsupported_format",
        problem.detail,
        fields=tuple(
            dict.fromkeys(_REALTIME_FORMAT_FIELDS[f] for f in problem.fields)
        ),
    )


# @spec ING-NIMWS-007, ING-NIMWS-009, ING-NIMWS-013, ING-NIMWS-014
# @spec ING-NIMWS-022, ING-EOU-003, ING-EP-026
def validate_session_update(
    update: Mapping[str, Any],
    *,
    served_model: str,
    locales: frozenset[str],
    default_locale: str | None,
    first: bool,
) -> list[errors.FrontendError]:
    """Every rejection of one session object, each naming the dialect field.

    An unknown key is ``invalid_config_field`` on any update, and a named
    locale is checked on any update. The first update also dispositions the
    format, ``model``, ``prompt``, ``modalities`` and every section field for
    field like the gRPC table; later updates leave changed fields to the
    adapter (``config_change_rejected``). ``id``, ``object``,
    ``client_secret`` and ``provenance`` are echo keys.
    """
    found: list[errors.FrontendError] = []
    for key in update:
        if key not in _CONFIG_KEYS and key not in _ECHO_KEYS:
            found.append(
                _rejection(
                    "invalid_config_field", key, f"{key} is not a session field"
                )
            )
    for section, known in _SECTION_KEYS.items():
        for key in _section(update, section):
            if key not in known:
                found.append(
                    _rejection(
                        "invalid_config_field",
                        key,
                        f"{key} is not a field of {section}",
                    )
                )
    transcription = _section(update, "input_audio_transcription")
    if first or "language" in transcription:
        language = transcription.get("language", "")
        if isinstance(language, str):
            locale = surfaces.locale_error(
                language,
                field="language",
                locales=locales,
                default_locale=default_locale,
            )
            if locale is not None:
                found.append(locale)
    if not first:
        return found
    found.extend(_format_rejections(update))
    modalities = update.get("modalities", ["text"])
    if modalities != ["text"]:
        found.append(
            _rejection(
                "invalid_config_field", "modalities", 'only ["text"] is served'
            )
        )
    model = transcription.get("model", "")
    if model and model != served_model:
        found.append(
            _rejection(
                "invalid_config_field",
                "model",
                f"this server serves {served_model!r}",
            )
        )
    if transcription.get("prompt", ""):
        found.append(
            _rejection(
                "invalid_config_field", "prompt", "a prompt is not supported"
            )
        )
    recognition = _section(update, "recognition_config")
    alternatives = recognition.get("max_alternatives", 1)
    if _is_int(alternatives) and alternatives > 1:
        found.append(
            _rejection(
                "invalid_config_field",
                "max_alternatives",
                "greedy decoding has one hypothesis",
            )
        )
    for key in ("enable_profanity_filter", "enable_word_time_offsets"):
        if recognition.get(key) is True:
            found.append(
                _rejection(
                    "invalid_config_field", key, f"{key} is not supported"
                )
            )
    for section in _CAPABILITY_SECTIONS:
        if _section_set(section, _section(update, section)):
            found.append(
                _rejection(
                    "unsupported_capability",
                    section,
                    f"{section.replace('_', ' ')} is not available",
                )
            )
    custom = update.get("custom_configuration")
    if custom:
        found.append(
            _rejection(
                "invalid_config_field",
                "custom_configuration",
                "no custom configuration key is accepted",
            )
        )
    return found


def _section_set(name: str, section: Mapping[str, Any]) -> bool:
    """Whether a capability section is non-default."""
    if name == "speaker_diarization":
        return bool(section.get("enable_speaker_diarization")) or bool(
            section.get("max_speaker_count")
        )
    if name == "word_boosting":
        return bool(section.get("enable_word_boosting")) or bool(
            section.get("word_boosting_list")
        )
    return any(
        bool(section.get(key)) for key in _SECTION_KEYS["endpointing_config"]
    )


def _settings(session: Mapping[str, Any], served_model: str) -> dict[str, Any]:
    """The comparable value of every non-locale field present in ``session``.

    A later update is compared field by field with the configured session;
    a field absent from the update is unchanged.
    """
    values: dict[str, Any] = {}
    if "modalities" in session:
        values["modalities"] = tuple(session["modalities"] or ())
    if "input_audio_format" in session:
        values["input_audio_format"] = session["input_audio_format"]
    params = _section(session, "input_audio_params")
    for key in ("sample_rate_hz", "num_channels"):
        if key in params:
            values[key] = params[key]
    transcription = _section(session, "input_audio_transcription")
    if "model" in transcription:
        values["model"] = transcription["model"] or served_model
    if "prompt" in transcription:
        values["prompt"] = transcription["prompt"]
    recognition = _section(session, "recognition_config")
    for key in _SECTION_KEYS["recognition_config"]:
        if key in recognition:
            value = recognition[key]
            values[key] = (value or 1) if key == "max_alternatives" else value
    for section in _CAPABILITY_SECTIONS:
        if section in session:
            values[section] = _section_set(section, _section(session, section))
    if "custom_configuration" in session:
        values["custom_configuration"] = bool(session["custom_configuration"])
    return values


def _default_settings(served_model: str) -> dict[str, Any]:
    return _settings(
        {
            "modalities": ["text"],
            "input_audio_format": "pcm16",
            "input_audio_params": {"sample_rate_hz": 16_000, "num_channels": 1},
            "input_audio_transcription": {"model": "", "prompt": ""},
            "recognition_config": {
                "max_alternatives": 1,
                "enable_automatic_punctuation": False,
                "enable_verbatim_transcripts": False,
                "enable_word_time_offsets": False,
                "enable_profanity_filter": False,
            },
            "speaker_diarization": {},
            "word_boosting": {},
            "endpointing_config": {},
            "custom_configuration": {},
        },
        served_model,
    )


# ================================================================== adapter


class _Gone(Exception):
    """The transport is gone: a disconnect or a send beyond its bound."""


_TRIGGER = object()  # what a read returns once the shutdown trigger is seen


class RealtimeAdapter:
    """One claimed Riva Realtime connection over its ``StreamOwner``.

    Accepted commands are processed through one FIFO: the adapter reads one
    client event at a time, only between owner operations, so every append
    ahead of a ``clear`` completes its handoff before the ``clear``, and
    reads pause while ``feed`` awaits the engine. After ``done`` a receive
    observer keeps reading until the final's send begins.
    """

    def __init__(
        self, *, plugin: Any, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Bind one claimed connection's ASGI channel to the plugin."""
        self._plugin = plugin
        self._scope = scope
        self._receive = receive
        self._send = send
        self._owner: Any = None
        self._record: Any = None
        self._gone = False
        self._closed = False
        self._item_id = _hex_id("item")
        self._session_id = _hex_id("sess")
        self._hypothesis = ""
        self._format = "pcm16"
        self._rate = 16_000
        self._channels = 1
        self._locale = ""
        self._baseline: dict[str, Any] = {}
        self._riff: bytearray | None = None
        self._remaining: int | None = None
        self._observed: errors.FrontendError | None = None

    # ------------------------------------------------------------- entry

    # @spec ING-NIMWS-028, ING-NIMWS-029, ING-ERR-015, ING-ERR-051
    async def run(self) -> None:
        """Admit or refuse the connection, then serve it to its one end."""
        plugin = self._plugin
        connect = await self._receive()
        if connect.get("type") != "websocket.connect":
            return
        ingress = getattr(plugin, "ingress", None)
        admitted = (
            None
            if ingress is None
            else ingress.try_admit(admission.OwnerKind.realtime)
        )
        if admitted is None:
            refusal = (
                errors.FrontendError(
                    "service_unavailable", "the server is not serving"
                )
                if ingress is None
                else ingress.refusal(admission.OwnerKind.realtime)
            )
            await self._refuse(refusal)
            return
        registry = plugin.registry
        record = registry.register(
            admission.OwnerKind.realtime, asyncio.current_task()
        )
        surfaces.hold(record)
        try:
            self._record = record
            self._owner = lease.StreamOwner(
                lease_opener=plugin.factory.open,
                surface="realtime",
                config=plugin.config,
                clock=plugin.clock,
                registry=registry,
                record=record,
                catalog=plugin.catalog,
            )
            await self._serve()
        finally:
            surfaces.release_owner(registry, record, admitted)

    # @spec ING-NIMWS-028, ING-EP-095, ING-EP-110, ING-LIFE-061
    async def _refuse(self, error: errors.FrontendError) -> None:
        """Accept, send the one ``error``, close 1013: no owner, no read.

        The three messages are one refusal send that creates no owner: one
        bound (``response_send_timeout``, clipped to ``D``), and once that
        bound has passed one terminal attempt for the whole sequence. A
        failed or abandoned refusal starts no further frame.
        """
        event = json.dumps(
            project_event({"type": _ERROR, "error": _error_object(error)})
        )
        messages: tuple[Message, ...] = (
            {"type": "websocket.accept"},
            {"type": "websocket.send", "text": event},
            {"type": "websocket.close", "code": 1013},
        )

        async def sequence() -> None:
            for message in messages:
                await self._send(message)

        try:
            await surfaces.bounded_send(self._plugin, sequence(), terminal=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._gone = True
            return
        self._closed = True

    # @spec ING-NIMWS-017, ING-NIMWS-023, ING-ERR-005, ING-ERR-011
    # @spec ING-ERR-027, ING-ERR-033, ING-ERR-034, ING-ERR-062
    async def _serve(self) -> None:
        try:
            await self._send_bounded({"type": "websocket.accept"})
            await self._emit(
                _CREATED,
                conversation={
                    "id": _hex_id("conv"),
                    "object": "realtime.conversation",
                },
            )
            await self._configure()
            client = await self._stream()
            text = await self._finalize(client=client)
            shutdown = self._owner.finalization == "shutdown"
            # The final's send begins here: it is irrevocable from now on.
            await self._emit(
                _COMPLETED,
                item_id=self._item_id,
                content_index=0,
                transcript=text,
                is_last_result=True,
            )
            await self._close(1001 if shutdown else 1000)
        except errors.FrontendError as error:
            await self._terminate(error)
        except _Gone:
            await self._abandon()
        except asyncio.CancelledError:
            await self._cancelled()
        except Exception as unexpected:  # a bug here, not the client's
            await self._terminate(
                errors.FrontendError(
                    "internal", f"unexpected failure: {unexpected!r}"
                ),
                "engine",
            )

    # @spec ING-EP-110, ING-LIFE-061
    async def _terminate(
        self, error: errors.FrontendError, phase: str = "protocol"
    ) -> None:
        """Settle the owner's one cause, then project it and close.

        The terminal event and the close that follows it are sent in
        sequence as one send under one bound and, once that bound has
        passed, as one terminal attempt (ING-EP-110), as ``_refuse`` sends
        its refusal.
        """
        owner = self._owner
        await surfaces.settle(owner, error, phase)
        error = surfaces.selected_error(owner, error)
        if self._gone or self._closed:
            return
        event, close = _ending(self._plugin.catalog, error, owner)
        text = json.dumps(project_event(self._error_event(event, error)))
        messages: tuple[Message, ...] = (
            {"type": "websocket.send", "text": text},
            {"type": "websocket.close", "code": close},
        )

        async def sequence() -> None:
            for message in messages:
                await self._send(message)

        try:
            await self._bounded(sequence(), terminal=True)
        except _Gone:
            return
        self._closed = True

    async def _abandon(self) -> None:
        """The transport is gone: clean up and send nothing more."""
        self._gone = True
        await surfaces.settle_shielded(
            self._record,
            self._owner,
            errors.FrontendError("internal", "the transport is gone"),
            "transport",
        )

    async def _cancelled(self) -> None:
        """The plugin cancelled this owner: the shutdown abort row.

        Cleanup is joined through any further cancellation, so the record
        and admission outlive it (unless ``stop`` tore the plugin down).
        """
        shutdown = errors.FrontendError(
            "service_unavailable", "the server is shutting down"
        )
        await surfaces.settle_shielded(
            self._record, self._owner, shutdown, "shutdown"
        )
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            task.uncancel()
        await self._terminate(shutdown, "shutdown")

    # ------------------------------------------------------------- sends

    async def _send_bounded(self, message: Message) -> None:
        """One send or close under the send bound (``bounded_send``).

        A send that exceeds it, or fails, ends the connection by the
        transport path: nothing further is sent, the close included. Once
        the bound has passed such a send never starts; only the terminal
        error event and its close (``_terminate``) are then attempted.
        """
        if self._gone or self._closed:
            raise _Gone
        await self._bounded(self._send(message), terminal=False)

    async def _bounded(self, send: Awaitable[None], *, terminal: bool) -> None:
        """``send``, one message or the terminal sequence, under one bound."""
        try:
            await surfaces.bounded_send(
                self._plugin, send, record=self._record, terminal=terminal
            )
        except asyncio.CancelledError:
            raise
        except Exception as failure:
            self._gone = True
            raise _Gone from failure

    async def _emit(self, kind: str, **fields: Any) -> None:
        await self._send_event({"type": kind, **fields})

    async def _send_event(self, event: dict[str, Any]) -> None:
        text = json.dumps(project_event(event))
        await self._send_bounded({"type": "websocket.send", "text": text})

    async def _emit_error(self, kind: str, error: errors.FrontendError) -> None:
        """A non-terminal error event; the session continues."""
        await self._send_event(self._error_event(kind, error))

    def _error_event(
        self, kind: str, error: errors.FrontendError
    ) -> dict[str, Any]:
        event: dict[str, Any] = {"type": kind, "error": _error_object(error)}
        if kind == _FAILED:
            event.update(item_id=self._item_id, content_index=0)
        return event

    async def _close(self, code: int) -> None:
        await self._send_bounded({"type": "websocket.close", "code": code})
        self._closed = True

    # ------------------------------------------------------------- reads

    # @spec ING-NIMWS-012, ING-NIMWS-015, ING-ERR-026
    def _classify(
        self, message: Message
    ) -> dict[str, Any] | errors.FrontendError:
        """One received message as an event, or the error it is.

        Sizes are checked before any JSON or base64 decode: above
        ``ws_receive_max_bytes`` or ``ws_event_envelope_max_bytes`` it is
        ``request_too_large``. Malformed JSON, a non-object, a missing or
        unknown type and wrong field types are ``invalid_event``.
        """
        kind = message.get("type")
        if kind != "websocket.receive":
            raise _Gone
        config = self._plugin.config
        text = message.get("text")
        raw = message.get("bytes")
        size = len(raw) if raw is not None else len((text or "").encode())
        for limit_name in (
            "ws_receive_max_bytes",
            "ws_event_envelope_max_bytes",
        ):
            limit = int(getattr(config, limit_name))
            if size > limit:
                return errors.FrontendError(
                    "request_too_large",
                    f"a {size}-byte message exceeds {limit_name} ({limit})",
                )
        try:
            body = raw.decode("utf-8") if raw is not None else (text or "")
            event = json.loads(body)
        except (UnicodeDecodeError, ValueError):
            return errors.FrontendError(
                "invalid_event", "the message is not a JSON object"
            )
        if not isinstance(event, dict):
            return errors.FrontendError(
                "invalid_event", "the message is not a JSON object"
            )
        event_type = event.get("type")
        if not isinstance(event_type, str) or event_type not in EVENT_TYPES:
            return _rejection(
                "invalid_event", "type", f"unknown event type {event_type!r}"
            )
        if event_type == APPEND and not isinstance(event.get("audio"), str):
            return _rejection(
                "invalid_event", "audio", "audio must be a base64 string"
            )
        if event_type == UPDATE:
            session = event.get("session")
            if not isinstance(session, dict):
                return _rejection(
                    "invalid_event", "session", "session must be an object"
                )
            wrong = _structure_errors(session)
            if wrong:
                return errors.FrontendError(
                    "invalid_event",
                    "session fields have the wrong JSON type",
                    fields=tuple(wrong),
                )
        return event

    # @spec ING-LIFE-039
    async def _next_event(self) -> Any:
        """The next client event under the owner's client-wait bound.

        Returns ``_TRIGGER`` once the shutdown trigger reached the owner;
        raises the terminal error of an expired bound or invalid message.
        """
        try:
            message = await self._call(self._owner.receive(self._receive()))
        except lease.ShutdownObserved:
            return _TRIGGER
        if message.get("type") != "websocket.receive":
            raise _Gone
        event = self._classify(message)
        if isinstance(event, errors.FrontendError):
            phase = "input" if event.code == "request_too_large" else "protocol"
            raise await self._terminal(event, phase)
        return event

    async def _terminal(
        self, error: errors.FrontendError, phase: str
    ) -> errors.FrontendError:
        await surfaces.settle(self._owner, error, phase)
        return surfaces.selected_error(self._owner, error)

    async def _call(self, operation: Awaitable[T]) -> T:
        """One owner operation; a failure has already been cleaned up."""
        try:
            return await operation
        except errors.FrontendError:
            if self._gone:
                raise _Gone from None
            raise

    # ------------------------------------------------------ configuration

    # @spec ING-NIMWS-029, ING-NIMWS-022, ING-NIMWS-023, ING-ERR-034
    # @spec ING-CORE-021, ING-LIFE-001, ING-ERR-016, ING-VEH-007
    # @spec ING-ERR-063
    async def _configure(self) -> None:
        """Read updates until one validates and ``open`` succeeds.

        A rejected first update emits every field error and leaves the
        session unconfigured for a corrected one, all within the original
        preconfiguration bound; any other event first is ``protocol_order``.
        """
        plugin = self._plugin
        factory = plugin.factory
        owner = self._owner
        while True:
            event = await self._next_event()
            if event is _TRIGGER:
                raise await self._terminal(
                    errors.FrontendError(
                        "service_unavailable", "the server is shutting down"
                    ),
                    "shutdown",
                )
            if event["type"] != UPDATE:
                raise await self._terminal(
                    errors.FrontendError(
                        "protocol_order",
                        "the first event must be transcription_session.update",
                    ),
                    "protocol",
                )
            session = event["session"]
            rejections = validate_session_update(
                session,
                served_model=plugin.served_model,
                locales=frozenset(factory.locales),
                default_locale=factory.default_locale,
                first=True,
            )
            if rejections:
                for rejection in rejections:
                    await self._emit_error(_ERROR, rejection)
                continue
            audio = self._adopt(session)
            try:
                await self._call(
                    owner.open(
                        cadence=lease.STREAM_CADENCE, locale=self._locale
                    )
                )
            except errors.FrontendError as refused:
                # A locale refusal leaves the owner unconfigured with no
                # cause (ING-ERR-063): an ``error`` with no close, and a
                # corrected update within the original bound.
                if owner.cause is not None:
                    raise
                await self._emit_error(_ERROR, refused)
                continue
            if audio is not None:
                owner.attach_frontend(audio)
            await self._emit(_UPDATED, session=self._session_object())
            return

    def _adopt(
        self, session: Mapping[str, Any]
    ) -> frontend.StreamingAudioFrontend | None:
        """Record a valid first update; the audio frontend of a known cell."""
        factory = self._plugin.factory
        params = _section(session, "input_audio_params")
        transcription = _section(session, "input_audio_transcription")
        self._format = str(session.get("input_audio_format", "pcm16"))
        self._rate = int(params.get("sample_rate_hz", 16_000))
        self._channels = int(params.get("num_channels", 1))
        self._locale = str(
            transcription.get("language") or factory.default_locale
        )
        served = self._plugin.served_model
        self._baseline = {
            **_default_settings(served),
            **_settings(session, served),
        }
        if self._format == _DEFERRED:
            self._riff = bytearray()
            return None
        return frontend.StreamingAudioFrontend(
            encoding=_FORMATS[self._format], sample_rate_hz=self._rate
        )

    def _session_object(self) -> dict[str, Any]:
        """The honoured session ``transcription_session.updated`` carries."""
        session = bootstrap_session(self._plugin.served_model, self._locale)
        session["id"] = self._session_id
        session["input_audio_format"] = self._format
        session["input_audio_params"] = {
            "sample_rate_hz": self._rate,
            "num_channels": self._channels,
        }
        return session

    # ----------------------------------------------------------- stream

    # @spec ING-NIMWS-005, ING-NIMWS-018, ING-NIMWS-024, ING-LIFE-009
    async def _stream(self) -> bool:
        """Serve client events until ``done`` (``True``) or the trigger."""
        owner = self._owner
        while True:
            event = await self._next_event()
            if event is _TRIGGER:
                break
            kind = event["type"]
            if kind == DONE:
                return True
            if kind == APPEND:
                await self._append(event["audio"])
            elif kind == COMMIT:
                await self._emit(_COMMITTED, item_id=self._item_id)
            elif kind == CLEAR:
                if owner.frontend is not None:
                    owner.frontend.reset()
                if self._riff is not None:
                    self._riff = bytearray()
                await self._emit(_CLEARED, item_id=self._item_id)
            else:
                await self._update(event["session"])
        return False

    # @spec ING-NIMWS-018, ING-LIFE-065
    async def _append(self, payload: str) -> None:
        """Decode one append in bounded slices and feed each in turn.

        Every slice of an already-read append is fed, also once the
        shutdown trigger is observed, before shutdown finalization.
        """
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            raise await self._terminal(
                _rejection(
                    "invalid_audio", "audio", "the audio is not valid base64"
                ),
                "input",
            ) from None
        if self._riff is not None:
            resolved = await self._resolve_header(data)
            if resolved is None:
                return
            data = resolved
        if self._remaining is not None:
            data = data[: self._remaining]
            self._remaining -= len(data)
        owner = self._owner
        audio = owner.frontend
        if audio is None or not data:
            return
        bound = int(self._plugin.config.pre_submit_max_samples)
        for piece in audio.decode(data, max_samples=bound):
            if not len(piece):
                continue
            hypotheses = await self._call(owner.feed(piece))
            for text in hypotheses:
                if text == self._hypothesis:
                    continue
                delta = suffix(self._hypothesis, text)
                self._hypothesis = text
                if delta:
                    await self._emit(
                        _DELTA,
                        item_id=self._item_id,
                        content_index=0,
                        delta=delta,
                    )

    async def _resolve_header(self, data: bytes) -> bytes | None:
        """Buffer deferred-header bytes; return the audio once resolved."""
        assert self._riff is not None
        bound = int(self._plugin.config.max_riff_header_bytes)
        buffered = bytes(self._riff) + data
        header = frontend.sniff_riff(buffered, max_header_bytes=bound)
        if header is None:
            self._riff = bytearray(buffered)
            return None
        problem = surfaces.header_problem(header, self._rate, self._channels)
        if problem is not None:
            raise await self._terminal(_format_error(problem), "protocol")
        assert isinstance(header, frontend.RiffFormat)
        self._riff = None
        self._remaining = header.data_length
        self._owner.attach_frontend(
            frontend.StreamingAudioFrontend(
                encoding=header.encoding, sample_rate_hz=header.sample_rate_hz
            )
        )
        return buffered[header.data_offset :]

    # @spec ING-LIFE-008, ING-VEH-007
    async def _update(self, session: Mapping[str, Any]) -> None:
        """A later update: rejections first, then one truthful ack.

        Only the locale is mutable, through the serialized ``update_locale``;
        every other changed field is ``config_change_rejected``.
        """
        plugin = self._plugin
        factory = plugin.factory
        rejections = validate_session_update(
            session,
            served_model=plugin.served_model,
            locales=frozenset(factory.locales),
            default_locale=factory.default_locale,
            first=False,
        )
        changed = [
            name
            for name, value in _settings(session, plugin.served_model).items()
            if self._baseline.get(name) != value
        ]
        rejections.extend(
            _rejection(
                "config_change_rejected",
                name,
                f"{name} cannot change after configuration",
            )
            for name in changed
        )
        for rejection in rejections:
            await self._emit_error(_ERROR, rejection)
        transcription = _section(session, "input_audio_transcription")
        locale_rejected = any("language" in r.fields for r in rejections)
        if "language" in transcription and not locale_rejected:
            locale = str(transcription["language"] or factory.default_locale)
            if locale != self._locale:
                await self._change_locale(locale)
        await self._emit(_UPDATED, session=self._session_object())

    async def _change_locale(self, locale: str) -> None:
        """Run ``update_locale``; a refusal keeps the prior locale."""
        owner = self._owner
        try:
            await self._call(owner.update_locale(locale))
        except errors.FrontendError as refused:
            if owner.cause is not None or owner.state == "terminal":
                raise
            await self._emit_error(_ERROR, refused)
            return
        self._locale = locale

    # ------------------------------------------------------- finalization

    # @spec ING-ERR-057, ING-ERR-058, ING-NIMWS-023, ING-LIFE-024
    async def _finalize(self, *, client: bool) -> str:
        """The one finalization; after ``done`` an observer keeps reading.

        The observer holds at most one message, under the finalization
        bound, until the final's send begins: the owner stops its read when
        the stream ends, and the shutdown trigger stops it earlier.
        """
        owner = self._owner
        if self._riff:
            raise await self._terminal(
                errors.FrontendError(
                    "unsupported_format",
                    "the stream ended inside its RIFF/WAVE header",
                    fields=("input_audio_format",),
                ),
                "protocol",
            )
        observer = (
            surfaces.child(self._record, self._observe()) if client else None
        )
        failure: errors.FrontendError | None = None
        text = ""
        try:
            text = await owner.finalize()
        except errors.FrontendError as raised:
            failure = raised
        if observer is not None:
            await asyncio.wait({observer})
        observed = self._observed
        if observed is not None:
            if self._gone:
                raise _Gone
            raise surfaces.selected_error(owner, observed)
        if failure is not None:
            if self._gone:
                raise _Gone
            raise failure
        return text

    async def _observe(self) -> None:
        """Read one message after ``done`` and classify it."""
        owner = self._owner
        try:
            message = await owner.receive(self._receive())
        except (lease.ShutdownObserved, errors.FrontendError, RuntimeError):
            return  # the trigger, an expiry or the stream's end stops it
        if message.get("type") != "websocket.receive":
            self._gone = True
            observed = errors.FrontendError("internal", "the client is gone")
            phase = "transport"
        else:
            event = self._classify(message)
            if isinstance(event, errors.FrontendError):
                observed = event
                phase = (
                    "input" if event.code == "request_too_large" else "protocol"
                )
            else:
                observed = errors.FrontendError(
                    "session_terminal",
                    f"{event['type']} arrived after input_audio_buffer.done",
                )
                phase = "protocol"
        self._observed = observed
        await owner.fail(lease.Cause(observed.code, phase, None))


__all__ = [
    "EVENT_TYPES",
    "RealtimeAdapter",
    "RealtimeRoute",
    "bootstrap_endpoint",
    "bootstrap_session",
    "project_event",
    "suffix",
    "validate_session_update",
]

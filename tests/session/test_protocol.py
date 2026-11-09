"""Structural coupling: the frontend uses the seam only through its protocol.

Static rules (no model-plugin import, no model-plugin exception class, no
remote provider) are checked over the package's AST; the runtime rule (only
protocol members are touched) is checked with an attribute guard around the
fake factory and lease while real streams run on both surfaces.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
from support.audio import pcm16_bytes, pcm16_sine
from support.fake_session import FakeSessionFactory, fake_entry_points
from support.realtime import CREATED

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "vllm_riva_frontend"

FACTORY_MEMBERS = frozenset(
    {
        "served_cadences",
        "locales",
        "default_locale",
        "open",
        "drain",
        "teardown_all",
    }
)
LEASE_MEMBERS = frozenset(
    {"feed", "update_locale", "flush", "finish", "abort", "release", "teardown"}
)
MODEL_PLUGIN_EXCEPTIONS = frozenset(
    {"SessionRefused", "SessionFailed", "SessionError"}
)


def _modules() -> list[tuple[Path, ast.Module]]:
    paths = sorted(PACKAGE.glob("*.py"))
    assert paths, PACKAGE
    return [(p, ast.parse(p.read_text(), filename=str(p))) for p in paths]


def _imported(tree: ast.Module) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _members(protocol: type) -> set[str]:
    names = set(getattr(protocol, "__protocol_attrs__", ()) or ())
    if not names:
        names = {
            name for name in vars(protocol) if not name.startswith("_")
        } | set(getattr(protocol, "__annotations__", {}))
    return {name for name in names if not name.startswith("_")}


# ------------------------------------------------------------------ static


# @spec ING-CORE-013
def test_the_package_imports_no_model_plugin_and_catches_none_of_its_classes():
    offenders = []
    for path, tree in _modules():
        for name in _imported(tree):
            if "nemotron" in name.lower():
                offenders.append(f"{path.name}: import {name}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is not None:
                for sub in ast.walk(node.type):
                    ident = (
                        sub.id
                        if isinstance(sub, ast.Name)
                        else sub.attr
                        if isinstance(sub, ast.Attribute)
                        else None
                    )
                    if ident in MODEL_PLUGIN_EXCEPTIONS:
                        offenders.append(f"{path.name}: except {ident}")
    assert offenders == []


# @spec ING-CORE-013
def test_the_protocols_declare_exactly_the_seam_members():
    from vllm_riva_frontend.lease import SessionFactory, SessionLease

    assert _members(SessionFactory) == set(FACTORY_MEMBERS)
    assert _members(SessionLease) == set(LEASE_MEMBERS)
    update = inspect.signature(SessionLease.update_locale)
    assert update.return_annotation in (None, "None", type(None))
    feed = inspect.signature(SessionLease.feed)
    assert feed.parameters["on_accepted"].kind is inspect.Parameter.KEYWORD_ONLY
    opener = inspect.signature(SessionFactory.open)
    for name in ("cadence", "locale"):
        assert opener.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


# @spec ING-CORE-013
def test_seam_error_reads_the_two_attributes_with_getattr():
    from vllm_riva_frontend.errors import seam_error

    class Carrier(Exception):
        session_error = "refused"
        reason = "capacity"

    class OnlyReason(Exception):
        reason = "expired"

    assert seam_error(Carrier()) == ("refused", "capacity")
    assert seam_error(OnlyReason()) == (None, "expired")
    assert seam_error(RuntimeError("plain")) == (None, None)


# @spec ING-CORE-018
def test_there_is_no_remote_provider_mode_or_websocket_loopback():
    from vllm_riva_frontend.config import ConfigError, parse_config

    offenders = []
    for path, tree in _modules():
        stem = path.stem.lower()
        if any(word in stem for word in ("provider", "broker", "event_bus")):
            offenders.append(f"module {path.name}")
        for name in _imported(tree):
            top = name.split(".")[0]
            if top in {"websockets", "websocket", "aiohttp", "httpx"}:
                offenders.append(f"{path.name}: import {name}")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith(("ws://", "wss://"))
            ):
                offenders.append(f"{path.name}: {node.value!r}")
            if isinstance(node, ast.ClassDef):
                lowered = node.name.lower()
                if any(
                    w in lowered for w in ("provider", "broker", "eventbus")
                ):
                    offenders.append(f"{path.name}: class {node.name}")
    assert offenders == []

    config = parse_config({"riva_frontend": {}})
    assert [n for n in dir(config) if "provider" in n.lower()] == []
    for key in ("provider", "provider_mode", "session_provider"):
        with pytest.raises(ConfigError):
            parse_config({"riva_frontend": {key: "remote"}})


# @spec ING-CORE-009, ING-CORE-010, ING-CORE-018
def test_the_stream_owner_takes_no_provider_batcher_or_cadence_argument():
    from vllm_riva_frontend.lease import StreamOwner

    parameters = inspect.signature(StreamOwner).parameters
    assert set(parameters) == {
        "lease_opener",
        "surface",
        "config",
        "clock",
        "registry",
        "record",
        "catalog",
    }
    for parameter in parameters.values():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


# ----------------------------------------------------------------- runtime


class _Guard:
    """Forwards attribute reads to ``target`` and logs non-protocol reads."""

    def __init__(self, target: Any, allowed: frozenset[str], log: list[str]):
        self.__dict__["_target"] = target
        self.__dict__["_allowed"] = allowed
        self.__dict__["_log"] = log

    def __getattr__(self, name: str) -> Any:
        if not name.startswith("__") and name not in self._allowed:
            self._log.append(name)
        attribute = getattr(self._target, name)
        if name == "open":

            async def open_(*args: Any, **kwargs: Any) -> Any:
                lease = await attribute(*args, **kwargs)
                return _Guard(lease, LEASE_MEMBERS, self._log)

            return open_
        return attribute


class TestOnlyProtocolMembers:
    @pytest.fixture
    def access_log(self) -> list[str]:
        return []

    @pytest.fixture
    def entry_points(self, fake_factory, access_log):
        def constructor(engine_client, model_config):
            fake_factory.constructor(engine_client, model_config)
            return _Guard(fake_factory, FACTORY_MEMBERS, access_log)

        return fake_entry_points(fake=constructor)

    @pytest.fixture
    def fake_factory(self, clock):
        return FakeSessionFactory(final_text="hello world", clock=clock)

    def _check_calls(self, factory: FakeSessionFactory) -> None:
        opens = [c for c in factory.calls if c.name == "open"]
        assert opens
        for call in opens:
            assert call.args == ()
            assert set(call.kwargs) == {"cadence", "locale"}
        feeds = [c for c in factory.calls if c.name == "feed"]
        assert feeds
        for call in feeds:
            assert len(call.args) == 1
            assert set(call.kwargs) == {"on_accepted"}
            assert callable(call.kwargs["on_accepted"])
        update = [c for c in factory.calls if c.name == "update_locale"]
        for call in update:
            assert call.kwargs == {}

    # @spec ING-CORE-009, ING-CORE-010, ING-CORE-013, ING-CORE-018
    async def test_riva_stream_touches_only_protocol_members(
        self, riva, fake_factory, access_log, test_app
    ):
        call = riva()
        call.configure()
        call.audio(pcm16_bytes(pcm16_sine(0.5)))
        call.half_close()
        await call.wait()

        call.assert_ok()
        assert access_log == []
        self._check_calls(fake_factory)
        assert getattr(test_app.state, "core_realtime_scopes", []) == []

    # @spec ING-CORE-009, ING-CORE-010, ING-CORE-013, ING-CORE-018
    async def test_realtime_session_touches_only_protocol_members(
        self, realtime, first_update, fake_factory, access_log, test_app
    ):
        session = realtime()
        assert await session.connect()
        assert (await session.next())["type"] == CREATED
        await session.update(first_update("en-US"))
        await session.next()
        await session.update(first_update("de-DE"))
        await session.append(pcm16_bytes(pcm16_sine(0.5)))
        await session.done_event()
        await session.wait_closed()

        assert access_log == []
        self._check_calls(fake_factory)
        # The claimed scope never reached core's /v1/realtime route.
        assert getattr(test_app.state, "core_realtime_scopes", []) == []

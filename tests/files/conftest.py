"""Fixtures and helpers for the files leaf.

The frontend's modules are imported inside fixtures and tests, never at
module level, so collection succeeds while the implementation is missing and
every test then fails on the missing module.

``files_env`` builds one server the way core does (``support.app``), with the
frontend plugin attached, its state initialized and (by default) started, so
admission is open. Requests go through ``support.http.HttpExchange``, whose
body chunks, disconnect and individual sends the test controls.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pytest
from support.app import Served, serve_plugin
from support.fake_engine import FakeEngineClient
from support.fake_session import FakeSessionFactory, fake_entry_points
from support.fake_transcription import FakeTranscriptionServing
from support.grpc_harness import free_port
from support.http import HttpExchange, file_request
from support.state import make_args

# --------------------------------------------------------------- the server


@dataclass
class FilesEnv:
    """One served plugin (``support.app.Served``) seen from the files leaf."""

    served: Served

    plugin = property(lambda self: self.served.plugin)
    app = property(lambda self: self.served.app)
    state = property(lambda self: self.served.state)
    engine = property(lambda self: self.served.engine)
    models = property(lambda self: self.served.models)
    service = property(lambda self: self.served.transcription)
    factory = property(lambda self: self.served.factory)
    clock = property(lambda self: self.served.clock)

    def request(self, *args: Any, **kwargs: Any) -> HttpExchange:
        return file_request(self.app, *args, **kwargs)

    @property
    def calls(self) -> list[Any]:
        assert self.service is not None
        return self.service.calls


@pytest.fixture
async def files_env(clock):
    """Build ``FilesEnv`` servers on demand (``support.app.serve_plugin``).

    Keyword arguments: ``riva_frontend`` (merged over a free ``grpc_bind``),
    ``engine`` (``FakeEngineClient`` keywords), ``tasks``, ``factory`` (a
    ``FakeSessionFactory``), ``service``, ``load_tracking``, ``start`` and
    ``prepare`` (called with the engine before ``init_state``).
    """
    from vllm_riva_frontend.lifecycle import RivaFrontendPlugin

    envs: list[FilesEnv] = []

    async def make(
        *,
        riva_frontend: dict[str, Any] | None = None,
        engine: dict[str, Any] | None = None,
        tasks: Iterable[str] = ("transcription", "realtime"),
        factory: FakeSessionFactory | None = None,
        service: FakeTranscriptionServing | None = None,
        load_tracking: bool = False,
        start: bool = True,
        prepare: Any = None,
    ) -> FilesEnv:
        config = {"grpc_bind": f"127.0.0.1:{free_port()}"}
        config.update(riva_frontend or {})
        fake_engine = FakeEngineClient(
            riva_frontend=config, tasks=tuple(tasks), **(engine or {})
        )
        if prepare is not None:
            prepare(fake_engine)
        fake_factory = factory or FakeSessionFactory(clock=clock)
        plugin = RivaFrontendPlugin(
            clock=clock,
            entry_points=fake_entry_points(nemotron=fake_factory.constructor),
        )
        served = await serve_plugin(
            plugin,
            fake_engine,
            tasks=tasks,
            transcription=service,
            args=make_args(enable_server_load_tracking=load_tracking),
            start=start,
        )
        env = FilesEnv(served)
        envs.append(env)
        return env

    yield make

    for env in envs:
        # Teardown must not mask the test's result.
        with contextlib.suppress(Exception):
            await env.served.stop()

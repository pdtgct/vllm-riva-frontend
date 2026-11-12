"""The Starlette ``State`` core's API server builds.

``make_state`` fills a ``State`` the way ``init_app_state``
(``vllm/entrypoints/launchers/api_server/app_state.py``) does for the
attributes the frontend reads or core's plugin functions set:

* ``engine_client``, ``vllm_config``, ``args``, ``log_stats`` (``:73-76``);
* ``openai_serving_models`` (``:81``);
* ``openai_serving_transcription`` when the server has the ``transcription``
  task (``vllm/entrypoints/speech_to_text/factories.py:47-56``);
* ``enable_server_load_tracking`` and ``server_load_metrics`` (``:176-177``);
* ``request_shutdown``, which the launcher sets to its signal handler
  (``vllm/entrypoints/launchers/launcher.py:255-261``); here a
  ``RecordingShutdown`` that counts calls;
* ``draining``, which core's ``init_endpoint_plugins_state`` sets to
  ``False`` (``vllm/plugins/endpoint_plugins/interface.py:193``) and
  ``drain_endpoint_plugins`` to ``True`` (``:260``). ``make_state`` leaves
  it unset so a test sees whether the plugin relied on core setting it.

``make_args`` is the ``Namespace`` of serve arguments the app and the
plugin functions read (``api_server_count``, ``enable_server_load_tracking``,
the CORS and ``api_key`` options of ``init_entrypoints_middleware``).
``make_args(api_server_count=ABSENT)`` leaves the attribute out, as a
namespace built without core's serve parser would (ING-EP-081 counts it as
one process). ``make_state(draining=...)`` presets ``state.draining`` (core
sets it before or without a ``drain`` call, ING-EP-082 and ING-EP-102).

``RecordingShutdown(on_call=...)`` runs ``on_call()`` at each call and keeps
its results in ``observations``, so a test sees what held at the instant the
plugin requested shutdown (for example that admission was already closed,
ING-EP-111).
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Iterable
from typing import Any

from starlette.datastructures import State

from support.fake_engine import DEFAULT_MODEL, FakeEngineClient
from support.fake_transcription import (
    FakeServingModels,
    FakeTranscriptionServing,
)

#: Leave an argument out of the namespace altogether.
ABSENT = object()


class RecordingShutdown:
    """``state.request_shutdown``: synchronous, idempotent, counted."""

    def __init__(self, on_call: Callable[[], Any] | None = None) -> None:
        self.calls = 0
        self.on_call = on_call
        self.observations: list[Any] = []

    def __call__(self) -> None:
        self.calls += 1
        if self.on_call is not None:
            self.observations.append(self.on_call())

    @property
    def requested(self) -> bool:
        return self.calls > 0


def make_args(**overrides: Any) -> Namespace:
    """Serve arguments with core's attribute names and defaults."""
    values: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "served_model_name": None,
        "api_server_count": 1,
        "enable_server_load_tracking": False,
        "enable_force_include_usage": False,
        "disable_log_stats": True,
        "allowed_origins": [],
        "allow_credentials": False,
        "allowed_methods": ["*"],
        "allowed_headers": ["*"],
        "api_key": None,
        "enable_request_id_headers": False,
        "middleware": [],
        "root_path": None,
        "disable_fastapi_docs": True,
        "enable_offline_docs": False,
        "enable_log_requests": False,
        "max_log_len": None,
        "log_error_stack": False,
    }
    values.update(overrides)
    return Namespace(
        **{key: value for key, value in values.items() if value is not ABSENT}
    )


def make_state(
    engine: FakeEngineClient,
    *,
    tasks: Iterable[str] = ("transcription", "realtime"),
    args: Namespace | None = None,
    state: State | None = None,
    models: FakeServingModels | None = None,
    transcription: FakeTranscriptionServing | None = None,
    draining: bool | None = None,
) -> State:
    """Fill ``state`` as core does before ``init_endpoint_plugins_state``.

    A new ``State`` is built when none is given. ``draining`` is left unset
    unless given (core's Phase B sets it to ``False``).
    """
    tasks = tuple(tasks)
    state = state if state is not None else State()
    args = args if args is not None else make_args()
    if models is None:
        name = args.served_model_name or engine.model_config.served_model_name
        names = name if isinstance(name, list) else [name]
        models = FakeServingModels(*names)
    state.engine_client = engine
    state.vllm_config = engine.vllm_config
    state.args = args
    state.log_stats = not args.disable_log_stats
    state.openai_serving_models = models
    if "transcription" in tasks:
        state.openai_serving_transcription = (
            transcription
            if transcription is not None
            else FakeTranscriptionServing(models)
        )
    state.request_shutdown = RecordingShutdown()
    if draining is not None:
        state.draining = draining
    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0
    return state

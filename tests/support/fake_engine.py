"""A fake engine client with core's configuration attribute names.

The frontend reads these from the client core hands ``init_state`` (core
paths are in the RFC 1 + RFC 2 tree):

* ``EngineClient.vllm_config`` and ``EngineClient.model_config``
  (``vllm/engine/protocol.py:46-47``); ``errored``, ``dead_error``
  (``:61``, ``:69``); ``abort(request_id)`` (``:133``); ``check_health()``
  (``:164``);
* ``vllm_config.additional_config`` (the ``riva_frontend`` and
  ``stream_admission`` mappings; ``vllm/v1/core/sched/state_scheduler.py:85-91``
  reads the latter);
* ``vllm_config.cache_config.num_gpu_blocks`` (``vllm/config/cache.py:338``;
  the state-slot count is this minus one);
* ``vllm_config.scheduler_config.max_num_seqs`` and ``max_num_active_seqs``
  (``vllm/config/scheduler.py:62``, ``:69``);
* ``vllm_config.parallel_config.data_parallel_size``
  (``vllm/config/parallel.py:155``);
* ``vllm_config.shutdown_timeout`` (``vllm/config/vllm.py:482``; the launcher
  passes it to ``drain_endpoint_plugins``);
* ``model_config.served_model_name`` and ``model_config.max_model_len``
  (``vllm/config/model.py:336``, ``:256``).

The engine's stream deadlines default to ``StreamAdmissionConfig``'s
(``vllm/v1/core/stream_admission.py:38-40``): 15, 60 and 40 seconds.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

from vllm.v1.engine.exceptions import EngineDeadError

DEFAULT_MODEL = "nvidia/nemotron-asr-streaming-test"
_UNSET = object()


class FakeEngineClient:
    """See the module docstring."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        served_model_name: str | list[str] | None = None,
        tasks: Iterable[str] = ("transcription", "realtime"),
        num_gpu_blocks: int | None = 17,
        max_num_seqs: int = 16,
        max_num_active_seqs: int | None = None,
        data_parallel_size: int = 1,
        shutdown_timeout: float = 60.0,
        max_model_len: int = 76_800,
        riva_frontend: dict[str, Any] | None = None,
        stream_admission: dict[str, Any] | None = None,
        additional_config: dict[str, Any] | None = None,
    ) -> None:
        self.tasks = tuple(tasks)
        config: dict[str, Any] = dict(additional_config or {})
        if riva_frontend is not None:
            config["riva_frontend"] = riva_frontend
        if stream_admission is not None:
            config["stream_admission"] = stream_admission
        self.model_config = SimpleNamespace(
            model=model,
            served_model_name=served_model_name or model,
            max_model_len=max_model_len,
            tokenizer=model,
        )
        self.vllm_config = SimpleNamespace(
            model_config=self.model_config,
            additional_config=config,
            cache_config=SimpleNamespace(num_gpu_blocks=num_gpu_blocks),
            scheduler_config=SimpleNamespace(
                max_num_seqs=max_num_seqs,
                max_num_active_seqs=max_num_active_seqs,
            ),
            parallel_config=SimpleNamespace(
                data_parallel_size=data_parallel_size
            ),
            shutdown_timeout=shutdown_timeout,
        )
        self.renderer = None
        self.input_processor = None
        self.aborted: list[str] = []
        self.health_checks = 0
        self._dead: BaseException | None = None

    # -- liveness ----------------------------------------------------------

    def die(self, error: BaseException | None = None) -> None:
        """Make the engine dead.

        ``errored`` becomes true and ``check_health`` raises ``dead_error``.
        """
        self._dead = error or EngineDeadError()

    @property
    def errored(self) -> bool:
        return self._dead is not None

    @property
    def dead_error(self) -> BaseException:
        return self._dead if self._dead is not None else Exception("alive")

    @property
    def is_running(self) -> bool:
        return self._dead is None

    @property
    def is_stopped(self) -> bool:
        return self._dead is not None

    async def check_health(self) -> None:
        self.health_checks += 1
        if self._dead is not None:
            raise self._dead

    # -- calls the frontend makes ------------------------------------------

    async def abort(self, request_id: str | Iterable[str]) -> None:
        if isinstance(request_id, str):
            self.aborted.append(request_id)
        else:
            self.aborted.extend(request_id)

    async def get_supported_tasks(self) -> tuple[str, ...]:
        return self.tasks

    # -- test conveniences ---------------------------------------------------

    @property
    def additional_config(self) -> dict[str, Any]:
        return self.vllm_config.additional_config

    def set_capacity(
        self,
        *,
        num_gpu_blocks: int | None | object = _UNSET,
        max_num_seqs: int | object = _UNSET,
        max_num_active_seqs: int | None | object = _UNSET,
    ) -> None:
        """Change the capacity values after construction.

        Core updates ``num_gpu_blocks`` once the engine reports its cache;
        ``None`` is a legitimate value (not yet reported).
        """
        if num_gpu_blocks is not _UNSET:
            self.vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        if max_num_seqs is not _UNSET:
            self.vllm_config.scheduler_config.max_num_seqs = max_num_seqs
        if max_num_active_seqs is not _UNSET:
            self.vllm_config.scheduler_config.max_num_active_seqs = (
                max_num_active_seqs
            )

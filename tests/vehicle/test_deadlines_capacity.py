"""Engine stream deadlines, the frontend's bounds and capacity reading."""

from __future__ import annotations

import math

import pytest


def deadlines(section=None):
    from vllm_riva_frontend.config import read_stream_deadlines

    config = {} if section is None else {"stream_admission": section}
    return read_stream_deadlines(config)


# ------------------------------------------------------- engine deadlines


# @spec ING-EP-013
def test_absent_section_or_keys_take_the_documented_defaults() -> None:
    from vllm_riva_frontend.config import (
        DEFAULT_STREAM_DEADLINES,
        STREAM_ADMISSION_KEY,
        StreamDeadlines,
    )

    assert STREAM_ADMISSION_KEY == "stream_admission"
    assert deadlines() == StreamDeadlines(15, 60, 40)
    assert deadlines({}) == DEFAULT_STREAM_DEADLINES
    partial = deadlines({"never_fed_timeout": 5})
    assert (
        partial.never_fed_timeout,
        partial.parked_idle_timeout,
        partial.draining_timeout,
    ) == (5, 60, 40)


# @spec ING-EP-013
@pytest.mark.parametrize(
    "key", ["never_fed_timeout", "parked_idle_timeout", "draining_timeout"]
)
@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, None, "15"])
async def test_present_unreadable_deadline_fails_startup(
    clock, fake_factory, helpers, key, value
) -> None:
    failures = await helpers.startup_failures(
        clock,
        fake_factory,
        stream_admission={key: value},
    )
    assert helpers.mentions(failures, key)


# @spec ING-EP-013, ING-EP-068
async def test_init_state_reads_deadlines_from_additional_config(
    initialized,
) -> None:
    plugin, _ = initialized
    assert (
        plugin.deadlines.never_fed_timeout,
        plugin.deadlines.parked_idle_timeout,
        plugin.deadlines.draining_timeout,
    ) == (15, 60, 40)


# @spec ING-EP-071
@pytest.mark.engine
def test_copied_defaults_are_pinned_against_the_core_build() -> None:
    """The one frontend test that needs RFC 1 (ING-EP-072 checks that)."""
    from vllm.v1.core.stream_admission import StreamAdmissionConfig

    from vllm_riva_frontend.config import DEFAULT_STREAM_DEADLINES

    core = StreamAdmissionConfig()
    assert (
        DEFAULT_STREAM_DEADLINES.never_fed_timeout,
        DEFAULT_STREAM_DEADLINES.parked_idle_timeout,
        DEFAULT_STREAM_DEADLINES.draining_timeout,
    ) == (
        core.never_fed_timeout,
        core.parked_idle_timeout,
        core.draining_timeout,
    )


# ------------------------------------------------ bounds below deadlines


def _config(**options):
    from vllm_riva_frontend.config import parse_config

    return parse_config({"riva_frontend": options})


# @spec ING-EP-014, ING-EP-032
def test_default_bounds_pass_against_default_deadlines() -> None:
    from vllm_riva_frontend.config import DEFAULT_STREAM_DEADLINES, check_bounds

    assert check_bounds(_config(), DEFAULT_STREAM_DEADLINES) == []


# @spec ING-EP-014
@pytest.mark.parametrize(
    "option, value",
    [
        ("first_audio_timeout", 15),
        ("session_idle_timeout", 60),
        ("lease_call_timeout", 60),
        ("session_finalization_timeout", 40),
    ],
)
def test_a_bound_equal_to_its_deadline_fails(option, value) -> None:
    from vllm_riva_frontend.config import DEFAULT_STREAM_DEADLINES, check_bounds

    failures = check_bounds(
        _config(**{option: value}), DEFAULT_STREAM_DEADLINES
    )
    assert any(option in failure for failure in failures)


# @spec ING-EP-014
def test_every_bound_failure_is_reported() -> None:
    from vllm_riva_frontend.config import StreamDeadlines, check_bounds

    failures = check_bounds(_config(), StreamDeadlines(5, 20, 10))
    for option in (
        "first_audio_timeout",
        "session_idle_timeout",
        "lease_call_timeout",
        "session_finalization_timeout",
    ):
        assert any(option in failure for failure in failures), option


# @spec ING-EP-014
async def test_lowered_engine_deadline_fails_startup(
    clock, fake_factory, helpers
) -> None:
    failures = await helpers.startup_failures(
        clock, fake_factory, stream_admission={"never_fed_timeout": 8}
    )
    assert helpers.mentions(failures, "first_audio_timeout")


# @spec ING-EP-014
async def test_raised_frontend_bound_with_raised_deadline_passes(
    clock, fake_factory, helpers
) -> None:
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with(
        options={"session_idle_timeout": 80, "lease_call_timeout": 80},
        stream_admission={"parked_idle_timeout": 90},
    )
    await helpers.initialize(plugin, engine)
    assert plugin.config.session_idle_timeout == 80


# --------------------------------------------------------------- capacity


# @spec ING-EP-030
def test_capacity_is_pages_minus_one_and_active_or_max_seqs() -> None:
    from support.fake_engine import FakeEngineClient

    from vllm_riva_frontend.config import Capacity, read_capacity

    engine = FakeEngineClient(num_gpu_blocks=17, max_num_seqs=16)
    assert read_capacity(engine.vllm_config) == Capacity(16, 16)
    engine.set_capacity(max_num_active_seqs=8)
    assert read_capacity(engine.vllm_config) == Capacity(16, 8)
    engine.set_capacity(num_gpu_blocks=129, max_num_active_seqs=None)
    assert read_capacity(engine.vllm_config) == Capacity(128, 16)


# @spec ING-EP-030
async def test_init_state_records_capacity(
    clock, fake_factory, helpers
) -> None:
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with(
        num_gpu_blocks=33, max_num_seqs=64, max_num_active_seqs=20
    )
    await helpers.initialize(plugin, engine)
    assert (plugin.capacity.slots, plugin.capacity.active_limit) == (32, 20)


# @spec ING-EP-029
@pytest.mark.parametrize("pages", [None, 0])
async def test_unknown_page_count_fails_startup(
    clock, fake_factory, helpers, pages
) -> None:
    failures = await helpers.startup_failures(
        clock, fake_factory, num_gpu_blocks=pages
    )
    assert helpers.mentions(failures, "num_gpu_blocks")


# @spec ING-EP-029
async def test_unreadable_active_limit_fails_startup(
    clock, fake_factory, helpers
) -> None:
    from vllm_riva_frontend.lifecycle import StartupError

    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with()
    del engine.vllm_config.scheduler_config.max_num_seqs
    del engine.vllm_config.scheduler_config.max_num_active_seqs
    with pytest.raises(StartupError) as info:
        await helpers.initialize(plugin, engine)
    assert helpers.mentions(info.value.failures, "max_num_seqs")
    assert fake_factory.teardown_calls == 0
    assert not hasattr(fake_factory, "constructed_with")


# @spec ING-EP-028
async def test_data_parallel_above_one_fails_startup(
    clock, fake_factory, helpers
) -> None:
    failures = await helpers.startup_failures(
        clock, fake_factory, data_parallel_size=2
    )
    assert helpers.mentions(failures, "data_parallel")
    assert not hasattr(fake_factory, "constructed_with")

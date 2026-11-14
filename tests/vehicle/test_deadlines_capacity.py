"""Engine stream deadlines, the frontend's bounds and capacity reading."""

from __future__ import annotations

import math

import pytest


def deadlines(timeouts=None, **engine_kw):
    """The deadlines ``read_stream_deadlines`` reads from a fake engine."""
    from support.fake_engine import FakeEngineClient

    from vllm_riva_frontend.config import read_stream_deadlines

    engine = FakeEngineClient(stream_admission_timeouts=timeouts, **engine_kw)
    return read_stream_deadlines(engine.vllm_config)


# ------------------------------------------------------- engine deadlines


# @spec ING-EP-112
def test_absent_keys_take_the_documented_defaults() -> None:
    from vllm_riva_frontend.config import (
        DEFAULT_STREAM_DEADLINES,
        STREAM_ADMISSION_TIMEOUTS_FLAG,
        StreamDeadlines,
    )

    assert STREAM_ADMISSION_TIMEOUTS_FLAG == "--stream-admission-timeouts"
    assert deadlines() == StreamDeadlines(15, 60, 40)
    assert deadlines({}) == DEFAULT_STREAM_DEADLINES
    partial = deadlines({"never_fed_timeout": 5})
    assert (
        partial.never_fed_timeout,
        partial.parked_idle_timeout,
        partial.draining_timeout,
    ) == (5, 60, 40)


# @spec ING-EP-112
def test_deadlines_come_from_the_scheduler_setting() -> None:
    """``--stream-admission-timeouts`` is read; ``additional_config`` is not."""
    read = deadlines(
        {
            "never_fed_timeout": 30,
            "parked_idle_timeout": 90.5,
            "draining_timeout": 50,
        }
    )
    assert (
        read.never_fed_timeout,
        read.parked_idle_timeout,
        read.draining_timeout,
    ) == (30, 90.5, 50)


# @spec ING-EP-112
def test_a_core_without_the_setting_takes_the_defaults() -> None:
    """An RFC 2 build without RFC 1 has no ``stream_admission_timeouts``."""
    from support.fake_engine import FakeEngineClient

    from vllm_riva_frontend.config import (
        DEFAULT_STREAM_DEADLINES,
        read_stream_deadlines,
    )

    engine = FakeEngineClient()
    del engine.vllm_config.scheduler_config.stream_admission_timeouts
    read = read_stream_deadlines(engine.vllm_config)
    assert read == DEFAULT_STREAM_DEADLINES


# @spec ING-EP-112
@pytest.mark.parametrize(
    "key", ["never_fed_timeout", "parked_idle_timeout", "draining_timeout"]
)
@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, None, "15", True])
async def test_present_unreadable_deadline_fails_startup(
    clock, fake_factory, helpers, key, value
) -> None:
    failures = await helpers.startup_failures(
        clock,
        fake_factory,
        stream_admission_timeouts={key: value},
    )
    assert helpers.mentions(failures, key, "--stream-admission-timeouts")


# @spec ING-EP-112
async def test_a_setting_that_is_not_a_mapping_fails_startup(
    clock, fake_factory, helpers
) -> None:
    from vllm_riva_frontend.lifecycle import StartupError

    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with()
    engine.vllm_config.scheduler_config.stream_admission_timeouts = [15]
    with pytest.raises(StartupError) as info:
        await helpers.initialize(plugin, engine)
    assert helpers.mentions(
        info.value.failures, "stream_admission_timeouts", "mapping"
    )


# @spec ING-EP-114
@pytest.mark.parametrize(
    "section", [{"never_fed_timeout": 30}, {}, None, "anything"]
)
async def test_a_stream_admission_key_in_additional_config_fails_startup(
    clock, fake_factory, helpers, section
) -> None:
    """The key core never read is refused, naming the setting core reads."""
    failures = await helpers.startup_failures(
        clock,
        fake_factory,
        additional_config={"stream_admission": section},
    )
    assert helpers.mentions(
        failures, "stream_admission", "--stream-admission-timeouts"
    )
    assert not hasattr(fake_factory, "constructed_with")


# @spec ING-EP-114
async def test_the_refused_key_is_reported_with_an_unreadable_deadline(
    clock, fake_factory, helpers
) -> None:
    failures = await helpers.startup_failures(
        clock,
        fake_factory,
        additional_config={"stream_admission": {}},
        stream_admission_timeouts={"draining_timeout": 0},
    )
    assert helpers.mentions(failures, "'stream_admission'")
    assert helpers.mentions(failures, "draining_timeout")


# @spec ING-EP-114, ING-EP-014
async def test_the_refused_key_is_reported_with_the_bound_failures(
    clock, fake_factory, helpers
) -> None:
    """A readable deadline below a bound and the refused key: both reported."""
    failures = await helpers.startup_failures(
        clock,
        fake_factory,
        additional_config={"stream_admission": {}},
        stream_admission_timeouts={"never_fed_timeout": 8},
    )
    assert helpers.mentions(failures, "'stream_admission'")
    assert helpers.mentions(failures, "first_audio_timeout")


# @spec ING-EP-112, ING-EP-068
async def test_init_state_reads_deadlines_from_the_scheduler_config(
    initialized,
) -> None:
    plugin, _ = initialized
    assert (
        plugin.deadlines.never_fed_timeout,
        plugin.deadlines.parked_idle_timeout,
        plugin.deadlines.draining_timeout,
    ) == (15, 60, 40)


# @spec ING-EP-112
async def test_init_state_uses_a_raised_scheduler_deadline(
    clock, fake_factory, helpers
) -> None:
    plugin = helpers.make_frontend(
        clock, factories={"nemotron": fake_factory.constructor}
    )
    engine = helpers.engine_with(
        stream_admission_timeouts={"draining_timeout": 55}
    )
    await helpers.initialize(plugin, engine)
    assert plugin.deadlines.draining_timeout == 55


# @spec ING-EP-071
@pytest.mark.engine
def test_copied_defaults_are_pinned_against_the_core_build() -> None:
    """The one frontend test that needs RFC 1 (ING-EP-072 checks that).

    It pins the frontend's copy of core's defaults, the field name the
    frontend reads, and that a value core accepts reaches the frontend.
    """
    from types import SimpleNamespace

    from vllm.config import SchedulerConfig

    from vllm_riva_frontend.config import (
        DEFAULT_STREAM_DEADLINES,
        read_stream_deadlines,
    )

    assert dict(SchedulerConfig.DEFAULT_STREAM_ADMISSION_TIMEOUTS) == {
        "never_fed_timeout": DEFAULT_STREAM_DEADLINES.never_fed_timeout,
        "parked_idle_timeout": DEFAULT_STREAM_DEADLINES.parked_idle_timeout,
        "draining_timeout": DEFAULT_STREAM_DEADLINES.draining_timeout,
    }
    scheduler = SchedulerConfig.default_factory(
        stream_admission_timeouts={"never_fed_timeout": 30}
    )
    read = read_stream_deadlines(SimpleNamespace(scheduler_config=scheduler))
    assert (
        read.never_fed_timeout,
        read.parked_idle_timeout,
        read.draining_timeout,
    ) == (30, 60, 40)
    assert (
        read_stream_deadlines(
            SimpleNamespace(scheduler_config=SchedulerConfig.default_factory())
        )
        == DEFAULT_STREAM_DEADLINES
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
        clock,
        fake_factory,
        stream_admission_timeouts={"never_fed_timeout": 8},
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
        stream_admission_timeouts={"parked_idle_timeout": 90},
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

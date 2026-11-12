"""Fixtures for the audio leaf.

Most audio tests drive ``frontend`` directly. The hand-off tests (what the
lease receives from a stream, and the WAV ``Recognize`` hands core) run the
started plugin (the shared ``started`` fixture) with the Riva servicer called
directly through ``support.riva``: ``FakeClock`` time, ``FakeSessionFactory``
leases and ``FakeTranscriptionServing`` for core's service. The credit test
reads ``StreamOwner.outstanding_samples`` on an owner built directly (the
shared ``make_owner`` fixture).
"""

from __future__ import annotations

from typing import Any

import pytest
from support.riva import StreamingCall


@pytest.fixture
def plugins(plugin):
    return [plugin]


async def open_stream(plugin: Any, **fields: Any) -> StreamingCall:
    """A ``StreamingRecognize`` configured with ``fields``, once opened.

    ``fields`` build the ``RecognitionConfig`` (``support.riva``); the call
    returns once the factory opened the lease or the handler ended.
    """
    call = StreamingCall(plugin.servicer, plugin.clock)
    call.configure(**fields)
    await call.until(lambda: bool(plugin.factory.leases), what="open")
    return call

"""Fixtures for the operations leaf.

The four operational routes and the bootstrap route are exercised over
``test_app`` (core's routers, middleware and plugin wiring) with ``httpx``;
readiness needs the plugin started, so the gRPC listener binds a free port.
``riva_options`` is the ``riva_frontend`` mapping a test adds.
"""

from __future__ import annotations

import pytest

# ``riva_options``, ``fake_engine``, ``plugin`` and ``started`` are the
# shared fixtures of ``tests/conftest.py``; this leaf serves the plugin.


@pytest.fixture
def plugins(plugin):
    return [plugin]


@pytest.fixture
def client(test_app):
    from support.app import http_client

    return http_client(test_app)

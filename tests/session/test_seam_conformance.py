"""The seam conformance suite (vllm_riva_frontend_dev) against the fake factory.

The suite's tests live in ``vllm_riva_frontend_dev.seam_conformance``; this
module runs them in the frontend's CPU lane with ``seam`` returning
``support.seam.FakeSeam`` over ``FakeSessionFactory`` (the ``fake_seam``
fixture), on the plugin's ``FakeClock``. The model plugin's repository runs
the same suite against its real factory with its own ``seam``.
"""

import pytest
from vllm_riva_frontend_dev.seam_conformance import *  # noqa: F403


@pytest.fixture
def seam(fake_seam):
    return fake_seam

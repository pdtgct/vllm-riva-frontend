# vllm-riva-frontend-dev

The seam conformance suite of `vllm-riva-frontend`, released at the frontend's
version and pinning it exactly. It is test-only: install it in a test
environment, never in a deployment.

A consumer (a model plugin's test suite) runs it against its own session
factory:

1. Add `vllm-riva-frontend-dev==<frontend version>` to its test dependencies.
2. In `conftest.py`, call
   `pytest.register_assert_rewrite("vllm_riva_frontend_dev.seam_conformance")`
   before the suite is imported.
3. In a test module, `from vllm_riva_frontend_dev.seam_conformance import *`.
4. In that module, define the `seam` fixture, returning its adapter.
5. The adapter provides the hooks listed in the `vllm_riva_frontend_dev`
   package docstring, over the factory under test.
6. Running the tests needs what the frontend imports, a vLLM tree included;
   collecting them does not.

The contract is the frontend repository's session design, section
"Seam conformance" (`docs/intent/session/session-design.md`).

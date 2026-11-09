"""Shared fakes and helpers for the frontend's tests.

The modules here import core (the vLLM tree on ``PYTHONPATH``) and the test
dependencies, never the frontend package: ``tests/test_support.py`` checks
them on their own. Each leaf's tests build on them through the fixtures in
``tests/conftest.py``.

* ``clock``: the ``Clock`` seam (``FakeClock`` with virtual time,
  ``RealClock``).
* ``fake_session``: ``FakeSessionFactory`` and ``FakeLease``, the structural
  session seam, scriptable and recording.
* ``fake_engine``: ``FakeEngineClient`` with core's configuration attribute
  names.
* ``fake_transcription``: ``FakeTranscriptionServing`` and
  ``FakeServingModels``, the core serving objects the file path uses.
* ``state``: the Starlette ``State`` core's ``init_app_state`` builds.
* ``app``: a FastAPI app with core-like routes, core's middleware and the
  endpoint-plugin wiring, wrappers around core's lifecycle functions,
  ``serve_plugin`` (build, state, ``init_state``, ``start`` in one call) and
  ``drain_plugin`` (one plugin's ``drain`` under ``FakeClock``).
* ``asgi``: in-loop ASGI drivers for WebSockets (``AsgiWebSocket``,
  ``BlockingSendWebSocket``) and raw HTTP (``asgi_http``).
* ``http``: ``HttpExchange`` (raw HTTP with test-controlled chunk timing,
  disconnects and blocked or failing sends), multipart body builders and
  core's error-shape assertion.
* ``grpc_harness``: a fake servicer context, a test-driven request stream
  and a real ``grpc.aio`` server on a free port.
* ``riva``: Riva message builders over the pinned stubs, ``StreamingCall``
  (the one ``StreamingRecognize`` driver) and ``recognize``.
* ``realtime``: Realtime event builders in the public shape,
  ``RealtimeClient`` (the one Realtime driver) and the readers of the
  pinned ``error`` object.
* ``audio``: deterministic PCM16, mu-law, A-law and RIFF/WAV bytes.
"""

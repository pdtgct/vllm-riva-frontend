---
parent: high-level-design
prefix: ING
facets: [CORE, LIFE]
---

# Session

## Context and Design Philosophy

A live stream (Riva `StreamingRecognize`, a Riva Realtime WebSocket session) runs on a session
factory that the model plugin publishes. This leaf is the frontend's side of that seam and the
per-stream owner that drives it: how the factory is found and checked, which calls the frontend
makes on it and in what order, how a stream ends, and which time bounds apply. Wire formats belong
to the streaming leaf; the lease calls themselves are specified by the model plugin's session
design (`SESS-*`), which this leaf consumes and does not restate.

Principles:

- **Structural coupling only.** The frontend and the model plugin import neither each other nor
  each other's exceptions. The factory and lease are protocols; error information travels as two
  attributes.
- **One owner per stream, one terminal cause.** A stream has one serialized owner. It records
  exactly one winning cause for how the stream ends and never overlaps two lease calls.
- **Finish or abort, never both; release always.** The two ways to end an engine stream are
  mutually exclusive; the resource is always released, or handed to the factory.
- **Direct translation.** Each surface translates its dialect straight to lease calls and results.
  There is no generic session event bus, and no surface re-implements what the model's session
  owns.

## Responsibility split

The frontend owns decode and resample, the bounded buffer between a transport read and `feed`,
read-side flow control, wire projection, and the transport owner's lifecycle. The model plugin's
session owns the packer (cadence segmentation, `chunk_sequence`, the final-tail unit), the engine
binding, and the consequences of a failed unit (`SESS-PACK-*`, `SESS-ENG-*`, `SESS-LEASE-005`).
No surface duplicates those behaviours: there is no accepted-audio queue in the frontend beyond the
pre-submit bound (audio leaf), no cadence arithmetic, no batcher, no generation or render loop and
no internal WebSocket loopback. Complete files do not use the factory (files leaf).

Every streaming surface binds only through the factory and lease the model plugin publishes, in
process. It implements no remote provider, provider-mode switch or generic session broker.

## Discovery and selection

The entry-point group `vllm_riva_frontend.session_factories` is owned and documented by this
package. The model plugin registers a constructor under it. `init_state` calls every constructor in
the group with `(engine_client, engine_client.model_config)`, unchanged.

- A constructor returns `None` when it does not serve the model.
- Exactly one constructor must return a factory, unless `additional_config.riva_frontend.session_factory`
  names an entry point, in which case only that constructor's result is used and an absent or
  declining named factory fails startup. Zero or several accepting factories without a selection
  fail startup, listing the entry-point names.
- Constructors start no task or session; a factory is inert until its first `open`. Every factory
  a constructor returned belongs to the vehicle, which settles a discarded one when it discards it
  and every constructed one when startup fails (vehicle leaf, startup checks).
- A constructor that raises fails startup with its message. That raise is the frontend's only
  engine-capability check: the engine-client protocol does not declare the stream-admission calls,
  so only the factory, which uses them, knows whether the engine has them.

## Structural protocol

The frontend uses the following surface and nothing else.

| Object | Member | Use |
| --- | --- | --- |
| factory | `served_cadences: frozenset[str]` | startup check; cadence selection |
| factory | `locales: frozenset[str]` | locale validation on every surface |
| factory | `default_locale: str \| None` | default when the client names none |
| factory | `open(*, cadence, locale)` | opens a lease; may raise a refusal or failure |
| factory | `drain(deadline) -> list` | await orphan cleanup until an absolute `time.monotonic()` deadline (`math.inf` in the vehicle's factory settlement); returns the retained task faults (Factory obligations 3 and 4), each exactly once, clearing them |
| factory | `teardown_all() -> set[Task]` | terminal close, called after `drain(math.inf)`; cancels and returns every task the factory owns (generation and cleanup tasks), which the vehicle awaits without reading their exceptions |
| lease | `feed(samples, *, on_accepted) -> list[str]` | one call per piece; returns cumulative transcripts, one per unit whose text changed |
| lease | `update_locale(locale) -> None` | locale change |
| lease | `flush() -> str` | final tail; returns the final cumulative transcript or raises |
| lease | `finish()`, `abort()`, `release()` | terminal calls (`release` returns after the slot has retired) |
| lease | `teardown()` | synchronous, engine-free; not called by the frontend, whose owners end a session through `abort` and `release` in every path, including `stop` |

Errors: an exception from `open` or a lease may carry `session_error` (`"refused"` or `"failed"`)
and `reason`. The frontend reads both with `getattr` and treats an exception with neither as an
unexpected failure (errors leaf maps every reason). It catches no model-plugin exception class by
name.

## Cadence and locale

Streaming surfaces open at `560ms`. `init_state` fails startup if `served_cadences` does not
contain it; the factory and the model separately refuse an unserved cadence, so a stream never
opens at one.

A stream's locale is the client's, validated against `locales`. When the client names none, the
locale is `default_locale`. When that is `None`, the client must name one, and a client that names
none is refused with `invalid_config_field` naming `language_code` (Riva) or the dialect's own
field (Realtime). A name outside `locales` is `unknown_locale`. The same rules govern the Realtime
bootstrap response and a Realtime update (streaming leaf), and the file path (files leaf).

## Seam conformance

The repository ships a conformance suite for the seam, run against a fake factory implementing the
protocol; the model plugin repository runs the same suite against its real factory. The fake
factory models cleanup and holding tasks that outlive the owner that handed the session over, so
the CPU tests exercise a settlement that must await factory work no owner still holds.

The suite checks what the model plugin's session specification states: `drain` takes an absolute
deadline, awaits the cleanup tasks until they finish or the deadline passes, and returns the
recorded cleanup faults once, clearing them (`SESS-OWN-004`); `teardown_all` marks the factory
closed, cancels every task it owns and returns them (`SESS-OWN-005`), including the generation task
of a session whose `abort` was cancelled and the cleanup task of a session whose `release` was
cancelled; once closed, `open` and every lease call other than `teardown` fail (`SESS-OWN-006`);
a `release` after a cancelled `abort` keeps a tracked task until the generation task ends
(`SESS-ABORT-002`); a cancelled `release` leaves the session with the factory's cleanup
(`SESS-REL-003`, `SESS-OWN-003`); the call order the owner relies on; and the error attributes,
including a reason the frontend does not know.

### Factory obligations

The frontend imposes the following obligations on every factory. This list is the canonical one;
the vehicle leaf refers to its items by number. The vehicle's factory settlement relies on them,
the suite checks them, and each is owed as a statement in the model plugin's session
specification.

1. **Infinite drain.** `drain` accepts `math.inf` as its deadline and then returns only when every
   cleanup and holding task that exists when it is called has ended, cancelling no task by the
   wait. The vehicle calls it after every owner has ended, so no lease call can register a further
   task during the wait and the call awaits all of them.
2. **Ordered ending.** Each cleanup task and each holding task (the task that holds a session after
   a failed retirement wait) ends only after its session's generation task has ended, and none
   cancels it.
3. **Retained faults.** The factory retains, and returns through `drain`'s fault list, exactly
   these faults, even after their session is dropped:
   1. an exception, other than `CancelledError`, of a cleanup task's retirement wait that reports
      a dead engine or is unexpected;
   2. an exception, other than `CancelledError`, of a generation task that no `open` or lease call
      surfaced to its caller: the task ended after its session was handed over to a cleanup or
      holding task, or its cancellation handler raised while an uncancelled `abort` awaited it
      (`abort` then returns normally);
   3. an exception, other than `CancelledError`, of a cleanup or holding task itself.

   Anything surfaced to its caller by `open` or a lease call is excluded: a refusal that `open`
   raises with `session_error` `"refused"`, and a generation failure projected as a lease `reason`
   (`engine_dead` among them) or raised with `session_error` `"failed"`. In the model plugin these
   are, for example, a `StreamAdmissionError` raised as `SessionRefused`, and an `EngineDeadError`
   projected as reason `engine_dead` or raised as `SessionFailed`. A cleanup or holding task that
   `teardown_all` cancels as a last resort ends with `CancelledError` and is not retained: the
   vehicle alone reports it, as a last-resort cancellation fault, and `drain` never returns it.
4. **Sole reporter.** The factory is the only reporter of its tasks' faults: `drain` returns each
   retained fault exactly once and clears it, so a fault returned by one caller is never returned
   to another.
5. **Cancellation scope.** `open`, `abort`, `release` and the factory's own cleanup and holding
   tasks deliver no cancellation to a generation task after its cancellation handler has begun.
   `teardown_all` and the lease's `teardown()` cancel unconditionally by contract (`SESS-OWN-005`,
   `SESS-TEAR-001`) and are excepted; that no cancellation reaches a generation task in its
   handler across the whole system follows from the vehicle's settlement order, which calls
   `teardown_all` only after every handed-over generation task has ended.
6. **Release hand-over.** `release` does not suspend before its cancellation-safe hand-over
   coverage is in place, so a cancellation delivered at any suspension after the call has started
   hands the session over to the factory's cleanup. `SESS-REL-003` covers a cancellation before the
   retirement wait finishes; this obligation adds that no earlier suspension exists.
7. **Closed means inert.** The factory starts no task after `teardown_all`.

The suite checks each obligation, and includes these tests:

- an ordinary capacity refusal (`open` raises `SessionRefused`) and an ordinary generation failure
  (seen by the owner as a lease reason) leave `drain`'s fault list empty;
- a generation task whose cancellation handler raises while it awaits the engine abort, after its
  session was handed over, yields exactly one fault, returned by a later `drain` after the cleanup
  task has ended and dropped the session, and by no later `drain`;
- a generation task whose cancellation handler raises while an uncancelled `abort` awaits it: the
  `abort` returns normally, and exactly one fault is returned by a later `drain` and by no later
  `drain`;
- a cancelled `open`, which requests cancellation twice with no suspension between the requests,
  after its generation task has started: the generation task receives no cancellation after its
  cancellation handler has begun, a holding or cleanup task covers it, and no factory task is still
  running after the vehicle's factory settlement;
- a cancelled `open` whose generation task is cancelled before its first step: the task never runs
  its body, makes no engine call, and ends within one turn of the event loop;
- the release hand-over is tested through the frontend's own cancellation mechanism (Cleanup
  order), run against the real factory and the fake: the owner runs `release` as a
  task, lets it reach its first suspension, cancels it and awaits it, and the session is then with
  the factory's cleanup. A fake instrumented to cancel at its own retirement wait would not test
  obligation 6.

A factory that fails these cannot be settled without cancelling a generation task while it awaits
the engine client's abort, or without losing or repeating a task's fault.

Obligations 3 and 4 need a change to the model plugin's code: its done-callback retrieves a
generation task's exception without recording it, its recorded faults are the retirement-wait
failures alone, and its cleanup drops the session when it ends. The others are true of its code
today and owe only statements in its specification (vehicle leaf, Dependencies).

## The stream owner

One owner task serves one stream. It registers before reading anything (vehicle leaf), and its
states are:

| State | Entered when | Meaning |
| --- | --- | --- |
| `awaiting_configuration` | registration | only the first message may be read |
| `configured` | `open` returned a lease after complete validation | the lease exists; the transport is consumed |
| `streaming` | first message after configuration | audio is accepted |
| `finalizing` | finalization accepted: client half-close or `done`, or shutdown finalization | no more audio, updates or finals accepted; later traffic is `session_terminal` |
| terminal | cleanup done | the owner unregisters |

### First message and configuration

The first client message on every streaming surface is that dialect's configuration. Any other
message first is `protocol_order`: gRPC `FAILED_PRECONDITION`; Realtime `error`, then close.

The handler reads exactly that message, validates it fully (either a concrete accepted audio cell,
or the dialect's explicit deferred-RIFF declaration; the streaming leaf lists the checks), calls
`open`, and only then advances the gRPC request iterator or receives another WebSocket message,
with one exception: after a rejected initial configuration the Realtime handler stays
unconfigured and receives further `transcription_session.update` events until one validates and
`open` succeeds, all within the original `preconfiguration_timeout`, which a retry does not
reset. No audio is admitted before `open` succeeds; any other event in that interval is the
wrong-first-message case.
Pipelined audio before `open` remains under transport flow control and enters no frontend buffer.
In deferred-RIFF mode the handler holds at most `max_riff_header_bytes` after `open` and calls no
`feed` until the concrete cell resolves.

When `open` returns, the engine has admitted the stream. The owner enters `configured` before the
transport is consumed. Riva emits no invented configuration or admission acknowledgement. The
Realtime adapter's required `transcription_session.updated` acknowledges the validated
configuration and adds no admission vocabulary. A `SessionRefused` for capacity is answered `busy`
with `authority: provider` and consumes no audio (vehicle leaf).

A finite `preconfiguration_timeout` begins when the owner registers and ends only when the complete
configuration validates and `open` succeeds. Expiry is `configuration_timeout`: the owner cancels
a pending `open`, releases its registration and creates no lease. It calls neither `abort` nor
`release` for a lease it never received; a cancelled `open` is the factory's to clean up
(`SESS-OPEN-006`). The bootstrap POST creates no owner and starts no timer.

### Hypotheses

Each cumulative transcript string a `feed` call returns is projected by the surface:

- Riva emits a changed cumulative hypothesis as a non-final result only when `interim_results` is
  true.
- Realtime emits the true incremental suffix between consecutive cumulative hypotheses.
- A `feed` that returns no transcript (no unit completed, or no unit changed the text) emits
  nothing.

### Finalization

Three triggers select a finalization: client half-close (Riva), `input_audio_buffer.done`
(Realtime), and **shutdown finalization**, which the vehicle leaf raises at the drain settle point.
Each produces exactly one dialect final, after normal finish and release succeed; multi-utterance reuse
is not part of this design. The surface then closes. A finalization with zero audio is a valid
degenerate stream: it calls `flush`, which sends the zero-sample final tail, then `finish` and
`release`, and yields one empty final. A finalization while a deferred RIFF header is still
unresolved is split: if zero bytes were received, it is a zero-audio stream with a valid empty
final; if a partial header was received, the stream ends with `unsupported_format` naming the
fields, with `abort` and `release` and no final.

For every streaming surface, normal finalization is one sequence, derived at one point shared by
all surfaces:

0. Feed the audio frontend's drained tail (the resampler's lookahead and any byte residual, audio
   leaf) to the lease, under the finalization bound. A residual that is not a whole sample is
   discarded.
1. `flush`, always, including for zero audio. Its return value must be a string; that string is the
   final transcript. Any other return is a failed `flush`.
2. After a successful `flush`: `finish` exactly once, then `release`.
3. Only after both succeed does the surface expose the dialect final. A failure of either emits no
   final; its projection is in the errors leaf.

If `flush` fails, or cancellation, a deadline, a disconnect, an idle timeout or an abort during
shutdown wins before normal completion, the owner uses only the calls the lease's state allows: an
open or failed lease takes `abort`, then `release`, and never `finish`; a lease that has already
been flushed is never aborted (the flushed state allows only `finish` and `release`),
and its cleanup goes to `release`. `finish` and `abort` are mutually exclusive, and `release` never
precedes the selected terminal cleanup.

**Traffic after client-selected finalization.** Once the owner accepts the half-close (Riva) or
`input_audio_buffer.done` (Realtime), it keeps reading the transport with a receive observer until
the final's send begins. The observer runs under the finalization bound and holds at most one
received message. It reads during the final-tail `feed` and `flush` too: this is the one exception
to the rule that a handler stops reading while `feed` awaits the engine. On the Realtime adapter its
dispositions are:

- a malformed or unknown envelope keeps `invalid_event` and close 1008 (streaming leaf);
- a message above the receive cap keeps `request_too_large` and close 1009 (each of these, like
  the next, runs the state-aware cleanup and suppresses a final whose send has not begun);
- every other client message (`input_audio_buffer.append`, `clear`, `commit` or `done`, or
  `transcription_session.update`) fails the session with `session_terminal` (state-aware abnormal
  cleanup) and suppresses the final if its send has not begun. When the observer fires after
  `release`, no lease call is made.

Once the final's send has begun, the final is irrevocable: the owner reads no further traffic and
the connection closes. On gRPC the half-close ends the request stream, so the observer sees only
cancellation. The observer is cancelled when the owner observes the shutdown trigger, including
when the trigger finds this finalization running: after the trigger the owner reads nothing
(below).

### Shutdown finalization

At the drain settle point the vehicle leaf raises a shutdown trigger with the drain deadline; it
owns only that trigger and deadline. The trigger preserves every completed step and the selected
cause; what the owner does depends on its state:

| Owner state at the trigger | Transition |
| --- | --- |
| waiting for configuration or first audio, no lease | nothing to finalize: end the stream with the shutdown code (errors leaf, `service_unavailable`) |
| `open` pending | let it complete or cancel it under the cancelled-`open` rule; then treat the result as no lease (end with the shutdown code) or as an open lease (below) |
| open or idle with a lease | the sequence below |
| already finalizing (client-selected) | cancel the receive observer; continue from its current step; never re-feed the tail or re-`flush`; the final then follows the normal path, with that finalization's own deadlines, codes and outcome (waits clipped to the drain deadline): on success the final is delivered and the stream ends as a client-selected finalization ends (gRPC `OK`; WebSocket close 1000, not 1001); a wait that ends because it reached the drain deadline before its own deadline ends it with the shutdown abort row |
| abnormal cleanup already selected | continue it; the cause stays the selected one |

For an open lease the stream's own owner runs the same state-aware sequence as any finalization,
serialized with its other work:

1. On observing the trigger the owner selects provisional shutdown mode, before anything else.
2. It wakes if it is blocked in a transport receive by cancelling that receive, not a lease call. If
   a lease call is outstanding it first awaits the call's settlement under the call's own deadline,
   the wait clipped to the drain deadline; two lease calls never overlap.
3. It enters `finalizing` for shutdown finalization. After the trigger the
   owner reads no further transport message, and unread traffic is not classified:
   `session_terminal` applies to client-selected finalization only.
4. It runs the finalization sequence (step 0 to 3 above) under the time remaining to the drain
   deadline. The final is delivered only after `release` succeeded, then the stream ends as the
   vehicle leaf states (gRPC `OK`; WebSocket close 1001).
5. If the settlement or the sequence fails, its finalization bound or a deadline the owner already
   held expires, or the drain deadline arrives first, the cause becomes an abort during shutdown:
   an open or broken lease takes `abort` then `release`; a lease already flushed takes `release`
   only. The stream ends with the failure of the errors leaf's phase table (`UNAVAILABLE`;
   WebSocket close 1012). Once the drain deadline has passed, these cleanup calls run under
   `session_cleanup_timeout` (Cleanup order, below), not under the time remaining to it.

**Cause classification.** The owner selects provisional shutdown mode at the instant it observes
the trigger, before it settles an outstanding call, unless a client-selected finalization is
already running (it keeps its own codes, table above) or an abnormal cause was already selected
(it is kept). Provisional shutdown mode is not an immutable cause: any failure or deadline expiry
after the trigger is observed, including during the settlement of the outstanding call (a
`lease_call_timeout`, the finalization bound), ends the stream with the shutdown abort row
(`service_unavailable`, `UNAVAILABLE`, close 1012) because the session is being shut down, not with
the row that failure or deadline has outside shutdown. Every deadline keeps its original absolute
value and identity; only the waits are clipped to the drain deadline.

A client disconnect during shutdown finalization wins as usual and selects the abnormal path.

### Locale updates and configuration changes (Realtime)

A later update that changes locale is validated and run through the serialized
`update_locale`. An unknown locale answers `unknown_locale` and keeps the prior selection. A valid
locale is acknowledged only after the call succeeds and applies to every unit packed after the
call (`SESS-LOC-001`, `SESS-LOC-002`). Accepted finalization freezes the current locale and
rejects every later update.

After the first successful update and open, every changed cadence, audio-format, sample-rate,
channel or recognition-option field answers `config_change_rejected` and leaves the active lease
and front end unchanged, even when no audio has arrived. Every later update before accepted
finalization ends with exactly one `transcription_session.updated`: the adapter emits the field
rejection events first and then the acknowledgement reflecting exactly the honoured subset (an
honoured locale change, or the unchanged session when nothing is honoured). The acknowledgement is
last, truthful and has no admission claim. A later Riva `streaming_config` is terminal
`protocol_order` with abnormal lease cleanup.

## Terminal cause and cleanup

### One cause

Every adapter funnels terminal selection through its serialized owner. The owner records exactly one
winning cause at an owner-operation boundary:

- a transport cancellation, or an expired deadline, already observed before the next boundary wins
  there; among expired deadlines the earliest absolute deadline wins, a phase's code requires that
  phase's own unclipped deadline to have expired, a phase deadline wins a tie between unclipped
  deadlines, and an abnormal cause already selected is kept (errors leaf); a provisional normal or
  shutdown finalization is not such a cause and yields to a later abnormal cause or deadline;
- otherwise an already-computed validation or protocol result, or the first timer, shutdown,
  factory, input or engine cause recorded at that boundary, stays primary;
- later transport or cleanup failures are secondary evidence and never replace the primary catalog
  code.

After selecting an abnormal cause (a transport cancellation, disconnect, deadline, timer,
validation, input or engine cause), the owner cancels and awaits bounded settlement of its current
`open`, `feed`, `update_locale`, `flush` or `finish`, and only then runs terminal cleanup, shielded
from further cancellation. Shutdown finalization is the one exception: it settles the current call
(awaits it under its own deadline, the wait clipped to the drain deadline) instead of cancelling it,
because its purpose is a complete final; if the call fails or the deadline arrives, the cause
becomes an abort during shutdown and the abnormal rule applies. Two lease calls never overlap, and
every lease call is serialized by the owner.

### Cleanup order

| Selected outcome | Calls |
| --- | --- |
| Normal | `flush` (success), `finish`, `release`; wire final after both |
| Any other cause, lease `open` or `failed` | `abort`, then `release` |
| Any other cause, lease `flushed` | `release` only: a flushed lease does not accept `abort` |
| Cause selected while `open` is pending | cancel `open`; no lease call (the factory owns the cancelled `open`) |
| `finish` fails or is cancelled after a successful `flush` | `release` only: the stream already ended normally at `flush`, and a flushed lease does not accept `abort` |
| cleanup bound expires, before or after the drain deadline | an outstanding `release` is cancelled; an outstanding `abort` is cancelled and its settlement awaited, then `release` begins and is cancelled. The owner runs `release` as a task, lets it reach its first suspension, never cancels it before its first step, and awaits the cancelled `release` before the owner or its continuation counts as ended. Either cancelled `release` is the hand-over to the factory's cleanup (`SESS-REL-003`, Factory obligation 6) |
| `stop` (torn-down plugin) | the shutdown abort row, as in the drain settlement: the rows above by lease state, under `session_cleanup_timeout` |

If `finish` settles successfully, the normal engine terminal stands and cleanup proceeds only to
`release`; a transport failure arriving afterwards suppresses the wire final but does not turn the
stream into an abort.

Terminal cleanup runs under one finite positive `session_cleanup_timeout`, shared by its calls; while
the plugin is draining and the drain deadline has not passed, the bound is instead the time
remaining to the drain deadline. Cleanup that starts in the vehicle's settlement, after the drain
deadline or when a drain fails or is cancelled, runs under `session_cleanup_timeout` from its
start, because the time remaining to the deadline is then zero and would skip `abort` and
`release`. The owner always reaches `release`, in and out of shutdown: if `abort` consumed the
budget, the owner cancels `abort` and awaits that call's settlement, then starts `release` and
cancels it, which hands the session to one tracked cleanup task of the factory (`SESS-REL-003`,
`SESS-OWN-003`). The rule is stated in what the owner can observe. The owner runs `release` as a
task and lets it run to its first suspension before any cancellation, whatever budget is left (the
current mechanism yields one step of the event loop after creating the task,
`src/vllm_riva_frontend/lease.py`, `_cleanup_call`). It never cancels a `release` before the call's
first step: a call cancelled then never runs, so no cleanup task would own the session. Factory
obligation 6 makes every suspension after the first step a safe point, so the owner needs no signal
from the factory that a particular wait has been reached. The cleanup task is registered when the
cancellation is delivered inside the call, so the owner awaits the cancelled `release` before it or
its continuation counts as ended; the vehicle's factory settlement relies on every hand-over being
registered by then. The conformance test for this runs the owner's own mechanism against the real
factory and the fake (Factory obligations). A cancelled `abort` leaves the lease `aborted`, where
`release` is allowed (`SESS-ABORT-002`). When the cleanup bound expires, the owner stops waiting and the
factory's cleanup owns the session. This is not an escalation (vehicle leaf). A cleanup failure
whose reason is `engine_dead` is an escalation; any other cleanup failure is recorded and never
replaces the primary result.

The hand-over is not by itself the shutdown guarantee. The factory's cleanup task has no deadline of
its own, and a generation task cancelled by `abort` can still be running when the owner ends. The
generation task's cancellation belongs to `abort` or to a cancelled `open`, and the task receives no
cancellation after its cancellation handler has begun. In `drain`, `stop` and a rollback the vehicle
settles in one order (vehicle leaf, Settlement): owner settlement, then the awaited continuations
and unsettled sends, then the fault collector's end, then `factory.drain(math.inf)`, which awaits
every handed-over cleanup task and, through it, the end of its generation task, then
`teardown_all()` and an await of every task it returns, then one more `factory.drain(now)` for the
faults of tasks that ended meanwhile. By then every returned task has ended, so `teardown_all`'s
cancellation has no effect; a task still running is cancelled by it as a last resort and recorded
by the vehicle as a fault. No engine-facing work of a session
outlives `drain` or `stop`, whichever path its owner took.

If a primary cancellation, deadline, validation, transport or engine error is selected, a later
cleanup error is recorded without replacing the primary wire result. On an otherwise normal path, a
`finish` failure, a `release` failure or a cleanup-bound expiry emits no final and does not claim
the resource was released; its code, status and close are in the errors leaf's phase table.

### Disconnect

A detected disconnect (WebSocket disconnect, TCP FIN or RST, gRPC stream cancellation) selects
abnormal termination immediately. gRPC observes cancellation through the context's done callback at
once, even during a lease call. A WebSocket disconnect during an outstanding call is observed when
that call returns or its per-operation bound expires, whichever is first, because the adapter reads
the transport only between calls, except for the receive observer of client-selected finalization
(above), which observes it at once. Once observed: shielded, bounded `abort` and `release`. The
session is never left to the engine's idle deadlines. An undetected half-open connection falls to
the idle bound below, the only application-level mechanism available; transport keepalive may turn
it into a detected drop sooner but is never the activity clock.

## Bounds

Every lease call and every wait for client input runs under a finite frontend bound.

| Bound | Starts | Resets | Expiry |
| --- | --- | --- | --- |
| `preconfiguration_timeout` | owner registered | never | `configuration_timeout` |
| `first_audio_timeout` | successful `open` | not reset; replaced by the idle bound on first accepted audio | `idle_timeout` |
| `session_idle_timeout` | first accepted audio | each successful application-level audio acceptance | `idle_timeout` |
| `lease_call_timeout` | a `feed` or `update_locale` call starts | never | `service_unavailable` (retryable); the lease is broken (`abort`, `release`) |
| `max_session_duration` (optional) | successful `open` | never | `request_timeout` |
| `session_finalization_timeout` | finalization accepted | never | `finalization_timeout` |
| `session_cleanup_timeout` | terminal cleanup begins, including the settlement of a cancelled call | never | hand-over to the factory (above), an unsettled call and the rest of cleanup staying under the record's continuation, which the vehicle's settlement awaits before its factory settlement awaits the factory's cleanup through `drain(math.inf)` and then calls `teardown_all()` |
| `response_send_timeout` | a send or close starts | never | transport failure: no further sends; cleanup as for a disconnect |

"Accepted audio" is a `feed` call whose `on_accepted` fired, not socket bytes, keepalive or rejected
overflow. Client-idle timing (`first_audio_timeout`, `session_idle_timeout`) is suspended while a lease call
is outstanding: time spent waiting for the engine is not the client's idleness, and the client's
audio is held back by the owner's flow control, not by the client. Each call has its own
per-operation bound instead: `lease_call_timeout` for `feed` and `update_locale`, the finalization
bound for the finalization sequence, `preconfiguration_timeout` for `open`. Expiry of
`lease_call_timeout` is `service_unavailable` (the server did not answer in time, so the client may
retry), cancels the call, which breaks the lease, and takes `abort` then `release`. The other
phases take their codes from the errors leaf's phase table and their cleanup from the state-aware
rules above (a lease already flushed is never aborted). Idle timing
resumes when the call returns.

Accepted finalization atomically disarms the idle, first-audio and duration bounds and
applies the finalization and cleanup bounds. Expiry of the first-audio, idle or duration bound
before accepted finalization selects abnormal termination (`abort`, `release`) and projects the
code in the table. While the plugin is draining, every wait is clipped to the time remaining to
the drain deadline, if that is smaller; the bound keeps its own absolute deadline and identity, and
a wait that ends because it reached the drain deadline expires as the drain deadline (the shutdown
row), not as its phase. The bounds are checked at startup against the engine deadlines
(vehicle leaf). Every send and close on a streaming surface, including the refusals that create no
owner, is bounded by `response_send_timeout`, clipped to the gRPC deadline and the drain deadline.

**Sends.** The drain announces its deadline to every send in flight, with or without an owner, so a
send already waiting re-clips its bound at once. A send whose bound has already passed when it is
due does not start, and takes the transport-failure path, unless it carries a terminal status (a
gRPC status or abort; a Realtime `error` event, or the `failed` event that ends the session, and its
close). A terminal status gets one attempt that adds no wait: it starts, runs to its first
suspension and is cancelled if it has not completed, so the shutdown and deadline rows of the errors
leaf reach a client that is still reading. A send that has not completed when its bound passes has
failed, even if it completes later. The handler cancels it and does not wait for it to settle: a
send still unsettled after one turn of the event loop stays under the stream's record until it
settles or `stop`, so cleanup starts at once and spends no second budget. An admission refusal or
a query refusal creates no record; its own request handler awaits the send inline under its
bound, and if the send is still unsettled one turn of the event loop after the bound passed, the
send moves, from that moment of abandonment, to the vehicle's registry as a send without an owner,
which takes no ingress-cap count. The vehicle's settlement closes that registry before its final
wait on unsettled sends; a refusal send abandoned after the close is cancelled and awaited by its
own handler instead, so no refusal send outlives the settlement as detached work (vehicle leaf,
Owner registry). The vehicle's settlement and `stop` await the sends under records and in the
registry. Once the plugin is
torn down, every send bound counts as passed. A send's failure, once established (its bound seen to
pass, or a terminal attempt found incomplete), is final: a cancellation of the handler while it
abandons the send changes nothing and no further send follows. A send that the plugin's own
cancellation of the owner interrupts first (the vehicle's settlement of a drain, or `stop`) is not a
transport failure: the owner takes the shutdown abort row, whose terminal status is sent under its
bound, or gets the one attempt once that bound has passed. A terminal status or close that this
cancellation itself interrupts ends the stream with no further send. The stream's record stays
registered until its handler, its cleanup continuation and every unsettled send have ended, so the
drain and `stop` reach a terminal send and await it. After a gRPC send fails, the handler ends
without writing a status; the HTTP/2 stream stays open until the client cancels, its deadline
passes, transport keepalive finds the peer dead or the server stops. The owner, its slot and its
lease are already released by then. When several deadlines apply, the earliest absolute deadline
wins (the errors leaf's precedence rule).

### Engine idle abort

If a lease fails with reason `expired` (the engine ended an idle stream), the surface projects
`idle_timeout`. This is the backstop. The frontend's bounds are below the engine's deadlines, but
the frontend cannot guarantee that they fire first: sub-unit traffic (accepted audio shorter than
one unit, including before the engine's never-fed deadline) does not reset the engine's clock, and
client-idle timing is suspended while a lease call is outstanding, so the engine can end a stream
first.

## Ownership

Every stream owner and every file-request owner that passed the ingress cap registers exactly one
owner task before reading its first message or request body, and keeps it tracked until its last
cleanup step (vehicle leaf): for a stream, `release` or the hand-over of the session to the factory,
in `stop` as in `drain`. The vehicle's settlement and `stop` await every owner and its continuation
and then run the factory settlement (`factory.drain(math.inf)`, which awaits every handed-over
cleanup task and its generation task without cancelling either, then `teardown_all()` and an await
of every task it returns; vehicle leaf), so a hand-over to a continuation or to the factory never
leaves engine-facing work running after `drain` or `stop` returns, and no generation task receives a
cancellation after its cancellation handler has begun. A stream owner serializes every lease call
and atomically selects exactly one terminal cause; normal finalization and abnormal termination
never both execute `finish` and `abort`. The frontend starts no detached task. The factory owns
every engine-facing task of its sessions, including the cleanup of a session whose owner is gone.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
| --- | --- | --- | --- |
| Coupling to the model plugin | Entry-point discovery, structural protocol, `getattr` on two attributes | Import the model plugin's classes; a base class in the frontend | The two packages release separately, and neither can import the other without a cycle or a lock-step version |
| Engine capability check | The factory constructor raises | A frontend check on the engine client | The engine-client protocol does not declare the stream-admission calls; only the party using them can test for them |
| Factory selection | Exactly one accepts, unless a name is configured | First accepting; priority order | Two factories for one model is a deployment error; ordering would hide it |
| Cadence | `560ms` for every stream; startup fails if not served | Client-chosen cadence; model default | Riva has no cadence field; one cadence for both streaming surfaces means one clip decodes on the same lookahead over either |
| Default locale | The factory's `default_locale`; client must name one when `None` | A frontend-chosen default; always `auto` | The locale table is the model's; a guessed default changes transcripts silently |
| Final transcript | Taken from `flush`'s string at one shared point | Per-surface accumulation of `feed` results | A transcript is final only after a normal close; one derivation point cannot diverge between surfaces |
| Final delivery | Only after `finish` and `release` succeed | After `flush` | A wire terminal that precedes resource cleanup lets a client reconnect into a slot that is still held |
| Cleanup after a flushed lease | `release` only | `abort` then `release` | A flushed lease rejects `abort`, and the stream has already ended at `flush`; `abort` would be a failure that hides the real outcome |
| Cleanup budget | One bound for the whole cleanup, `release` always reached | A bound per call | A per-call bound lets a slow `abort` multiply the wait; reaching `release` with no budget hands the session to the factory's tracked cleanup, which the vehicle's factory settlement awaits at shutdown |
| Cleanup-bound expiry | Hand to the factory; no escalation | Escalate | The factory is the owner of an orphaned session and reports real faults (vehicle leaf) |
| Cleanup in the drain settlement | `session_cleanup_timeout` from the start of cleanup; on expiry the same cancelled-`release` hand-over as outside shutdown; the vehicle's factory settlement then awaits the factory's cleanup with `drain(math.inf)` before `teardown_all()` | The time remaining to the drain deadline; the lease's `teardown()` on expiry | After the deadline the remaining time is zero, which would skip `abort` and `release`. `teardown()` cancels an already cancelled generation task again, which can interrupt the engine client's abort, and starts no factory task that `drain` would await; one expiry rule keeps the owner's sequence identical in and out of shutdown |
| Cleanup in `stop` | The shutdown abort row, as in the drain settlement | The lease's `teardown()` only, after closing the factory | Core does not guarantee that the engine has stopped when `stop` runs, so a `teardown()` after `abort` could interrupt the engine client's abort; `abort` and `release` end the session the same way in every path |
| Cancelled `open` | Left to the factory; owner calls nothing on a lease it never received | Owner aborts and releases a lease that arrives late | A late lease is the owner's only after `open` returns it; the factory's table already owns the session through the cancellation |
| First-audio and idle bounds | One timer: first-audio duration until the first accepted audio, then the idle duration | Idle bound from `open` only | A client that never sends audio should be freed quickly, below the engine's never-fed deadline |
| Idle timing during lease calls | Suspended; each call has its own bound, expiry `service_unavailable` | Keep the idle clock running; project call expiry as `idle_timeout` | Waiting on the engine is not client idleness; a slow engine call is the server's failure and the client may retry |
| Shutdown cause | Intent recorded when the trigger is observed; shutdown finalization provisional; later failures and deadlines take the shutdown abort row; existing deadlines clipped to the drain deadline | Keep each failure's own row; replace every deadline by the drain deadline | The session is being shut down, so the client's right move is to retry elsewhere (1012, `UNAVAILABLE`); the shorter deadlines still bound each call |
| Traffic after client-selected finalization | A bounded receive observer until the final's send begins | Read nothing after accepted finalization | A violation becomes observable and has one outcome; the final is irrevocable once its send begins |
| Sends | Every send and close bounded by `response_send_timeout` | Unbounded sends on streams | A stalled reader would otherwise hold an owner, its slot and its lease |
| Shutdown finalization | The stream's own owner runs it; vehicle supplies only trigger and deadline | The vehicle drives lease calls; cancel at the settle point | Lease calls are serialized and state-aware only inside the owner; a second driver could overlap calls or abort a flushed lease |
| Finalization tail | Feed the audio frontend's drained tail before `flush` | Flush only | The resampler's lookahead and byte residual belong to the audio; dropping them loses the last samples |
| Unresolved deferred header at finalization | Zero bytes: empty final; partial header: `unsupported_format` | Always error; always empty final | No audio is a valid session; a partial header is a malformed file |
| WebSocket disconnect during a call | Observed within that call's bound | Concurrent reader task | One reader keeps the command FIFO and backpressure simple; the bound makes the delay finite. gRPC has a callback |
| Idle clock | Application-level accepted audio | Socket bytes; keepalive | A client trickling bytes or pings holds a slot without sending audio |
| Zero-audio finalization | Normal flush path with an empty final | Reject | A valid degenerate session; both dialects have clients that do this |
| Deferred-RIFF scan | After `open`, bounded, no `feed` until resolved | Before `open` | The stream is admitted only when the configuration is complete; the scan needs the audio that arrives after it |

## Open Questions & Future Decisions

### Deferred
1. **Per-utterance finals.** The model plugin binds no endpoint policy and its lease reports no
   segment completions. A stream's one final at the end is the behaviour until both are designed
   together with the frontend; then multi-utterance reuse of a session is revisited.
2. Outside shutdown, a cleanup that expires while its `abort` is cancelled mid-flight leaves the
   factory to retire the session through the cancelled `release`; the factory's cleanup task has
   no deadline of its own, so a stuck engine call holds that session until shutdown, where the
   factory settlement awaits it and core's repeated-signal exit bounds the wait. This cleanup
   order relies on the model plugin's `SESS-ABORT-002`: a cancelled `abort` leaves the lease
   `aborted`, where `release` is allowed.

## References

- Model plugin session design and specs: `pdtgct/vllm-nemotron-asr-streaming-plugin`
  `docs/intent/session/` (`SESS-FAC-*`, `SESS-OPEN-*`, `SESS-LEASE-*`, `SESS-LOC-*`, `SESS-EXP-001`,
  `SESS-OWN-*`, `SESS-ERR-*`).
- RFC 1 client path: `docs/design/persistent_state.md` on `pdtgct/vllm` branch
  `persistent-state-primitive-v2`.
- High-level design: `docs/high-level-design.md`; vehicle leaf (bounds against engine deadlines,
  escalation); streaming leaf (wire projection); errors leaf (reason mapping).

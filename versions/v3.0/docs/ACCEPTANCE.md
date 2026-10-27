# V3 research preview acceptance contract

[2026-10-15 中文最终工程验收结果](D:/2_datas/Vm/versions/v3.0/docs/ACCEPTANCE_RESULT.md) records completed evidence and remaining limits. This contract remains the requirement reference.

This document defines evidence required for the independently runnable V3 preview. It is not a declaration that tests have passed. Runtime evidence is recorded separately with exact commands, environment, timestamps, exit codes, and output receipts.

## Scope and protected boundaries

- Use existing frozen B01/B00/V2 weights and existing audio only. No additional training, pseudo labels, downloads, label fitting, or new data intake is part of this delivery.
- Preserve `models/emotion_model.pth`, finished teacher/mixed weights, existing environments, and V1/V2 source and default startup behavior. Compare protected SHA256 values before and after acceptance.
- Do not inspect or evaluate the sealed 622-song final test labels. Existing 584-song comparison metrics are descriptive evidence, not a fresh final-test result.
- B01 is a research candidate. Its PMEmo arousal/noninferiority limitations remain visible. Engineering acceptance cannot change the model release verdict.
- Three-seed metric averaging and three-member equal-weight prediction ensembling are different results. The application must identify which it displays.

## Acceptance matrix

| ID | Requirement or risk | Observable behavior and independent oracle | Required evidence |
| --- | --- | --- | --- |
| M01 | Frozen identity | B01 members are exactly seeds 20261003/04/05; B00/V2 are explicitly identified; checkpoint, scaler, extractor and manifest hashes match | Manifest inspection plus a successful fresh loader run; altered input must be rejected |
| M02 | No silent fallback | Missing or incompatible weights produce a readable error, never training or another model labeled B01 | Negative loader/CLI contract checks |
| M03 | Ensemble semantics | Equal mean of three raw member predictions; sample disagreement is separate and explicitly not calibrated confidence | Small known numerical fixture plus existing-feature inference comparison |
| T01 | Native whole-song clock | Sequence times are exactly 0, 0.5, ... through floor(duration/0.5); no time stretching, shifting, dropped interior frames or duplicate times | Boundary fixtures plus real decoded audio receipt |
| T02 | Distinct comparison clocks | V2 retains its 0.1 s clock; display alignment does not replace source clocks or raw arrays | Payload inspection and playback boundary checks |
| T03 | Native duration | Audio duration comes from native decoded samples; project and playback/export timing do not rely solely on MP3 header duration | Decode receipt and media probe |
| D01 | Raw/display separation | Smoothing creates a separate display array; raw values and source hashes remain byte-for-byte unchanged | Known impulse fixture, independent expected result, before/after raw file hashes |
| D02 | Unavailable estimates | Calibration intervals and tension remain unavailable; seed disagreement is not confidence; absent values are not filled with plausible emotion scores | JSON and UI inspection |
| D03 | Acoustic units | RMS, dBFS and centroid Hz are labeled as acoustic measurements; visual control mappings are separate | Known RMS/centroid fixture and exported field inspection |
| C01 | Cache identity | Audio content, feature contract and model/manifest identity participate in reuse; same filename with changed bytes is not reused | Isolated tamper and same-name/different-content cases |
| C02 | Cache integrity | Partial cache, missing receipt, altered data, or changed receipt fails explicitly | Isolated fixture mutations |
| P01 | Project containment | Project IDs, media names and stored paths cannot resolve outside the configured project root, including URL-encoded traversal | Store and HTTP negative cases |
| P02 | Existing projects | V3 writes to its independent root and does not rewrite V2 projects | Static entrypoint review plus before/after fixtures |
| H01 | Local HTTP boundary | Loopback bind, approved Host, same-origin mutations and server token; wrong/missing token or foreign origin cannot mutate | Real loopback HTTP contract tests |
| H02 | Media Range | Audio Range returns exact requested source bytes, correct 206 length/range; invalid/unsatisfiable ranges return 416 | Known byte fixture through HTTP |
| H03 | Invalid requests | Invalid JSON, wrong content type, malformed IDs, oversized requests and unsupported actions are rejected without starting work | HTTP negative cases and unchanged job count |
| J01 | Real state | Saved running status alone never establishes a live worker; PID plus creation identity and actual lock ownership are checked | Controlled dead/stale identity tests and one real job |
| J02 | Concurrency | One heavy analysis owns the execution lease; duplicate analyze does not start competing GPU jobs | Queue contract and process identity evidence |
| J03 | Cancellation | Cancel targets only its verified worker; state reflects termination; incomplete results never become successful cache entries | Controlled worker fixture and one finite cancellation exercise |
| J04 | Recovery | Resume uses retained config, command and matching input hashes; failed jobs expose cause and changed inputs are rejected | Controlled failed/stale job scenarios and resume receipt |
| E01 | Numerical export | JSON/CSV retain model identity, source clock, raw/display distinction, unit fields and unavailable flags | Parse exported files and compare against source payload |
| E02 | Video timing | Exported video and audio have matching requested start/span within frame/sample tolerance; complete output is published only after successful encode | Media probe plus frame/seek observations |
| E03 | Export containment | Output names cannot escape project renders; a failed encode does not advertise a playable successful output | Negative path test and failed subprocess fixture |
| U01 | End-to-end workflow | Launch, existing-audio analysis, timeline playback, model selection, raw/display toggle, export and reopen work | Real browser smoke and project reopen |
| U02 | Status clarity | UI shows offline whole-file analysis, research status, job stages, failure reason, and model identity | Browser inspection |
| R01 | Speed/resource evidence | Cold/warm runs distinguish decode, feature extraction, inference and render time; process suspension is not counted as stable compute throughput | Benchmark receipt on named hardware; memory/VRAM samples |
| R02 | Scope-safe acceptance | Audio acceptance samples are existing non-final-test inputs selected by format/duration properties, not favorable model scores | Immutable sample inventory with provenance |
| R03 | Protection | Protected model SHA256 values and old default entrypoints are unchanged after delivery | Final checksum and targeted diff report |

## Evidence levels and verdict

Each matrix row is recorded as **Pass**, **Partial**, **Fail**, or **Inconclusive**. Source inspection alone is insufficient for a runtime pass. Synthetic contract fixtures must be identified as synthetic and kept in temporary test directories, never registered as human annotations, training data, or model quality evidence.

Engineering preview acceptance requires a complete launch-to-export/reopen exercise, no unresolved material correctness/security finding, valid model provenance, and a final protection check. Model promotion is a separate decision requiring the pre-existing per-dataset/per-axis gates. No engineering test authorizes opening the sealed final test set.

## Planned verification layers

1. Pure numerical/store contracts with deterministic small fixtures.
2. Real loopback HTTP contracts, using an isolated ephemeral port and fixture project root.
3. Worker identity, cancellation/recovery, cache invalidation and export contracts with controlled non-GPU workers where appropriate.
4. Existing-feature B01/B00 equivalence and a real whole-audio CPU/GPU inference run by the owning coordinator.
5. Browser workflow and actual encoded-media probe, including reopen after server restart.
6. Bounded existing-audio compatibility and resource report. Record actual coverage; do not label a three-file smoke as an 18–24-song sweep.

The coordinator runs tests and all inference/benchmark/server workloads serially as required by shared process/GPU resources. This document's author prepares independent test contracts and does not launch those workloads.

## Real-audio inventory policy

Prefer existing files under `assets/audio` with independently verified duration/format metadata. Add other already-authorized existing inputs only after checking that they are outside sealed final-test roles. Include short, long, mono, stereo, silence/low-level and supported format cases where existing material is available. Generated silence/corruption fixtures test engineering behavior only and are never evidence of emotion accuracy.

## Human observations

Human listening is recorded separately: playback alignment, obvious discontinuities, and usability. It is not a requirement to collect new continuous annotations for this engineering preview. No automated script can claim that listening took place or that a curve matches a listener's emotion judgment.

## Coordinator evidence still required for final sign-off

The following is a completion checklist, not a claim of completion. Contract-test source is available in `../tests`; the coordinator must attach fresh execution evidence after the final source changes. Historical array locations for model equivalence are in `RAW_EQUIVALENCE.md`.

| ID | Root-run evidence to attach before assigning final result |
| --- | --- |
| M01 | Frozen manifest verification and fresh real checkpoint/scaler loads; protected/member hashes |
| M02 | Current contract-suite result plus missing/incompatible model command failure |
| M03 | Current contract-suite result and historical raw-member/ensemble comparison on matching audio |
| T01 | Current clock contracts and all completed real-audio output clocks |
| T02 | Original V2 native output length/clock, plus actual playback at beginning/interior/end |
| T03 | Native EOF sample receipts and playback/export media probe |
| D01 | Smoothing contracts, browser raw/display switch and before/after raw result hashes |
| D02 | Semantic negative contracts and browser unavailable/calibration/disagreement labels |
| D03 | Acoustic numerical contracts, real-result physical units and exported field inspection |
| C01 | Historical feature cache contracts and a real warm cache hit with matching identity |
| C02 | Tamper/partial-cache contract results and explicit failure logs |
| P01 | Store/HTTP traversal contracts including forged export-directory rejection |
| P02 | Final targeted diff/entrypoint inspection and V2 project protection evidence |
| H01 | Real loopback HTTP contract result and actual server bind/Host/Origin/token behavior |
| H02 | Real HTTP exact Range and HEAD results plus browser audio playback |
| H03 | Invalid body/content type/size/export setting contract results and no job creation |
| J01 | PID creation/reuse/early-exit contract results and actual job identity/status receipt |
| J02 | Duplicate submission contract result and one real process/lease inspection |
| J03 | Controlled real cancel result, verified worker exit and partial-output visibility |
| J04 | Fresh resume evidence with unchanged retained config; failed original attempts preserved |
| E01 | CSV/JSON contract results and real downloaded artifact parsing/provenance |
| E02 | Real video frame count, stream timestamps/durations, decode success and observed playback |
| E03 | Failed encoder/path contract results and no published completed output after failure |
| U01 | Browser launch-to-analysis/export/reopen steps with actual resulting project identity |
| U02 | Browser screenshots/state checks for research status, errors and model identity |
| R01 | Named hardware timing/resource report, cold versus warm distinctions and suspended-time caveat |
| R02 | Frozen 24-song engineering inventory and actual per-song completion/coverage count |
| R03 | Final original/teacher/mixed hash comparison and old source/entrypoint diff |

Windows publication retry also requires the two `test_store_contract.py` regressions: two transient sharing failures followed by exact-byte publication, and eight persistent failures that propagate while retaining the previous destination and pending temporary content. Preserve the failed engineering attempt and its original code binding when recording a revised verification run.

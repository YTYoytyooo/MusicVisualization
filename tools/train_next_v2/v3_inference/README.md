# V3 research preview inference

This independent package uses existing B01 and B00 checkpoints. It does not
train, fit calibration, import new datasets, read dynamic labels, or evaluate
the sealed 622-song final test. Original model/app files are read-only.

## Public interface

- `prepare_manifest(path=MANIFEST) -> Path`: verify audited checkpoint hashes,
  embedded provenance/scalers, original protected models, frozen extractor and
  dependency versions; create `models/next_va_v2/v3-preview-v1/manifest.json`
  once. Existing manifests must verify and are never silently regenerated.
- `build_inventory(output=LIBRARY, max_songs=24) -> dict`: create
  `data/v3.0/library.json` from the existing 18 illustrative validation songs and
  existing assets chosen by file-format and metadata-duration coverage. Final
  test audio hashes and recording groups are excluded using identity metadata.
  No labels or candidate model scores are loaded for selection. These examples
  support engineering checks, not new generalization claims.
- `analyze(audio_path, project_dir, device='cuda', progress=None,
  cancelled=None, include_v2=False, manifest_path=MANIFEST) -> Path`: save a
  fixed-weight offline whole-file raw result. The outer worker owns GPU/process
  and project locks. `progress(dict)` reports stage and optional counters;
  `cancelled()` is checked between extraction batches, heads and writes.
- `verify_result(project_dir, manifest_path=MANIFEST,
  expected_audio_sha256=None, include_v2=None) -> dict`: read-only hash,
  source/project, timeline, seed-aggregation and acoustic-array validation.
- `validate_output_system(system, duration, legacy=False)`: pure validation seam
  for native clocks, finite bounded outputs and three-member aggregation.

The CLI offers `prepare`, `inventory`, `analyze --audio ... --project ...
--device cpu|cuda [--include-v2]`, and `verify --project ...`. Use the independent
Python at `data/training/next_va/env/Scripts/python.exe` and the outer worker's
registered Vm-local offline/cache environment.

## Output and recovery contract

`project.json` uses schema version 3 with `id`, native `audio` geometry,
`manifest_sha256` and `created_at`. Native geometry comes from full `sf.read`
through EOF, not MP3 header length. `analysis/result.json` has the same audio
and manifest identity; `systems.b01` and `systems.b00` contain 0.5-second native
clocks, raw V/A, all three member arrays and their sample standard deviation.
The fixed arithmetic mean is a preview output; it is distinct from the mean
of three independently evaluated metric values.

Optional `systems.v2` uses unchanged original CPU feature and head methods on
the original 0.1-second clock. The previously rejected GPU V2 port is never used.
V2 comparisons can take considerably longer than cached sequence-head work.

`analysis/receipt.json` is the last commit write. It pins the result and project
SHA256, audio identity, model member identities, feature cache hash and manifest
SHA256. Repeated analysis verifies and reuses a complete result. Changed options
require a new project. Orphan/unreceipted outputs fail closed and remain for
inspection. Committed feature caches survive cancellation. The raised
`AnalysisCancelled` derives from `InterruptedError`.

Existing features are accepted only with a matching hash from the already frozen
comparison/B protocol. Newly extracted frozen features live under
`data/v3.0/cache/sequence`, and explicit V2 CPU results under
`data/v3.0/cache/legacy_cpu`. Audio, features and model versions are bound to
their receipts. A same-named file is not sufficient for a cache hit.

## Units and limitations

- V/A are psychological prediction axes in `[-1, 1]`.
- RMS amplitude/dBFS and spectral centroid Hz are local physical acoustic
  features derived from the unchanged feature extractor. They are not perceived
  loudness LUFS or emotional arousal/brightness scores.
- Seed disagreement is not calibrated confidence. Calibration remains
  `unavailable`; the failed B00 calibrator is not transferred to B01.
- Tension is `unavailable`; display smoothing is not applied to raw arrays.
- This is whole-song bidirectional inference, not a streaming/latency claim.
- The failed internal per-dataset/per-axis gate remains binding. A functioning
  preview does not promote B01 or justify replacing the existing model.
- Timing uses monotonic elapsed seconds, includes OS suspension if it happens,
  and separates decode, feature cache use, model loading/inference and V2 CPU.
  Root integration must independently record resource/acceptance measurements.

No tests or inference were executed by the implementation subagent; the root
worker performs fresh integrated checks before freezing/delivering artifacts.

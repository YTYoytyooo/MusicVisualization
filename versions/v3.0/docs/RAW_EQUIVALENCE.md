# Historical raw-output equivalence references

These files support engineering equivalence checks only. Do not calculate new accuracy metrics, inspect label arrays, fit parameters, or change the sealed final-test policy.

## B01: 18 existing representative songs

Base directory:

`D:\2_datas\Vm\data\training\next_va\optimization\comparison_jobs\f-validation-v2-mixed-b01-cpu-v1`

Protocol:

`D:\2_datas\Vm\data\training\next_va\optimization\protocols\f-v2-mixed-b01-validation-cpu-v2.json`

Each representative maps to `predictions/<key>.npz` and adjacent `predictions/<key>.json`. The key is `f'{index:04d}-{row["audio_sha256"][:16]}'`, where index is its unchanged index in `protocol.validation_songs`. Match the full `audio_sha256` to the V3 result, not just a filename or prefix.

| song_id | key | Native sequence frames |
| --- | --- | ---: |
| deam:1765 | 0118-1635e15c28f46822 | 91 |
| deam:1933 | 0146-3c1adefa25c3d501 | 91 |
| deam:1949 | 0151-16415623b45c651c | 91 |
| deam:2014 | 0165-e96dc16bd1925f17 | 803 |
| deam:216 | 0177-4e40abbcfb8e6307 | 91 |
| deam:743 | 0245-8410feb4a122486c | 91 |
| memo2496:1174 | 0289-035d06dd7b429b53 | 319 |
| memo2496:2087 | 0390-8c7329d6879a1731 | 393 |
| memo2496:2248 | 0411-c8f0e9755b70d225 | 320 |
| memo2496:744 | 0487-73b1142254414bad | 379 |
| memo2496:796 | 0492-57788b8549a9dbae | 591 |
| memo2496:976 | 0513-aa228652dacfeed5 | 457 |
| pmemo:221 | 0526-30e26276a574cfa7 | 67 |
| pmemo:306 | 0527-e28fabc97a26211e | 125 |
| pmemo:421 | 0533-6cbab34a66d296bb | 115 |
| pmemo:594 | 0554-4e83e95f9ef11545 | 129 |
| pmemo:614 | 0556-22cc95f60fa8e45b | 57 |
| pmemo:855 | 0574-43bf83a799fd7753 | 159 |

All 18 adjacent receipt bindings and NPZ file SHA256 values were read-only rechecked during this verification preparation. Each selected `times` array equals `arange(N) * 0.5`. `predictions` is float64 with shape `(6, N, 2)`:

- `predictions[2:5]`: B01 seed 20261003, 20261004, 20261005 raw outputs.
- `predictions[5]`: their existing equal-weight prediction ensemble.
- `predictions[1]`: the old mixed model, **not B00**.
- `predictions[0]`: V2 joined onto the sequence clock; for original V2 native output use `v2_times` and `v2_raw_five[:, :2]` instead.

The NPZ also contains labels and masks. **Do not use `ArtifactStore.read()` for this task**, because that helper materializes every NPZ member, including labels. Open the NPZ lazily and select only `times` and `predictions` (or the explicitly needed V2 raw/time fields). Do not access `y`, `common_mask`, `original_mask`, `v2_indices`, or scoring fields.

Receipt checks before comparing:

1. Adjacent receipt `song_id` equals the protocol row identity.
2. Receipt `file` equals the NPZ basename; SHA256 of NPZ equals receipt `sha256`; no adjacent `.tmp` exists.
3. Receipt `contract_sha256` equals SHA256 of the job's unchanged `launch.json`: `68ff3b227462318ac4a026c3a4318ab32d90d8670f7a6c50760c16b8cb9c45d5`.
4. Protocol SHA256 is `eb6e5dd3bd4bbae2a83f7081ae583bf803d249d234ae7cb6c3e6ae6bf1cbb5b1`.
5. Bind the full audio SHA256, sequence feature key/cache hash and B01 member checkpoint identities to the current manifest/result.

The old job's `verification.json` records 584 verified song receipts, five fresh checkpoint loads, unchanged protected models and sequence CPU/GPU maximum difference `3.5762786865234375e-07`. This is historical verification evidence, not a pass for current V3 output.

Compare V3 `systems.b01.times_seconds` with saved `times` exactly. Compare `member_raw_va` with `predictions[2:5]`, `raw_va` with `predictions[5]`, and disagreement with the sample standard deviation across the three old members. The pre-existing independent adapter tolerance is `rtol=1e-5, atol=1e-6`; do not widen it after observing a mismatch. Record the actual maximum difference and whether all arrays passed, without labels or model-quality metrics.

## B00: three fully native audio benchmarks

Base directory:

`D:\2_datas\Vm\data\training\next_va\optimization\inference_jobs\f-research-reference-benchmark-v1`

Each row below has `predictions/<full audio SHA256>.json` and `receipts/<full audio SHA256>.json`. All three are present in the frozen current 24-song V3 library. Their receipt `files` hashes were read-only rechecked during this preparation.

| Existing audio | Full audio SHA256 / filename stem | Native duration | Frames |
| --- | --- | ---: | ---: |
| assets/audio/test/happy_adveture.mp3 | 50aa6a32625fe8565403a0c8ceb772d7fb500b9267d0019eacfe0fc742b6c73d | 46.77331065759637 | 94 |
| assets/audio/demo/myfavouriteband有形ランペイジ/有形ランペイジ - フューチャー・イヴ.flac | 029836ea7ecfe8c0bdf2f32b6935ff78db1dd209c9f4038df5d03f912322e19c | 219.0 | 439 |
| assets/audio/demo/sasakure.UK - ナナクジャクの彼方.flac | 09fe0b88b335712e20fe74d19badf5bdb2256d2c7553c40e1f5d8291140f32e0 | 318.0 | 637 |

The prediction JSON has no label arrays. Compare `times_seconds`, `member_raw_va` `(3,N,2)`, `raw_va`, and `seed_disagreement_std` with current `systems.b00`. Its `source.audio_sha256` and receipt `audio_sha256` must match current audio. `acoustic` can additionally check exported feature-derived units.

Receipt provenance:

- `manifest_sha256`: `94c1a66b60aef475c01e4e8dd3b7e9b5871321f0018be4e00d59166428327edf` (old B00 manifest, deliberately different from the V3 wrapper manifest).
- `protocol_sha256`: `70b9a4c40549a951f49296472a3ead8c1bbb40497bd7ce14beeb8ee519a02046`.
- Every item in receipt `files` must hash correctly; includes the prediction JSON, feature NPZ and feature receipt.
- Old `verification.json` records native half-second clock equality, independent training-collator equality and maximum CPU/GPU prediction difference `3.5762786865234375e-07`.

Compare member identities/scaler/features across the two manifests, rather than requiring their wrapper manifest digests to be equal. No B00 arrays exist in the six-system B01/mixed/V2 comparison bundle; use these three native benchmark records.

# Independent V3 contracts

Run from `D:\2_datas\Vm` with the existing independent GPU Python, even though these contracts do not perform model inference:

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.\data\training\next_va\env\Scripts\python.exe' -m unittest discover -s versions/v3.0/tests -v
```

The suite uses synthetic deterministic engineering fixtures in `data/v3.0/test-scratch`. It does not load model weights, fit labels, read final-test labels, launch a GPU worker, or invoke a real encoder. HTTP tests start a temporary loopback server on an ephemeral port and close it before returning. Process-start and encoder-start failure paths are patched at the process boundary; their results establish error handling only.

Coverage:

- `test_store_contract.py`: project/audio identity, raw and commit integrity, partial output, traversal.
- `test_display_contract.py`: known smoothing values, independent missing axes and gaps, immutable raw values, separate native clocks in CSV.
- `test_jobs_contract.py`: persisted reproducibility inputs, PID creation identity, stale/early-dead workers, resume and process-start failure.
- `test_http_contract.py`: real HTTP Host/Origin/token checks, body validation, audio Range/HEAD, download containment and export prevalidation.
- `test_export_contract.py`: invalid ranges, changed source audio and unsuccessful encoder creation cannot publish a completed export.
- `test_inference_contract.py`: native clock and three-seed aggregation contracts, committed output identity and unavailable-field semantics.
- `test_feature_cache_contract.py`: real small NPZ reads in the frozen historical-cache branch, container hash, extractor key, dtype, shape and source-clock rejection. No new features are extracted.

Passing this suite does not establish real model equivalence, cold-feature compatibility, memory/VRAM performance, an actual playable video, browser usability, or human listening. Those remain separate rows in `../docs/ACCEPTANCE.md`, with root-run evidence required.

@echo off
setlocal
cd /d "D:\2_datas\Vm"
set "PYTHONUTF8=1"
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=8"
set "MKL_NUM_THREADS=8"
set "TEMP=D:\2_datas\Vm\data\training\next_va\optimization\tmp"
set "TMP=D:\2_datas\Vm\data\training\next_va\optimization\tmp"
set "HF_HUB_OFFLINE=1"
set "HF_HOME=D:\2_datas\Vm\data\training\next_va\optimization\cache\huggingface"
set "TORCH_HOME=D:\2_datas\Vm\data\training\next_va\optimization\cache\torch"
set "NUMBA_CACHE_DIR=D:\2_datas\Vm\data\training\next_va\optimization\cache\numba"
set "CUDA_CACHE_PATH=D:\2_datas\Vm\data\training\next_va\optimization\cache\cuda"
set "TRANSFORMERS_OFFLINE=1"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
set "PYTHONDONTWRITEBYTECODE=1"
"D:/2_datas/Vm/data/training/next_va/env/Scripts/python.exe" -u -m tools.train_next_v2.review_handoff.server --config "D:\2_datas\Vm\data\training\next_va\optimization\annotation_pilot\e-pilot58-review-handoff-v1\launch.json"
if errorlevel 1 pause

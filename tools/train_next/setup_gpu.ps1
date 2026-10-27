param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$trainingEnvironment = Join-Path $workspace 'data/training/next_va/env'
$trainingPython = Join-Path $trainingEnvironment 'Scripts/python.exe'
$existingPython = Join-Path $workspace 'venv/Scripts/python.exe'
if ((Test-Path -LiteralPath $trainingEnvironment) -and -not $Resume) { throw 'Training environment exists. Use -Resume to continue setup.' }
if (-not (Test-Path -LiteralPath $trainingPython)) {
    & $existingPython -m venv $trainingEnvironment
    if ($LASTEXITCODE -ne 0) { throw 'Unable to create isolated training environment' }
}
$pipCache = Join-Path $workspace 'data/training/next_va/pip-cache'
& $trainingPython -m pip install --cache-dir $pipCache 'torch==2.11.0' --index-url 'https://download.pytorch.org/whl/cu128'
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
& $trainingPython -m pip install --cache-dir $pipCache -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed' }
& $trainingPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Training dependency check failed' }
& $trainingPython -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; x=torch.randn(128,128,device='cuda'); print(torch.__version__, torch.cuda.get_device_name(0), float((x @ x.T).mean()))"
if ($LASTEXITCODE -ne 0) { throw 'GPU execution check failed' }

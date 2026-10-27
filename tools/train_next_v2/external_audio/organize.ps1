param([Parameter(Mandatory=$true)][string]$Launch, [switch]$Resume)
$ErrorActionPreference = 'Stop'
$taskConfig = Get-Content -LiteralPath $Launch -Raw | ConvertFrom-Json
$taskWorkspace = [IO.Path]::GetFullPath($taskConfig.cwd).TrimEnd('\')
if ([IO.Path]::GetFullPath((Get-Location).Path).TrimEnd('\') -ne $taskWorkspace) { throw 'Wrong working directory' }
function Assert-Contained([string]$Path) {
    $resolved = [IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($taskWorkspace + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Path outside Vm: $resolved" }
    $ancestor = $resolved
    while ($ancestor -and $ancestor.StartsWith($taskWorkspace, [StringComparison]::OrdinalIgnoreCase)) {
        if (Test-Path -LiteralPath $ancestor) {
            if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse-point ancestor requires inspection: $ancestor" }
        }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
    return $resolved
}
function Write-TaskJson([string]$Path, $Value) {
    $target = Assert-Contained $Path
    $temporary = $target + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
    [IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 30), [Text.UTF8Encoding]::new($false))
    [IO.File]::Move($temporary, $target, $true)
}
function Assert-Tree($Root, $Files, $Directories) {
    $resolved = Assert-Contained $Root
    $items = @(Get-Item -LiteralPath $resolved) + @(Get-ChildItem -LiteralPath $resolved -Recurse -Force)
    if (@($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Reparse points require separate inspection' }
    $actualFiles = @($items | Where-Object { -not $_.PSIsContainer })
    if ($actualFiles.Count -ne $Files.Count) { throw "File count changed in $resolved" }
    foreach ($entry in $Files) {
        $path = Assert-Contained (Join-Path $resolved $entry.relative_path)
        if (-not $path.StartsWith($resolved + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'File escaped registered collection' }
        $file = Get-Item -LiteralPath $path
        if ($file.Length -ne $entry.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "File changed: $path" }
    }
    $actualDirectories = @($items | Where-Object { $_.PSIsContainer -and $_.FullName -ne $resolved } | ForEach-Object { [IO.Path]::GetRelativePath($resolved, $_.FullName).Replace('\','/') } | Sort-Object)
    if (($actualDirectories -join "`n") -ne (@($Directories | Sort-Object) -join "`n")) { throw "Directory topology changed in $resolved" }
}
$taskDirectory = Assert-Contained $taskConfig.output
$taskLock = Assert-Contained $taskConfig.os_lock
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($taskLock)) | Out-Null
$taskLease = [IO.File]::Open($taskLock, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::ReadWrite)
$taskHeld = $false
try {
    if ($taskLease.Length -eq 0) { $taskLease.WriteByte(0); $taskLease.Flush() }
    $taskLease.Lock(0,1); $taskHeld = $true
    if (Test-Path -LiteralPath (Join-Path $taskDirectory 'report.json')) { throw 'Completed organization must not rerun' }
    if ((Test-Path -LiteralPath (Join-Path $taskDirectory 'process.json')) -and -not $Resume) { throw 'Existing attempt needs diagnosis and -Resume' }
    $taskProcess = [Diagnostics.Process]::GetCurrentProcess()
    Write-TaskJson (Join-Path $taskDirectory 'process.json') @{ started_at=[DateTime]::UtcNow.ToString('o'); process_identity=@{pid=$taskProcess.Id;creation_filetime=$taskProcess.StartTime.ToUniversalTime().ToFileTimeUtc().ToString()}; resumed=[bool]$Resume }
    foreach ($pin in $taskConfig.code_and_data_sha256.PSObject.Properties) {
        $path = Assert-Contained (Join-Path $taskWorkspace $pin.Name)
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pin.Value) { throw "Frozen input changed: $path" }
    }
    # Validate every resolved absolute source and destination before any recursive move.
    foreach ($collection in $taskConfig.collections) {
        $source = Assert-Contained $collection.source
        $destination = Assert-Contained $collection.destination
        if ([IO.Path]::GetPathRoot($source) -ne [IO.Path]::GetPathRoot($destination)) { throw 'Cross-volume moves are not permitted' }
        $hasSource = Test-Path -LiteralPath $source
        $hasDestination = Test-Path -LiteralPath $destination
        if ($hasSource -and -not $hasDestination) { Assert-Tree $source $collection.files $collection.directories }
        elseif ($Resume -and -not $hasSource -and $hasDestination) { Assert-Tree $destination $collection.files $collection.directories }
        else { throw "Source/destination collision or absence: $source -> $destination" }
    }
    $taskMoves = @()
    foreach ($collection in $taskConfig.collections) {
        $source = Assert-Contained $collection.source
        $destination = Assert-Contained $collection.destination
        if (Test-Path -LiteralPath $source) {
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination)) | Out-Null
            Move-Item -LiteralPath $source -Destination $destination
        }
        Assert-Tree $destination $collection.files $collection.directories
        if (Test-Path -LiteralPath $source) { throw "Source unexpectedly remains: $source" }
        $taskMoves += @{source=$source;destination=$destination;files=$collection.files.Count;hashes_unchanged=$true}
        $taskMoves[-1] | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $taskDirectory 'console.log') -Encoding utf8
    }
    $taskReport = @{job_id=$taskConfig.job_id;status='completed_files_verified';completed_at=[DateTime]::UtcNow.ToString('o');moves=$taskMoves;source_file_count=$taskConfig.file_count;all_file_hashes_unchanged=$true;filenames_unchanged=$true;files_deleted=0;audio_reencoded=$false}
    Write-TaskJson (Join-Path $taskDirectory 'report.json') $taskReport
    $taskReport | ConvertTo-Json -Depth 8
} catch {
    Write-TaskJson (Join-Path $taskDirectory 'failure.json') @{at=[DateTime]::UtcNow.ToString('o');error=$_.Exception.Message}
    throw
} finally {
    if ($taskHeld) { $taskLease.Unlock(0,1) }
    $taskLease.Dispose()
}

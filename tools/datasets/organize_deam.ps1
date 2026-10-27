$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$destination = [IO.Path]::GetFullPath((Join-Path $workspace 'data/datasets/deam/raw'))
if (-not $destination.StartsWith($workspace + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Destination is outside workspace' }
$mapping = @{'annotations'='annotations'; 'MEMD_audio'='audio'; 'features'='features'}
$plan = @()
foreach ($name in $mapping.Keys) {
    $source = [IO.Path]::GetFullPath((Join-Path $workspace $name))
    $target = [IO.Path]::GetFullPath((Join-Path $destination $mapping[$name]))
    if (-not $source.StartsWith($workspace + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or -not $target.StartsWith($destination + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe move path' }
    if (-not (Test-Path -LiteralPath $source -PathType Container)) { throw "Missing source: $source" }
    if (Test-Path -LiteralPath $target) { throw "Destination exists; refusing to merge: $target" }
    $entries = @(Get-ChildItem -LiteralPath $source -Recurse -Force)
    if ((Get-Item -LiteralPath $source).Attributes -band [IO.FileAttributes]::ReparsePoint -or ($entries | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })) { throw 'Reparse points require manual inspection' }
    $files = @($entries | Where-Object { -not $_.PSIsContainer } | ForEach-Object { @{path=$_.FullName.Substring($source.Length+1); bytes=$_.Length; modified_utc=$_.LastWriteTimeUtc.ToString('o')} })
    $plan += @{source=$source; target=$target; files=$files}
}
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$report = Join-Path $workspace 'data/datasets/deam/layout-migration.json'
@{planned_at=(Get-Date).ToUniversalTime().ToString('o'); moves=$plan} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $report -Encoding utf8
foreach ($item in $plan) {
    Move-Item -LiteralPath $item.source -Destination $item.target
    foreach ($file in $item.files) {
        $actual = Get-Item -LiteralPath (Join-Path $item.target $file.path)
        if ($actual.Length -ne $file.bytes -or $actual.LastWriteTimeUtc.ToString('o') -ne $file.modified_utc) { throw "Migration check failed: $($actual.FullName)" }
    }
    Write-Output "$($item.source) -> $($item.target): $($item.files.Count) files verified"
}

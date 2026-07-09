# Capacity sweep driver.
#
# For each target sample count: tile the prepared HIGGS training CSV up to that
# many rows (row i = HIGGS row i % file_rows) with tile_higgs.exe, run the
# OpenNN binary and the PyTorch script against it (each in its own process so an
# OOM kills only that run), record RESULT and peak RAM, then delete the file
# before moving on so only one large CSV is on disk at a time.
#
# The prepared HIGGS file (28 features + 1 label per row) comes from
# ../../throughput/higgs/prepare_higgs.py and lives under
# $OPENNN_BENCH_DATA/higgs (see ../DATA_POLICY.md).
#
#   usage:  powershell -File run_sweep.ps1 -Samples 5000000,10000000,...

param(
    [string]$HiggsCsv = "$env:OPENNN_BENCH_DATA\higgs\higgs_train.csv",
    [long[]]$Samples = @(5000000, 10000000, 20000000, 30000000, 40000000, 60000000, 80000000, 100000000, 140000000, 180000000),
    [string]$WorkDir = "$PSScriptRoot\_data",
    [string]$ResultsCsv = "$PSScriptRoot\results.csv"
)

$ErrorActionPreference = "Stop"
$tiler = "$PSScriptRoot\tile_higgs.exe"
$opennn = "$PSScriptRoot\..\..\..\build\bin\Release\opennn_capacity.exe"

if (-not (Test-Path $HiggsCsv)) {
    throw "HIGGS training CSV not found at '$HiggsCsv'. Set OPENNN_BENCH_DATA and run ../../throughput/higgs/prepare_higgs.py first (see ../DATA_POLICY.md)."
}

# HIGGS is 28 features + 1 label; values-per-row is fixed at 29.
$valuesPerRow = 29

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

function Parse-Result($lines) {
    $result = "MISSING"; $peak = ""
    foreach ($l in $lines) {
        if ($l -match '^RESULT=(.+)$') { $result = $Matches[1].Trim() }
        if ($l -match '^peak_mb=([-0-9.]+)') { $peak = $Matches[1] }
    }
    return @{ result = $result; peak = $peak }
}

if (-not (Test-Path $ResultsCsv)) {
    "samples,values_billion,csv_gb,engine,result,peak_mb" | Out-File -FilePath $ResultsCsv -Encoding utf8
}

foreach ($m in $Samples) {
    $csv = "$WorkDir\higgs_tiled_${m}s.csv"
    $valuesB = [math]::Round(($m * $valuesPerRow) / 1e9, 3)

    Write-Host "=== samples=$m  (~$valuesB billion values) ===" -ForegroundColor Cyan

    Write-Host "  tiling HIGGS..."
    & $tiler $HiggsCsv $m $csv
    if ($LASTEXITCODE -ne 0) { Write-Host "  tiler failed; stopping"; break }
    $csvGb = [math]::Round((Get-Item $csv).Length / 1GB, 2)
    Write-Host "  csv = $csvGb GB"

    # --- OpenNN ---
    Write-Host "  running OpenNN..."
    $o = & $opennn $csv 2>&1
    $op = Parse-Result $o
    Write-Host "    OpenNN: $($op.result)  peak=$($op.peak) MB" -ForegroundColor Yellow
    "$m,$valuesB,$csvGb,opennn,$($op.result),$($op.peak)" | Out-File -FilePath $ResultsCsv -Append -Encoding utf8

    # --- PyTorch ---
    Write-Host "  running PyTorch (pandas)..."
    $p = & python "$PSScriptRoot\pytorch_capacity.py" $csv 2>&1
    $pp = Parse-Result $p
    Write-Host "    PyTorch: $($pp.result)  peak=$($pp.peak) MB" -ForegroundColor Yellow
    "$m,$valuesB,$csvGb,pytorch,$($pp.result),$($pp.peak)" | Out-File -FilePath $ResultsCsv -Append -Encoding utf8

    Remove-Item $csv -Force
    Write-Host "  deleted csv"

    # Stop early once BOTH have failed — no point tiling bigger files.
    if ($op.result -ne "OK" -and $pp.result -ne "OK") {
        Write-Host "Both engines failed at samples=$m; stopping sweep." -ForegroundColor Red
        break
    }
}

Write-Host "`nResults written to $ResultsCsv" -ForegroundColor Green
Get-Content $ResultsCsv

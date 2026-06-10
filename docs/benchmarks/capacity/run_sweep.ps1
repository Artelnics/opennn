# Capacity sweep driver.
#
# For each sample count: generate a Rosenbrock CSV, run the OpenNN binary and
# the PyTorch script against it (each in its own process so an OOM kills only
# that run), record RESULT and peak RAM, then delete the file before moving on
# so only one large CSV is on disk at a time.
#
#   usage:  powershell -File run_sweep.ps1 -Variables 100 -Samples 2000000,5000000,...

param(
    [int]$Variables = 100,
    [long[]]$Samples = @(1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 8000000, 10000000, 12000000, 14000000),
    [string]$WorkDir = "$PSScriptRoot\_data",
    [string]$ResultsCsv = "$PSScriptRoot\results.csv"
)

$ErrorActionPreference = "Stop"
$gen = "$PSScriptRoot\generate_rosenbrock.exe"
$opennn = "$PSScriptRoot\..\..\..\build\bin\Release\opennn_capacity.exe"

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
    "samples,variables,values_billion,csv_gb,engine,result,peak_mb" | Out-File -FilePath $ResultsCsv -Encoding utf8
}

foreach ($m in $Samples) {
    $csv = "$WorkDir\rosenbrock_${Variables}v_${m}s.csv"
    $valuesB = [math]::Round(($m * ($Variables + 1)) / 1e9, 3)

    Write-Host "=== samples=$m  variables=$Variables  (~$valuesB billion values) ===" -ForegroundColor Cyan

    Write-Host "  generating..."
    & $gen $Variables $m $csv 1234
    if ($LASTEXITCODE -ne 0) { Write-Host "  generator failed; stopping"; break }
    $csvGb = [math]::Round((Get-Item $csv).Length / 1GB, 2)
    Write-Host "  csv = $csvGb GB"

    # --- OpenNN ---
    Write-Host "  running OpenNN..."
    $o = & $opennn $csv $Variables 2>&1
    $op = Parse-Result $o
    Write-Host "    OpenNN: $($op.result)  peak=$($op.peak) MB" -ForegroundColor Yellow
    "$m,$Variables,$valuesB,$csvGb,opennn,$($op.result),$($op.peak)" | Out-File -FilePath $ResultsCsv -Append -Encoding utf8

    # --- PyTorch ---
    Write-Host "  running PyTorch (pandas)..."
    $p = & python "$PSScriptRoot\pytorch_capacity.py" $csv $Variables 2>&1
    $pp = Parse-Result $p
    Write-Host "    PyTorch: $($pp.result)  peak=$($pp.peak) MB" -ForegroundColor Yellow
    "$m,$Variables,$valuesB,$csvGb,pytorch,$($pp.result),$($pp.peak)" | Out-File -FilePath $ResultsCsv -Append -Encoding utf8

    Remove-Item $csv -Force
    Write-Host "  deleted csv"

    # Stop early once BOTH have failed — no point generating bigger files.
    if ($op.result -ne "OK" -and $pp.result -ne "OK") {
        Write-Host "Both engines failed at samples=$m; stopping sweep." -ForegroundColor Red
        break
    }
}

Write-Host "`nResults written to $ResultsCsv" -ForegroundColor Green
Get-Content $ResultsCsv
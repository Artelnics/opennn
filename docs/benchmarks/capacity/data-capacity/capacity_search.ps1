# Capacity search: under a fixed committed-memory cap (Job Object), find the
# largest HIGGS dataset each engine can load + train before OOM.
#
# For each target sample count we tile the prepared HIGGS training CSV
# (28 features + 1 label per row) up to that many rows with tile_higgs.exe —
# row i is HIGGS row (i % file_rows), so every value is a real HIGGS row and the
# sample count can exceed the source file. Both engines run inside
# run_capped.exe with the SAME cap, so the crash point is determined by each
# engine's memory efficiency, not by background apps.
#
# The prepared HIGGS file comes from ../../throughput/higgs/prepare_higgs.py and
# lives under $OPENNN_BENCH_DATA/higgs (see ../DATA_POLICY.md).
#
#   usage:  powershell -File capacity_search.ps1 -CapGB 8

param(
    [double]$CapGB = 8,
    [string]$HiggsCsv = "$env:OPENNN_BENCH_DATA\higgs\higgs_train.csv",
    [long]$StartSamples = 5000000,
    [long]$StepSamples = 5000000,
    [long]$MaxSamples = 200000000,
    [string]$WorkDir = "$PSScriptRoot\_data",
    [string]$ResultsCsv = "$PSScriptRoot\capacity_results.csv"
)

$ErrorActionPreference = "Stop"
$tiler   = "$PSScriptRoot\tile_higgs.exe"
$capper  = "$PSScriptRoot\run_capped.exe"
$opennn  = "$PSScriptRoot\..\..\..\build\bin\Release\opennn_capacity.exe"
$cap     = [long]($CapGB * 1GB)

if (-not (Test-Path $HiggsCsv)) {
    throw "HIGGS training CSV not found at '$HiggsCsv'. Set OPENNN_BENCH_DATA and run ../../throughput/higgs/prepare_higgs.py first (see ../DATA_POLICY.md)."
}

# HIGGS is 28 features + 1 label; values-per-row is fixed at 29.
$valuesPerRow = 29

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
"engine,cap_gb,samples,values_billion,result" | Out-File -FilePath $ResultsCsv -Encoding utf8

function Test-Engine($engine, $csvFull) {
    # Returns "OK" or "OOM"/"FAIL". Runs the engine under the memory cap.
    if ($engine -eq "opennn") {
        $out = & $capper $cap $opennn $csvFull 2>&1
    } elseif ($engine -eq "tensorflow") {
        $out = & $capper $cap "python" "$PSScriptRoot\tensorflow_capacity.py" $csvFull "float64" 2>&1
    } else {
        $out = & $capper $cap "python" "$PSScriptRoot\pytorch_capacity.py" $csvFull "float64" 2>&1
    }
    $res = "FAIL"
    foreach ($l in $out) {
        if ($l -match 'RESULT=(\w+)') { $res = $Matches[1] }
    }
    # If the job killed the child before it could print RESULT, treat as OOM.
    $childExit = ($out | Select-String 'child_exit=(\d+)' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -Last 1)
    if ($res -eq "FAIL" -and $childExit -and [uint32]$childExit -ne 0) { $res = "OOM" }
    return $res
}

foreach ($engine in @("opennn","pytorch","tensorflow")) {
    Write-Host "######## $engine (cap ${CapGB}GB) ########" -ForegroundColor Cyan
    $samples = $StartSamples
    $lastOK = 0
    while ($samples -le $MaxSamples) {
        $csv = "$WorkDir\higgs_tiled_${samples}s.csv"
        $valuesB = [math]::Round(($samples * $valuesPerRow) / 1e9, 3)
        Write-Host ("  {0,12:N0} samples (~{1} B values) tiling..." -f $samples, $valuesB)
        & $tiler $HiggsCsv $samples $csv | Out-Null
        $full = (Resolve-Path $csv).Path

        $res = Test-Engine $engine $full
        Write-Host ("    -> {0}" -f $res) -ForegroundColor Yellow
        "$engine,$CapGB,$samples,$valuesB,$res" | Out-File -FilePath $ResultsCsv -Append -Encoding utf8

        Remove-Item $csv -Force

        if ($res -eq "OK") { $lastOK = $samples; $samples += $StepSamples }
        else {
            Write-Host ("  $engine max: {0:N0} samples ({1} B values) before OOM at {2:N0}" -f $lastOK, [math]::Round(($lastOK*$valuesPerRow)/1e9,3), $samples) -ForegroundColor Green
            break
        }
    }
}

Write-Host "`n==== capacity_results.csv ====" -ForegroundColor Green
Get-Content $ResultsCsv

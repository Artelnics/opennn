# Capacity search: under a fixed committed-memory cap (Job Object), find the
# largest Rosenbrock dataset each engine can load + train before OOM.
#
# Both engines run inside run_capped.exe with the SAME cap, so the crash point
# is determined by each engine's memory efficiency, not by background apps.
#
#   usage:  powershell -File capacity_search.ps1 -CapGB 8 -Variables 100

param(
    [double]$CapGB = 8,
    [int]$Variables = 100,
    [long]$StartSamples = 1000000,
    [long]$StepSamples = 1000000,
    [long]$MaxSamples = 30000000,
    [string]$WorkDir = "$PSScriptRoot\_data",
    [string]$ResultsCsv = "$PSScriptRoot\capacity_results.csv"
)

$ErrorActionPreference = "Stop"
$gen     = "$PSScriptRoot\generate_rosenbrock.exe"
$capper  = "$PSScriptRoot\run_capped.exe"
$opennn  = "$PSScriptRoot\..\..\..\build\bin\Release\opennn_capacity.exe"
$cap     = [long]($CapGB * 1GB)

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
"engine,cap_gb,variables,samples,values_billion,result" | Out-File -FilePath $ResultsCsv -Encoding utf8

function Test-Engine($engine, $csvFull, $samples) {
    # Returns "OK" or "OOM"/"FAIL". Runs the engine under the memory cap.
    if ($engine -eq "opennn") {
        $out = & $capper $cap $opennn $csvFull $Variables 2>&1
    } else {
        $out = & $capper $cap "python" "$PSScriptRoot\pytorch_capacity.py" $csvFull $Variables "float64" 2>&1
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

foreach ($engine in @("opennn","pytorch")) {
    Write-Host "######## $engine (cap ${CapGB}GB) ########" -ForegroundColor Cyan
    $samples = $StartSamples
    $lastOK = 0
    while ($samples -le $MaxSamples) {
        $csv = "$WorkDir\rb_${Variables}v_${samples}s.csv"
        $valuesB = [math]::Round(($samples * ($Variables + 1)) / 1e9, 3)
        Write-Host ("  {0,12:N0} samples (~{1} B values) generating..." -f $samples, $valuesB)
        & $gen $Variables $samples $csv 1234 | Out-Null
        $full = (Resolve-Path $csv).Path

        $res = Test-Engine $engine $full $samples
        Write-Host ("    -> {0}" -f $res) -ForegroundColor Yellow
        "$engine,$CapGB,$Variables,$samples,$valuesB,$res" | Out-File -FilePath $ResultsCsv -Append -Encoding utf8

        Remove-Item $csv -Force

        if ($res -eq "OK") { $lastOK = $samples; $samples += $StepSamples }
        else {
            Write-Host ("  $engine max: {0:N0} samples ({1} B values) before OOM at {2:N0}" -f $lastOK, [math]::Round(($lastOK*($Variables+1))/1e9,3), $samples) -ForegroundColor Green
            break
        }
    }
}

Write-Host "`n==== capacity_results.csv ====" -ForegroundColor Green
Get-Content $ResultsCsv
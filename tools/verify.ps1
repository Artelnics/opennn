[CmdletBinding()]
param(
    [ValidateSet("quick", "cpu", "cuda", "full")]
    [string] $Mode = "quick",

    [ValidateSet("cpu", "cuda")]
    [string] $Backend = "cpu",

    [string] $Filter,
    [string] $BuildRoot,
    [ValidateRange(1, 1024)]
    [int] $Jobs,
    [switch] $Reconfigure,
    [switch] $NoSccache
)

$ErrorActionPreference = "Stop"

if ($Mode -eq "quick" -and [string]::IsNullOrWhiteSpace($Filter)) {
    throw "Quick verification requires -Filter. Example: -Filter 'Dense.*:DenseNoBiasTest.*'"
}

function Import-VisualStudioEnvironment {
    if ((Get-Command cl.exe -ErrorAction SilentlyContinue) -and
        (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
        return
    }

    $programFilesX86 = [Environment]::GetFolderPath("ProgramFilesX86")
    $installerDirectory = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer"
    $vswhere = Join-Path $installerDirectory "vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw "Visual Studio Build Tools were not found (missing $vswhere)."
    }

    $installation = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($installation)) {
        throw "No Visual Studio installation with the C++ x64 tools was found."
    }
    $installation = $installation.Trim()

    $env:Path = "$installerDirectory;$env:Path"
    $devCmd = Join-Path $installation "Common7\Tools\VsDevCmd.bat"
    $cmdLine = "`"$devCmd`" -arch=x64 -host_arch=x64 >nul && set"
    $environmentLines = & $env:ComSpec /d /s /c $cmdLine
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize the Visual Studio x64 environment."
    }

    foreach ($line in $environmentLines) {
        $separator = $line.IndexOf("=")
        if ($separator -gt 0) {
            $name = $line.Substring(0, $separator)
            $value = $line.Substring($separator + 1)
            Set-Item -Path "Env:$name" -Value $value
        }
    }

    if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
        $ninjaDirectory = Join-Path $installation `
            "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
        if (Test-Path -LiteralPath (Join-Path $ninjaDirectory "ninja.exe")) {
            $env:Path = "$ninjaDirectory;$env:Path"
        }
    }
}

Import-VisualStudioEnvironment

$cmake = Get-Command cmake.exe -ErrorAction SilentlyContinue
if (-not $cmake) {
    throw "cmake.exe was not found on PATH."
}
if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
    throw "ninja.exe was not found after initializing Visual Studio."
}

$scriptPath = Join-Path $PSScriptRoot "verify.cmake"
$arguments = @(
    "-DOPENNN_VERIFY_MODE=$Mode",
    "-DOPENNN_VERIFY_BACKEND=$Backend"
)
if (-not [string]::IsNullOrWhiteSpace($Filter)) {
    $arguments += "-DOPENNN_TEST_FILTER=$Filter"
}
if (-not [string]::IsNullOrWhiteSpace($BuildRoot)) {
    $arguments += "-DOPENNN_BUILD_ROOT:PATH=$BuildRoot"
}
if ($Jobs -gt 0) {
    $arguments += "-DOPENNN_VERIFY_JOBS=$Jobs"
}
if ($Reconfigure) {
    $arguments += "-DOPENNN_VERIFY_RECONFIGURE=ON"
}
if ($NoSccache) {
    $arguments += "-DOPENNN_USE_SCCACHE=OFF"
}
$arguments += @("-P", $scriptPath)

& $cmake.Source @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

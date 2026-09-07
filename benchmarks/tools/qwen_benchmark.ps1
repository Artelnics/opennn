[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('prepare', 'build', 'smoke', 'run', 'unlock')]
    [string]$Action = 'run',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArguments
)

$ErrorActionPreference = 'Stop'
$Repository = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$DataRoot = if ($env:OPENNN_BENCH_DATA) {
    [IO.Path]::GetFullPath($env:OPENNN_BENCH_DATA)
} else {
    Join-Path $env:USERPROFILE 'opennn-benchmark-data'
}
$QwenRoot = Join-Path $DataRoot 'qwen3'
$ToolsRoot = Join-Path $QwenRoot 'tools'
$UvVersion = '0.12.9'
$PythonVersion = '3.12.3'
$UvDirectory = Join-Path $ToolsRoot "uv-$UvVersion"
$Uv = Join-Path $UvDirectory 'uv.exe'
$Venv = Join-Path $QwenRoot 'python\venv'
$Python = Join-Path $Venv 'Scripts\python.exe'

$env:OPENNN_BENCH_DATA = $DataRoot

function Get-PortablePython {
    New-Item -ItemType Directory -Force -Path $UvDirectory | Out-Null
    if (-not (Test-Path -LiteralPath $Uv)) {
        $Download = Join-Path $ToolsRoot 'downloads'
        New-Item -ItemType Directory -Force -Path $Download | Out-Null
        $Archive = Join-Path $Download "uv-$UvVersion-x86_64-pc-windows-msvc.zip"
        $Checksum = "$Archive.sha256"
        $Base = "https://github.com/astral-sh/uv/releases/download/$UvVersion"
        if (-not (Test-Path -LiteralPath $Archive)) {
            Invoke-WebRequest -Uri "$Base/uv-x86_64-pc-windows-msvc.zip" -OutFile $Archive
        }
        Invoke-WebRequest -Uri "$Base/uv-x86_64-pc-windows-msvc.zip.sha256" -OutFile $Checksum
        $Expected = ((Get-Content -LiteralPath $Checksum -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
        $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
        if ($Actual -ne $Expected) {
            throw "uv archive hash mismatch: $Actual != $Expected"
        }
        Expand-Archive -LiteralPath $Archive -DestinationPath $UvDirectory -Force
    }

    $env:UV_PYTHON_INSTALL_DIR = Join-Path $QwenRoot 'python\installations'
    $env:UV_CACHE_DIR = Join-Path $QwenRoot 'python\cache'
    $env:OPENNN_BENCH_UV = $Uv
    if (-not (Test-Path -LiteralPath $Python)) {
        & $Uv python install $PythonVersion
        if ($LASTEXITCODE -ne 0) { throw 'uv python install failed' }
        & $Uv venv --python $PythonVersion $Venv
        if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' }
    }
    & $Uv pip install --python $Python 'numpy>=2.0,<3' safetensors 'huggingface_hub[hf_xet]>=0.34,<2'
    if ($LASTEXITCODE -ne 0) { throw 'benchmark Python dependency installation failed' }
}

function Import-VisualStudioEnvironment {
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $VsWhere)) { throw "vswhere.exe not found at $VsWhere" }
    $VsRoot = (& $VsWhere -latest -property installationPath).Trim()
    if (-not $VsRoot) { throw 'Visual Studio installation not found' }
    $VcVars = Join-Path $VsRoot 'VC\Auxiliary\Build\vcvars64.bat'
    $EnvironmentLines = & cmd.exe /d /s /c "`"$VcVars`" >nul && set"
    if ($LASTEXITCODE -ne 0) { throw 'vcvars64.bat failed' }
    foreach ($Line in $EnvironmentLines) {
        $Separator = $Line.IndexOf('=')
        if ($Separator -gt 0) {
            [Environment]::SetEnvironmentVariable(
                $Line.Substring(0, $Separator),
                $Line.Substring($Separator + 1),
                'Process')
        }
    }
    $Ninja = Join-Path $VsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja'
    $env:PATH = "$Ninja;$env:PATH"
}

function Find-Cudnn {
    $Root = Join-Path $env:ProgramFiles 'NVIDIA\CUDNN'
    $Headers = Get-ChildItem -LiteralPath $Root -Filter cudnn_version.h -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending
    $CudaVersion = (& nvcc --version | Select-String 'release\s+([0-9]+)\.').Matches.Groups[1].Value
    foreach ($Header in $Headers) {
        $VersionDirectory = $Header.Directory.Name
        if ($CudaVersion -and -not $VersionDirectory.StartsWith("$CudaVersion.")) { continue }
        $InstallRoot = $Header.Directory.Parent.Parent.FullName
        $Library = Join-Path $InstallRoot "lib\$VersionDirectory\x64\cudnn.lib"
        if (Test-Path -LiteralPath $Library) {
            return @{ Include = $Header.Directory.FullName; Library = $Library }
        }
    }
    throw "cuDNN headers/libraries not found below $Root"
}

function Build-Benchmarks {
    Import-VisualStudioEnvironment
    $Cudnn = Find-Cudnn
    $OpenNNBuild = Join-Path $QwenRoot 'build\opennn-windows-cuda'
    cmake -S $Repository -B $OpenNNBuild -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        -DOpenNN_DISABLE_CUDA=OFF `
        -DOpenNN_BUILD_BENCHMARKS=ON `
        -DOpenNN_BUILD_EXAMPLES=OFF `
        -DCMAKE_CUDA_ARCHITECTURES=89 `
        "-DCUDNN_INCLUDE_DIR=$($Cudnn.Include)" `
        "-DCUDNN_LIBRARY=$($Cudnn.Library)"
    if ($LASTEXITCODE -ne 0) { throw 'OpenNN CMake configure failed' }
    cmake --build $OpenNNBuild --target qwen_opennn opennn_tests
    if ($LASTEXITCODE -ne 0) { throw 'OpenNN Qwen benchmark build failed' }

    $OpenNNBinary = Get-ChildItem -LiteralPath $OpenNNBuild -Filter qwen_opennn.exe -Recurse -File |
        Select-Object -First 1
    if (-not $OpenNNBinary) { throw "qwen_opennn.exe not found below $OpenNNBuild" }
    $env:OPENNN_QWEN_BIN = $OpenNNBinary.FullName

    $LlamaSource = Join-Path $ToolsRoot 'llama.cpp'
    if (-not (Test-Path -LiteralPath (Join-Path $LlamaSource 'CMakeLists.txt'))) {
        throw 'llama.cpp is not prepared; run the prepare action first'
    }
    $LlamaBuild = Join-Path $LlamaSource 'build-windows-cuda'
    cmake -S $LlamaSource -B $LlamaBuild -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        -DGGML_CUDA=ON `
        -DGGML_NATIVE=OFF `
        -DCMAKE_CUDA_ARCHITECTURES=89
    if ($LASTEXITCODE -ne 0) { throw 'llama.cpp CMake configure failed' }
    cmake --build $LlamaBuild --target llama-bench llama-server
    if ($LASTEXITCODE -ne 0) { throw 'llama.cpp benchmark build failed' }

    $env:OPENNN_LLAMA_BENCH_BIN = (Get-ChildItem -LiteralPath $LlamaBuild -Filter llama-bench.exe -Recurse -File |
        Select-Object -First 1).FullName
    $env:OPENNN_LLAMA_SERVER_BIN = (Get-ChildItem -LiteralPath $LlamaBuild -Filter llama-server.exe -Recurse -File |
        Select-Object -First 1).FullName
    $env:OPENNN_OLLAMA_BIN = Join-Path $ToolsRoot "ollama\v0.33.2\ollama.exe"
}

function Set-BenchmarkClocks {
    & nvidia-smi -lgc 2505,2505
    $Graphics = $LASTEXITCODE
    & nvidia-smi -lmc 11201,11201
    $Memory = $LASTEXITCODE
    if ($Graphics -eq 0 -and $Memory -eq 0) {
        $env:OPENNN_BENCH_CLOCKS_LOCKED = '1'
        return $true
    }
    Remove-Item Env:OPENNN_BENCH_CLOCKS_LOCKED -ErrorAction SilentlyContinue
    Write-Warning 'The driver refused one or both clock locks; results will be diagnostic-only.'
    return $false
}

function Reset-BenchmarkClocks {
    & nvidia-smi -rgc | Out-Host
    & nvidia-smi -rmc | Out-Host
    Remove-Item Env:OPENNN_BENCH_CLOCKS_LOCKED -ErrorAction SilentlyContinue
}

function Set-BinaryEnvironment {
    $OpenNNBuild = Join-Path $QwenRoot 'build\opennn-windows-cuda'
    $LlamaBuild = Join-Path $ToolsRoot 'llama.cpp\build-windows-cuda'
    $env:OPENNN_QWEN_BIN = (Get-ChildItem -LiteralPath $OpenNNBuild -Filter qwen_opennn.exe -Recurse -File |
        Select-Object -First 1).FullName
    $env:OPENNN_LLAMA_BENCH_BIN = (Get-ChildItem -LiteralPath $LlamaBuild -Filter llama-bench.exe -Recurse -File |
        Select-Object -First 1).FullName
    $env:OPENNN_LLAMA_SERVER_BIN = (Get-ChildItem -LiteralPath $LlamaBuild -Filter llama-server.exe -Recurse -File |
        Select-Object -First 1).FullName
    $env:OPENNN_OLLAMA_BIN = Join-Path $ToolsRoot 'ollama\v0.33.2\ollama.exe'
}

New-Item -ItemType Directory -Force -Path $QwenRoot | Out-Null

switch ($Action) {
    'prepare' {
        Get-PortablePython
        $env:HF_HOME = Join-Path $QwenRoot 'huggingface-cache'
        $env:HF_XET_CACHE = Join-Path $QwenRoot 'huggingface-cache\xet'
        & $Python (Join-Path $Repository 'benchmarks\prepare.py') qwen --data-root $DataRoot
        if ($LASTEXITCODE -ne 0) { throw 'Qwen preparation or validation failed' }
    }
    'build' {
        Get-PortablePython
        Build-Benchmarks
    }
    'unlock' {
        Reset-BenchmarkClocks
    }
    { $_ -in @('smoke', 'run') } {
        Get-PortablePython
        Import-VisualStudioEnvironment
        Set-BinaryEnvironment
        [void](Set-BenchmarkClocks)
        try {
            $Arguments = @('--family', 'qwen')
            if ($Action -eq 'smoke') {
                $Arguments += @('--prompt-tokens', '128', '--generate-tokens', '16',
                                '--rounds', '1', '--repeats', '1', '--no-wait', '--label', 'smoke')
            }
            if ($RunnerArguments) { $Arguments += $RunnerArguments }
            & $Python (Join-Path $Repository 'benchmarks\run.py') @Arguments
            if ($LASTEXITCODE -notin @(0, 3)) { throw "benchmark runner failed with $LASTEXITCODE" }
        }
        finally {
            Reset-BenchmarkClocks
        }
    }
}

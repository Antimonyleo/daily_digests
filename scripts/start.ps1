[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8765,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

function Find-Uv {
    $Command = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -ne $Command) {
        return $Command.Source
    }

    $LocalUv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path $LocalUv) {
        return $LocalUv
    }
    return $null
}

$UvPath = Find-Uv
if ($null -eq $UvPath) {
    Write-Error "DailyDigest needs uv. Run scripts\install.ps1 once, then try again."
    exit 1
}

New-Item -ItemType Directory -Force -Path "data" | Out-Null

Write-Host "Syncing DailyDigest dependencies..."
& $UvPath sync --frozen
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$StartArguments = @(
    "run", "dd", "start",
    "--host", $HostAddress,
    "--port", $Port
)
if ($NoBrowser) {
    $StartArguments += "--no-browser"
}

& $UvPath @StartArguments
exit $LASTEXITCODE

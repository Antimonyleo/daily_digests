[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$LocalUv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $UvCommand -and -not (Test-Path $LocalUv)) {
    Write-Host "Installing uv (one time; no administrator access required)..."
    $Installer = Invoke-RestMethod "https://astral.sh/uv/install.ps1"
    Invoke-Expression $Installer
}

$UvBin = Join-Path $env:USERPROFILE ".local\bin"
$env:Path = $UvBin + [IO.Path]::PathSeparator + $env:Path
$UvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $UvCommand) {
    if (-not (Test-Path $LocalUv)) {
        Write-Error "uv did not install correctly. See https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    }
}

Write-Host "Starting DailyDigest..."
& (Join-Path $PSScriptRoot "start.ps1")
exit $LASTEXITCODE

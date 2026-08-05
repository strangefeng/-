param(
    [int]$MaxNew = 0,
    [int]$Epochs = 0,
    [int]$Batch = 0,
    [string]$Device = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$ArgsList = @(".\scripts\run_aircraft_ai_round16.py")

if ($MaxNew -gt 0) {
    $ArgsList += "--max-new"
    $ArgsList += "$MaxNew"
}

if ($Epochs -gt 0) {
    $ArgsList += "--epochs"
    $ArgsList += "$Epochs"
}

if ($Batch -gt 0) {
    $ArgsList += "--batch"
    $ArgsList += "$Batch"
}

if ($Device -ne "") {
    $ArgsList += "--device"
    $ArgsList += "$Device"
}

python @ArgsList

param(
    [string[]]$Only,
    [switch]$DryRun,
    [string]$LogPath
)

$ErrorActionPreference = 'Stop'

$Repo = Split-Path -Parent $PSScriptRoot
$Python = 'D:\yolo11\.conda\python.exe'
$Manifest = Join-Path $Repo 'experiments\aircraft_ai_dataset_mainline.json'
$SummaryScript = Join-Path $Repo 'scripts\summarize_aircraft_ai_suite.py'
$SuiteCsv = Join-Path $Repo 'results\aircraft_ai_dataset_mainline.csv'
$SummaryMd = Join-Path $Repo 'analysis\aircraft_ai_dataset_mainline_summary.md'
$LogDir = Join-Path $Repo 'runs\aircraft_ai_mainline\logs'

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (-not $LogPath) {
    $LogPath = Join-Path $LogDir 'aircraft_ai_full_suite.transcript.log'
}

Start-Transcript -Path $LogPath -Append | Out-Null

$env:YOLO_CONFIG_DIR = Join-Path $Repo '.yolo-config'

$argsList = @(
    (Join-Path $Repo 'run_experiment_suite.py'),
    '--manifest', $Manifest,
    '--export-format', 'none'
)

if ($Only) {
    $argsList += '--only'
    $argsList += $Only
}

if ($DryRun) {
    $argsList += '--dry-run'
}

Write-Host "[suite] repo=$Repo"
Write-Host "[suite] manifest=$Manifest"
Write-Host "[suite] csv=$SuiteCsv"
Write-Host "[suite] summary=$SummaryMd"

& $Python @argsList
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $DryRun) {
    & $Python $SummaryScript --csv $SuiteCsv --output $SummaryMd --suite-id 'aircraft_ai_dataset_mainline'
    if ($LASTEXITCODE -ne 0) {
        Stop-Transcript | Out-Null
        exit $LASTEXITCODE
    }
}

Write-Host "[suite] done"
Stop-Transcript | Out-Null

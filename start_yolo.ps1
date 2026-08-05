param(
    [ValidateSet('menu', 'list', 'verify', 'smoke', 'formal', 'stage2', 'summary-stage2', 'audit-stage2', 'status')]
    [string]$Action = 'menu'
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = 'D:\yolo11\.conda\python.exe'
$Weights = [System.IO.Path]::GetFullPath((Join-Path $Repo '..\yolo11n.pt'))
$Runner = Join-Path $Repo 'run_experiment_suite.py'
$ResultsCsv = Join-Path $Repo 'results\aircraft_skin_defects_new_mainline.csv'
$Stage2Summary = Join-Path $Repo 'analysis\stage2_formal_summary\summary.md'
$SmokeDataset = Join-Path $Repo 'datasets\standardized_5cls\aircraft_skin_defects\data.yaml'

function Initialize-RepoRuntime {
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Dedicated Python not found: $Python"
    }
    if (-not (Test-Path -LiteralPath $Weights)) {
        throw "Model weights not found: $Weights"
    }
    $configDir = Join-Path $Repo '.yolo-config'
    New-Item -ItemType Directory -Force -Path $configDir | Out-Null
    $env:YOLO_CONFIG_DIR = $configDir
    Set-Location $Repo
}

function Invoke-RepoPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-RepoScript {
    param(
        [string]$ScriptPath,
        [string[]]$ScriptArgs = @()
    )
    & $Python $ScriptPath @ScriptArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Script failed with exit code $LASTEXITCODE"
    }
}

function Show-Header {
    param([string]$Title)
    Write-Host "`n==== $Title ====" -ForegroundColor Cyan
}

function Show-ExperimentList {
    Show-Header 'Experiment List'
    Invoke-RepoScript $Runner @('--list')
}

function Invoke-VerifyEnvironment {
    Show-Header 'Verify Environment'
    Invoke-RepoPython '-c' "import numpy; print('numpy', numpy.__version__)"
    Invoke-RepoPython '-c' "import cv2; print('cv2', cv2.__version__)"
    Invoke-RepoPython '-c' "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
    Invoke-RepoPython '-c' "import ultralytics; print('ultralytics', ultralytics.__version__)"
}

function Invoke-SmokeTest {
    Show-Header 'Smoke Test'
    $device = (& $Python -c "import torch; print('0' if torch.cuda.is_available() else 'cpu')").Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to detect device"
    }
    Invoke-RepoScript (Join-Path $Repo 'train_one.py') @(
        '--dataset-yaml', $SmokeDataset,
        '--model', $Weights,
        '--imgsz', '640',
        '--epochs', '3',
        '--batch', '16',
        '--device', $device,
        '--project', (Join-Path $Repo 'runs\smoke_clean'),
        '--name', 'smoke_aircraft_skin_defects_clean'
    )
}

function Invoke-FormalBaseline {
    Show-Header 'Formal Baseline'
    & powershell -ExecutionPolicy Bypass -File (Join-Path $Repo 'scripts\run_aircraft_new_formal.ps1')
    if ($LASTEXITCODE -ne 0) {
        throw "Formal baseline launcher failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Stage2Formal {
    Show-Header 'Stage-2 Formal'
    & powershell -ExecutionPolicy Bypass -File (Join-Path $Repo 'scripts\run_stage2_formal.ps1')
    if ($LASTEXITCODE -ne 0) {
        throw "Stage-2 launcher failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Stage2Summary {
    Show-Header 'Stage-2 Summary'
    Invoke-RepoScript (Join-Path $Repo 'scripts\summarize_stage2_results.py')
    if (Test-Path -LiteralPath $Stage2Summary) {
        Write-Host "Summary written to $Stage2Summary" -ForegroundColor Green
    }
}

function Invoke-Stage2Audit {
    Show-Header 'Stage-2 Scratch Audit'
    Invoke-RepoScript (Join-Path $Repo 'scripts\audit_stage2_scratch_cases.py')
}

function Show-Status {
    Show-Header 'Runtime Status'
    $procs = Get-Process | Where-Object { $_.ProcessName -like '*python*' -or $_.ProcessName -like '*yolo*' } |
        Select-Object ProcessName, Id, StartTime, Responding
    if ($procs) {
        $procs | Format-Table -AutoSize
    } else {
        Write-Host 'No python/yolo training process is running right now.' -ForegroundColor Yellow
    }

    if (Test-Path -LiteralPath $ResultsCsv) {
        Show-Header 'Latest Recorded Results'
        $rows = Import-Csv $ResultsCsv | Select-Object -Last 8
        $rows |
            Select-Object experiment_id, experiment_title, epochs, run_status, resolved_device, map50_95, scratch_ap50_95 |
            Format-Table -AutoSize
    } else {
        Write-Host "Results CSV not found: $ResultsCsv" -ForegroundColor Yellow
    }

    if (Test-Path -LiteralPath $Stage2Summary) {
        Show-Header 'Latest Stage-2 Summary'
        Get-Content -Path $Stage2Summary -TotalCount 40
    }
}

function Show-Menu {
    while ($true) {
        Write-Host ''
        Write-Host 'YOLO Launcher' -ForegroundColor Cyan
        Write-Host '1. List experiments'
        Write-Host '2. Verify environment'
        Write-Host '3. Run smoke test'
        Write-Host '4. Run formal baseline'
        Write-Host '5. Run stage-2 formal experiments'
        Write-Host '6. Generate stage-2 summary'
        Write-Host '7. Run stage-2 scratch audit'
        Write-Host '8. Show status'
        Write-Host '9. Exit'
        $choice = Read-Host 'Choose an action (1-9)'
        switch ($choice) {
            '1' { Show-ExperimentList }
            '2' { Invoke-VerifyEnvironment }
            '3' { Invoke-SmokeTest }
            '4' { Invoke-FormalBaseline }
            '5' { Invoke-Stage2Formal }
            '6' { Invoke-Stage2Summary }
            '7' { Invoke-Stage2Audit }
            '8' { Show-Status }
            '9' { return }
            default { Write-Host 'Invalid selection. Please choose 1-9.' -ForegroundColor Yellow }
        }
    }
}

Initialize-RepoRuntime

switch ($Action) {
    'menu' { Show-Menu }
    'list' { Show-ExperimentList }
    'verify' { Invoke-VerifyEnvironment }
    'smoke' { Invoke-SmokeTest }
    'formal' { Invoke-FormalBaseline }
    'stage2' { Invoke-Stage2Formal }
    'summary-stage2' { Invoke-Stage2Summary }
    'audit-stage2' { Invoke-Stage2Audit }
    'status' { Show-Status }
    default { throw "Unsupported action: $Action" }
}

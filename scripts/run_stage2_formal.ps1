$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:YOLO_CONFIG_DIR = Join-Path $repo '.yolo-config'
$python = 'D:\yolo11\.conda\python.exe'
$project = Join-Path $repo 'runs\aircraft_new_stage2_formal'
$experiments = @('Exp_A', 'Exp_B', 'Exp_C')
foreach ($experiment in $experiments) {
    Write-Host "[stage2] starting $experiment" -ForegroundColor Cyan
    & $python 'run_experiment_suite.py' --only $experiment --device 0 --epochs 100 --batch 16 --workers 8 --imgsz 640 --project $project
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment $experiment failed with exit code $LASTEXITCODE"
    }
    Write-Host "[stage2] finished $experiment" -ForegroundColor Green
}

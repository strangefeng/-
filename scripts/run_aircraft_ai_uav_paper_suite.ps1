param(
  [ValidateSet("quick", "baselines", "ablation", "robustness-data", "robustness-val", "edge-export", "all")]
  [string]$Stage = "quick",
  [int]$Epochs = 100,
  [int]$ImgSize = 800,
  [int]$Batch = 8,
  [string]$Device = "auto",
  [string]$Python = "D:\yolo11\.conda\python.exe"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$env:YOLO_CONFIG_DIR = Join-Path $Repo ".yolo-config"
if (-not (Test-Path $Python)) {
  $Python = "python"
}

function Run-Suite($Manifest, $Only = @()) {
  $args = @(
    ".\run_experiment_suite.py",
    "--manifest", $Manifest,
    "--imgsz", "$ImgSize",
    "--epochs", "$Epochs",
    "--batch", "$Batch",
    "--device", "$Device",
    "--skip-existing"
  )
  if ($Only.Count -gt 0) {
    $args += "--only"
    $args += $Only
  }
  & $Python @args
}

function Build-RobustnessData {
  & $Python ".\scripts\build_uav_corruption_benchmark.py" --source-split test --levels 1 2 3
}

function Validate-Robustness {
  $yamlDir = ".\datasets\robustness\aircraft_ai_uav_clean_split_corruptions\yamls"
  & $Python ".\scripts\validate_paper_suite.py" `
    --source-csv ".\results\aircraft_ai_uav_paper_baselines.csv" ".\results\aircraft_ai_uav_paper_ablation.csv" `
    --yaml-dir $yamlDir `
    --imgsz $ImgSize `
    --batch $Batch `
    --device $Device `
    --skip-existing
}

function Export-EdgeModels {
  & $Python ".\scripts\export_paper_models.py" `
    --source-csv ".\results\aircraft_ai_uav_paper_baselines.csv" ".\results\aircraft_ai_uav_paper_ablation.csv" `
    --formats onnx `
    --imgsz $ImgSize
}

if ($Stage -eq "quick") {
  Run-Suite ".\experiments\aircraft_ai_uav_paper_baselines.json" @("B03_YOLOV8N", "B09_YOLO11N", "B11_YOLO26N")
  Run-Suite ".\experiments\aircraft_ai_uav_paper_ablation.json" @("A01_YOLO26N", "A02_YOLO26N_NWD_LOSS", "A03_YOLO26N_NWD_ASSIGNER")
}
elseif ($Stage -eq "baselines") {
  Run-Suite ".\experiments\aircraft_ai_uav_paper_baselines.json"
}
elseif ($Stage -eq "ablation") {
  Run-Suite ".\experiments\aircraft_ai_uav_paper_ablation.json"
}
elseif ($Stage -eq "robustness-data") {
  Build-RobustnessData
}
elseif ($Stage -eq "robustness-val") {
  Validate-Robustness
}
elseif ($Stage -eq "edge-export") {
  Export-EdgeModels
}
elseif ($Stage -eq "all") {
  Run-Suite ".\experiments\aircraft_ai_uav_paper_baselines.json"
  Run-Suite ".\experiments\aircraft_ai_uav_paper_ablation.json"
  Build-RobustnessData
  Validate-Robustness
  Export-EdgeModels
}

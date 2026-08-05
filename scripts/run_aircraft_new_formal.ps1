param()

$ErrorActionPreference = 'Stop'
$repo = 'D:\桌面\毕设\数据集\yolo-baseline-repro'
$python = 'D:\yolo11\.conda\python.exe'
$runner = Join-Path $repo 'run_experiment_suite.py'
$model = 'D:\桌面\毕设\数据集\yolo11n.pt'
$project = Join-Path $repo 'runs\aircraft_new_formal'
$logDir = Join-Path $repo 'logs'
$queueLog = Join-Path $logDir 'aircraft_new_formal_queue.log'
$statusJson = Join-Path $logDir 'aircraft_new_formal_status.json'

New-Item -ItemType Directory -Force -Path $project | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:YOLO_CONFIG_DIR = Join-Path $repo '.yolo-config'
Set-Location $repo

function Write-Status {
    param(
        [string]$Stage,
        [string]$State,
        [string]$LogPath,
        [string]$Message
    )
    $payload = [ordered]@{
        timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        stage = $Stage
        state = $State
        log_path = $LogPath
        message = $Message
        project = $project
    } | ConvertTo-Json -Depth 3
    Set-Content -Encoding UTF8 $statusJson $payload
}

function Invoke-Stage {
    param(
        [string]$ExperimentId,
        [string]$StageName
    )

    $logPath = Join-Path $logDir ("{0}_{1}.log" -f $StageName, (Get-Date -Format 'yyyyMMdd_HHmmss'))
    Add-Content -Encoding UTF8 $queueLog ("[{0}] START {1} log={2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $ExperimentId, $logPath)
    Write-Status -Stage $ExperimentId -State 'running' -LogPath $logPath -Message 'training started'

    & $python $runner --only $ExperimentId --model $model --project $project 2>&1 | Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Add-Content -Encoding UTF8 $queueLog ("[{0}] FAIL {1} exit_code={2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $ExperimentId, $exitCode)
        Write-Status -Stage $ExperimentId -State 'failed' -LogPath $logPath -Message ("exit_code={0}" -f $exitCode)
        exit $exitCode
    }

    Add-Content -Encoding UTF8 $queueLog ("[{0}] DONE {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $ExperimentId)
    Write-Status -Stage $ExperimentId -State 'completed' -LogPath $logPath -Message 'training completed'
}

Write-Status -Stage 'queue' -State 'running' -LogPath $queueLog -Message 'B0 then B1 queued on single CPU'
Invoke-Stage -ExperimentId 'B0' -StageName 'B0_yolo11_default_formal'
Invoke-Stage -ExperimentId 'B1' -StageName 'B1_yolo11_taskadapt_formal'
Write-Status -Stage 'queue' -State 'completed' -LogPath $queueLog -Message 'B0 and B1 completed'


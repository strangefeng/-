@echo off
setlocal enableextensions

for %%I in ("%~dp0..") do set "REPO=%%~fI"
for %%I in ("%REPO%\..\yolo11n.pt") do set "MODEL=%%~fI"
set "PYTHON=D:\yolo11\.conda\python.exe"
set "RUNNER=%REPO%\run_experiment_suite.py"
set "PROJECT=%REPO%\runs\aircraft_new_formal"
set "LOGDIR=%REPO%\logs"
set "QUEUELOG=%LOGDIR%\aircraft_new_formal_queue.log"
set "M1LOG=%LOGDIR%\M1_formal.log"
set "STATUS=%LOGDIR%\aircraft_new_formal_status.txt"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
if not exist "%PROJECT%" mkdir "%PROJECT%"
set "YOLO_CONFIG_DIR=%REPO%\.yolo-config"

pushd "%REPO%"
echo [%date% %time%] START M1 > "%QUEUELOG%"
echo stage=M1 state=running log=%M1LOG% > "%STATUS%"
"%PYTHON%" "%RUNNER%" --only M1 --model "%MODEL%" --project "%PROJECT%" > "%M1LOG%" 2>&1
if errorlevel 1 goto fail_m1

echo [%date% %time%] DONE M1 >> "%QUEUELOG%"
echo stage=M1 state=completed log=%M1LOG% > "%STATUS%"
popd
exit /b 0

:fail_m1
echo [%date% %time%] FAIL M1 >> "%QUEUELOG%"
echo stage=M1 state=failed log=%M1LOG% > "%STATUS%"
popd
exit /b 1

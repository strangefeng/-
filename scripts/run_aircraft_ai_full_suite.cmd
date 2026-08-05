@echo off
setlocal

set "REPO=D:\桌面\毕设\数据集\yolo-baseline-repro"
set "PYTHON=D:\yolo11\.conda\python.exe"
set "CSV=%REPO%\results\aircraft_ai_dataset_mainline.csv"
set "SUMMARY=%REPO%\analysis\aircraft_ai_dataset_mainline_summary.md"
set "LOGDIR=%REPO%\runs\aircraft_ai_mainline\logs"
set "STDOUT=%LOGDIR%\aircraft_ai_full_suite_task.out.log"
set "STDERR=%LOGDIR%\aircraft_ai_full_suite_task.err.log"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
if exist "%CSV%" del /f /q "%CSV%"
if exist "%SUMMARY%" del /f /q "%SUMMARY%"

cd /d "%REPO%"
"%PYTHON%" "%REPO%\scripts\run_aircraft_ai_full_suite.py" 1>"%STDOUT%" 2>"%STDERR%"

endlocal

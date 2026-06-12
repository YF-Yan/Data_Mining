@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1

REM 一键：训练 → 启动 Streamlit
REM   run_dev.bat
REM   run_dev.bat --skip-train
REM   run_dev.bat --train-only
REM   run_dev.bat --download
REM   run_dev.bat --download-only
REM   run_dev.bat -- --k 4 --seed 42
REM   run_dev.bat -- --force          强制重训（忽略已有 output）

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "SKIP_TRAIN=0"
set "TRAIN_ONLY=0"
set "DO_DOWNLOAD=0"
set "DOWNLOAD_ONLY=0"
set "TRAIN_ARGS="

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="--skip-train" (set "SKIP_TRAIN=1" & shift & goto parse_args)
if /i "%~1"=="--train-only" (set "TRAIN_ONLY=1" & shift & goto parse_args)
if /i "%~1"=="--download-only" (set "DO_DOWNLOAD=1" & set "DOWNLOAD_ONLY=1" & shift & goto parse_args)
if /i "%~1"=="--download" (set "DO_DOWNLOAD=1" & shift & goto parse_args)
if "%~1"=="--" (
    shift
    set "TRAIN_ARGS=%*"
    goto args_done
)
set "TRAIN_ARGS=!TRAIN_ARGS! %~1"
shift
goto parse_args

:args_done
set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python。请先执行：
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    exit /b 1
)

echo 项目目录: %ROOT%
echo Python:   %PY%
echo.

if "%DO_DOWNLOAD%"=="1" (
    echo === 下载数据 ===
    echo.
    "%PY%" scripts\download_data.py
    if errorlevel 1 exit /b 1
    if not exist "data\online_retail.csv" (
        echo [错误] 下载后仍未找到 data\online_retail.csv
        exit /b 1
    )
    if "%DOWNLOAD_ONLY%"=="1" (
        echo.
        echo 数据已就绪。
        exit /b 0
    )
)

if "%SKIP_TRAIN%"=="0" (
    echo !TRAIN_ARGS! | findstr /i /c:"--force" >nul
    if errorlevel 1 (
        "%PY%" -c "from utils.io import all_periods_trained; import sys; sys.exit(0 if all_periods_trained() else 1)"
        if not errorlevel 1 (
            echo [提示] 已检测到完整训练产物，跳过训练。
            set "SKIP_TRAIN=1"
        )
    )
)

if "%SKIP_TRAIN%"=="0" (
    if not exist "data\online_retail.csv" (
        echo [警告] 未找到 data\online_retail.csv，训练可能失败。
        echo        可先执行: run_dev.bat --download
        echo        或: python scripts\download_data.py
        echo.
    )
    echo === 训练 (run_train.py) ===
    echo.
    if defined TRAIN_ARGS (
        "%PY%" scripts\run_train.py !TRAIN_ARGS!
    ) else (
        "%PY%" scripts\run_train.py
    )
    if errorlevel 1 exit /b 1
)

if "%TRAIN_ONLY%"=="1" (
    echo.
    echo 训练完成。
    exit /b 0
)

echo === 启动客户端 (Streamlit) ===
echo.
"%PY%" -m streamlit run app.py
exit /b %ERRORLEVEL%

:show_help
if exist "%ROOT%run_dev_help.txt" (
    type "%ROOT%run_dev_help.txt"
) else (
    echo.
    echo 用法: run_dev.bat [--skip-train] [--train-only] [--download]
    echo 详见 README.md
    echo.
)
exit /b 0

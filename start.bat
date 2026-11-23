@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE

rem Find streamlit: PATH first, then common per-user Anaconda/Miniconda installs
rem (they are not in PATH when the script is started from a desktop shortcut),
rem then fall back to "python -m streamlit".
set "RUN=streamlit run app.py"
where streamlit >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\anaconda3\Scripts\streamlit.exe" (
        set "RUN="%USERPROFILE%\anaconda3\Scripts\streamlit.exe" run app.py"
    ) else if exist "%USERPROFILE%\miniconda3\Scripts\streamlit.exe" (
        set "RUN="%USERPROFILE%\miniconda3\Scripts\streamlit.exe" run app.py"
    ) else (
        set "RUN=python -m streamlit run app.py"
    )
)

start "" http://localhost:8501
%RUN%
pause

@echo off
setlocal

echo ========================================
echo    Stockk Persona Analyzer - WebApp   
echo ========================================
echo.

:: 1. Start the Backend in a separate window
echo [1] Launching Diagnostic Backend (FastAPI)...
if exist ".venv" (
    source .venv\Scripts\activate.bat
) else if exist "venv" (
    source venv\Scripts\activate.bat
)

start cmd /k "cd webapp\backend && ..\..\.venv\Scripts\activate.bat && python api.py"

echo     (Backend starting in new window on http://localhost:8100)
echo.

:: 2. Start the Frontend in the current window
echo [2] Launching Persona Lab (Next.js)...
cd webapp\frontend
call npm run dev

pause

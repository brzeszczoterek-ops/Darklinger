@echo off
setlocal
title DARKLINGER / V-Core

wsl.exe -- bash -lc "cd ~/DARKLINGER && source .venv/bin/activate && exec darklinger-ui"

if errorlevel 1 (
    echo.
    echo DARKLINGER nie wystartowal. Sprawdz WINDOWS.md oraz sciezke ~/DARKLINGER w WSL.
    pause
)

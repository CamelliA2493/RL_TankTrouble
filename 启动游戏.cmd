@echo off
chcp 65001 >nul
cd /d "%~dp0"
"%~dp0python\python.exe" -B -X utf8 "%~dp0launch_game.py"
if errorlevel 1 pause

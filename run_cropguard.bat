@echo off
title CropGuard AI Launcher
color 0A
echo ================================================================
echo               Launching CropGuard AI System
echo ================================================================
echo.
echo Starting Streamlit server...
echo Your default web browser will open automatically at http://localhost:8501
echo.
cd /d "%~dp0"
"C:\Users\darsh\anaconda3\envs\tensorflow_env\python.exe" -m streamlit run app.py
pause

@echo off
cd /d "%~dp0"

title Local RAG System

echo Starting RAG System...
echo.

python rag_pipeline.py

pause
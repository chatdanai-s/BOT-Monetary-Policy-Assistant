@echo off
cd /d "%~dp0"

title RAG Vector Store Builder

echo Building vector database...
echo.

python rag_pipeline.py --build

pause
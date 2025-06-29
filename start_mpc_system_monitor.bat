@echo off
REM Activate the python environment and starts the Python API MCP Server

REM Project path
set PROJECT_DIR=%~dp0
cd %PROJECT_DIR%

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Start the MCP Server
python mcp_system_monitor_server.py

REM Deactivate the virtual environment in the end
call deactivate

@echo off
cd /d "%~dp0"
echo 로컬 서버를 http://localhost:8000 에서 실행합니다. (종료: Ctrl+C)
python -m http.server 8000
pause

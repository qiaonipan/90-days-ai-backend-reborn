@echo off
REM Quick test runner script for Windows

echo Running tests with coverage...
pytest --cov=. --cov-report=term-missing -v

echo.
echo Test coverage report generated!
echo For HTML report, run: pytest --cov=. --cov-report=html

pause

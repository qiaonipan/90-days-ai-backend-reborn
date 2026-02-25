#!/bin/bash
# Quick test runner script

echo "Running tests with coverage..."
pytest --cov=. --cov-report=term-missing -v

echo ""
echo "Test coverage report generated!"
echo "For HTML report, run: pytest --cov=. --cov-report=html"

#!/usr/bin/env bash
# run_tests.sh — Run all PolicyLens unit tests from the project root.
set -euo pipefail
cd "$(dirname "$0")"
python tests/test_pipeline.py

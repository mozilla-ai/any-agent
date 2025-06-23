#!/usr/bin/env bash
# Thank you to https://jaredkhan.com/blog/mypy-pre-commit for this super helpful script!
# This script is called by the pre-commit hook.
set -o errexit

# Change directory to the project root directory.
cd "$(dirname "$0")/.."

# Install the dependencies into the mypy env.
# Note that this can take seconds to run.
python -m pip install -e '.[all]' --quiet

# Run on all files. The pre-commit hook was using ignore-missing-imports, so I will keep it here for the initial implementation.
python -m mypy --ignore-missing-imports src/
python -m mypy --ignore-missing-imports tests/

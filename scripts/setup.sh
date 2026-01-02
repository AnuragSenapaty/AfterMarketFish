#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# AfterMarketFish bootstrap script:
# - create venv
# - install deps
# - run tests
# -----------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

echo "== AfterMarketFish: setup + test =="
echo "Repo: ${REPO_ROOT}"

# Pick python interpreter
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: '${PYTHON_BIN}' not found. Install Python 3 and try again."
  exit 1
fi

PY_VER="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
PY_MAJOR="$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info[0])')"
PY_MINOR="$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info[1])')"

echo "Python: ${PYTHON_BIN} (version ${PY_VER})"

# Require Python >= 3.10
if [[ "${PY_MAJOR}" -lt 3 ]] || [[ "${PY_MAJOR}" -eq 3 && "${PY_MINOR}" -lt 10 ]]; then
  echo "ERROR: Python >= 3.10 required. Found ${PY_VER}."
  exit 1
fi

# Create venv if needed
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR} ..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Virtual environment already exists: ${VENV_DIR}"
fi

# Activate venv
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "Using venv python: $(which python)"
python -m pip install --upgrade pip wheel setuptools

# Install project dependencies
REQ_FILE="${REPO_ROOT}/requirements.txt"
if [[ -f "${REQ_FILE}" ]]; then
  echo "Installing requirements from ${REQ_FILE} ..."
  python -m pip install -r "${REQ_FILE}"
else
  echo "WARNING: requirements.txt not found at ${REQ_FILE}"
  echo "         Continuing anyway (you may need to create it)."
fi

# Install test dependencies
echo "Installing test dependencies ..."
python -m pip install pytest

# Run tests
echo "Running tests ..."
cd "${REPO_ROOT}"
pytest -q

echo "All tests passed."


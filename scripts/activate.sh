#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "No .venv found. Create it with: bash scripts/setup_and_test.sh"
  return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
echo "Activated venv: ${VENV_DIR}"
echo "Python: $(which python)"

python run_lichess_bot.py --ckpt models/ffnn.pt --vocab models/vocab.json
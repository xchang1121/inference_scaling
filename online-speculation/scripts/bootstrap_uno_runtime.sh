#!/usr/bin/env bash
set -euo pipefail

readonly UNO_SOURCE="${1:?Pass the mounted Uno source repository path}"
readonly PROJECT_SOURCE="${2:?Pass the mounted online-speculation path}"
readonly BASE_SOURCE="${3:?Pass the mounted Uno-1B base model path}"
readonly ADAPTER_SOURCE="${4:?Pass the mounted Uno-1B adapter path}"
readonly RESULT_PATH="${5:?Pass the mounted runtime-result JSON path}"

readonly UNO_COMMIT="ed2ee36bb7a3aea8732ebc635b3f09490a032ea3"
readonly BASE_SHA256="6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365"
readonly ADAPTER_SHA256="5a499229d19ef4a69eb0b21884819d1b67cd983ba02b7ee2031ba8567dedfe4e"
readonly EXPECTED_FA_SHA256="9001c730642cdc1ea44ed8130b0dc80e763519d6efc01e4de44b0700a0dfa13d"
readonly EXPECTED_FA_BYTES="253651546"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run the Python runtime bootstrap as the non-root WSL user" >&2
  exit 1
fi

for required in "${UNO_SOURCE}" "${PROJECT_SOURCE}" "${BASE_SOURCE}" "${ADAPTER_SOURCE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required source does not exist: ${required}" >&2
    exit 1
  fi
done

readonly WORK_ROOT="${HOME}/online-speculation-work"
readonly UNO_DIR="${WORK_ROOT}/uno"
readonly MODEL_ROOT="${WORK_ROOT}/models"
readonly BASE_DIR="${MODEL_ROOT}/K2-Horizon-0.9B"
readonly ADAPTER_DIR="${MODEL_ROOT}/K2-Horizon-0.9B-Uno"
readonly VENV_DIR="${HOME}/.venvs/uno-cu128"
readonly WHEEL_DIR="${HOME}/.cache/uno-wheels"

mkdir -p "${WORK_ROOT}" "${MODEL_ROOT}" "${WHEEL_DIR}" "${HOME}/.venvs"

if [[ ! -e "${UNO_DIR}" ]]; then
  git clone --no-hardlinks "${UNO_SOURCE}" "${UNO_DIR}"
elif [[ ! -d "${UNO_DIR}/.git" ]]; then
  echo "Existing Uno target is not a Git worktree: ${UNO_DIR}" >&2
  exit 1
fi

if [[ -n "$(git -C "${UNO_DIR}" status --porcelain)" ]]; then
  echo "Refusing to replace a dirty WSL Uno worktree" >&2
  exit 1
fi
git -C "${UNO_DIR}" checkout --detach "${UNO_COMMIT}"
if [[ "$(git -C "${UNO_DIR}" rev-parse HEAD)" != "${UNO_COMMIT}" ]]; then
  echo "Uno source revision mismatch" >&2
  exit 1
fi

mkdir -p "${BASE_DIR}" "${ADAPTER_DIR}"
rsync --archive --exclude .git "${BASE_SOURCE}/" "${BASE_DIR}/"
rsync --archive --exclude .git "${ADAPTER_SOURCE}/" "${ADAPTER_DIR}/"

actual_base_sha256="$(sha256sum "${BASE_DIR}/model-00000-of-00001.safetensors" | cut -d ' ' -f 1)"
actual_adapter_sha256="$(sha256sum "${ADAPTER_DIR}/adapter_model.safetensors" | cut -d ' ' -f 1)"
if [[ "${actual_base_sha256}" != "${BASE_SHA256}" ]]; then
  echo "Uno-1B base model SHA-256 mismatch" >&2
  exit 1
fi
if [[ "${actual_adapter_sha256}" != "${ADAPTER_SHA256}" ]]; then
  echo "Uno-1B adapter SHA-256 mismatch" >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3.10 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Permanent reproducible entry point. One-off range/tunnel helpers were removed
# after installation. Existing verified wheels support offline reuse.
fetch_locked_wheel() {
  local lock_file="${1}" filename expected_bytes expected_sha download_url wheel_path
  filename="$(jq --exit-status --raw-output .filename "${lock_file}")"
  expected_bytes="$(jq --exit-status --raw-output .bytes "${lock_file}")"
  expected_sha="$(jq --exit-status --raw-output .sha256 "${lock_file}")"
  download_url="$(jq --exit-status --raw-output .url "${lock_file}")"
  if [[ "${filename}" != "$(basename "${filename}")" || "${download_url}" != https://* ]]; then
    echo "Invalid locked filename or HTTPS source" >&2
    return 1
  fi
  wheel_path="${WHEEL_DIR}/${filename}"
  if [[ ! -e "${wheel_path}" ]]; then
    curl --fail --location --proto '=https' --proto-redir '=https' \
      --retry 4 --retry-all-errors --connect-timeout 15 --speed-time 60 --speed-limit 1024 \
      --continue-at - --output "${wheel_path}.download.part" "${download_url}"
    if [[ "$(stat --format=%s "${wheel_path}.download.part")" != "${expected_bytes}" || \
          "$(sha256sum "${wheel_path}.download.part" | cut -d ' ' -f 1)" != "${expected_sha}" ]]; then
      echo "Downloaded wheel does not match lock; preserved, not installed" >&2
      return 1
    fi
    mv --no-clobber -- "${wheel_path}.download.part" "${wheel_path}"
  fi
  if [[ "$(stat --format=%s "${wheel_path}")" != "${expected_bytes}" || \
        "$(sha256sum "${wheel_path}" | cut -d ' ' -f 1)" != "${expected_sha}" ]]; then
    echo "Saved wheel does not match lock; preserved, not installed" >&2
    return 1
  fi
  echo "VERIFIED ${filename} ${expected_sha}"
}

python -m pip install --upgrade pip 'setuptools<82' wheel
fetch_locked_wheel "${PROJECT_SOURCE}/config/torch_wheel.lock.json"
fetch_locked_wheel "${PROJECT_SOURCE}/config/triton_wheel.lock.json"
python -m pip install "${WHEEL_DIR}/torch-2.11.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl" \
  "${WHEEL_DIR}/triton-3.6.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" \
  --find-links "${WHEEL_DIR}" --index-url https://download.pytorch.org/whl/cu128

mapfile -t flash_wheels < <(find "${WHEEL_DIR}" -maxdepth 1 -type f \
  -name 'flash_attn-2.8.3*cp310*linux_x86_64.whl' -print)
if [[ "${#flash_wheels[@]}" -eq 0 ]]; then
  fetch_locked_wheel "${PROJECT_SOURCE}/config/flash_attn_wheel.lock.json"
  mapfile -t flash_wheels < <(find "${WHEEL_DIR}" -maxdepth 1 -type f \
    -name 'flash_attn-2.8.3*cp310*linux_x86_64.whl' -print)
fi
if [[ "${#flash_wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one pinned FlashAttention wheel" >&2
  exit 1
fi
readonly FLASH_WHEEL="${flash_wheels[0]}"
readonly FLASH_WHEEL_SHA256="$(sha256sum "${FLASH_WHEEL}" | cut -d ' ' -f 1)"
if [[ "${FLASH_WHEEL_SHA256}" != "${EXPECTED_FA_SHA256}" || "$(stat --format=%s "${FLASH_WHEEL}")" != "${EXPECTED_FA_BYTES}" ]]; then
  echo "Pinned FlashAttention release asset failed SHA-256 or length verification; refusing installation" >&2
  exit 1
fi

python -m pip install "${FLASH_WHEEL}"
python -m pip install --editable "${UNO_DIR}"
python -m pip check

python "${PROJECT_SOURCE}/scripts/wsl_runtime_smoke.py" \
  --output "${RESULT_PATH}" \
  --uno-source "${UNO_DIR}" \
  --base-model "${BASE_DIR}" \
  --adapter "${ADAPTER_DIR}" \
  --flash-wheel-sha256 "${FLASH_WHEEL_SHA256}"

echo "Uno WSL runtime bootstrap completed: ${VENV_DIR}"

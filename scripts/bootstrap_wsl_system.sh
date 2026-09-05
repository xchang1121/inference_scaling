#!/usr/bin/env bash
set -euo pipefail

readonly LINUX_USER="${1:-singm}"
readonly WSL_CONFIG_SOURCE="${2:?Pass the mounted path to config/wsl.conf}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "bootstrap_wsl_system.sh must run as root" >&2
  exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "Uno requires Linux x86_64" >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID}" != "ubuntu" || "${VERSION_ID}" != "22.04" ]]; then
  echo "Expected Ubuntu 22.04, found ${PRETTY_NAME}" >&2
  exit 1
fi

if [[ ! -f "${WSL_CONFIG_SOURCE}" ]]; then
  echo "Missing WSL configuration: ${WSL_CONFIG_SOURCE}" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
# Keep the original Ubuntu source list and only upgrade its official URLs.
# HTTP archive access timed out on this host; both HTTPS endpoints were tested.
if [[ ! -e /etc/apt/sources.list.online-uno-original ]]; then
  cp --preserve=mode,timestamps /etc/apt/sources.list /etc/apt/sources.list.online-uno-original
fi
sed -i \
  -e 's|http://archive.ubuntu.com/ubuntu/|https://archive.ubuntu.com/ubuntu/|g' \
  -e 's|http://security.ubuntu.com/ubuntu/|https://security.ubuntu.com/ubuntu/|g' \
  /etc/apt/sources.list
apt-get --error-on=any update
apt-get install --yes --no-install-recommends \
  build-essential \
  ca-certificates \
  cmake \
  curl \
  git \
  jq \
  ninja-build \
  pkg-config \
  python3.10-dev \
  python3.10-venv \
  rsync

if ! id -u "${LINUX_USER}" >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash "${LINUX_USER}"
fi

install --mode 0644 "${WSL_CONFIG_SOURCE}" /etc/wsl.conf
install --directory --owner "${LINUX_USER}" --group "${LINUX_USER}" \
  "/home/${LINUX_USER}/online-speculation-work"

echo "WSL system bootstrap completed for ${LINUX_USER}."

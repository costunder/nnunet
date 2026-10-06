#!/usr/bin/env bash
set -euo pipefail
VERSION_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$VERSION_REPO"
export CP_ARM=D
bash tools/server_v17_crossed.sh "$@"

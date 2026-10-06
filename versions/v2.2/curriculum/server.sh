#!/usr/bin/env bash
set -euo pipefail
VERSION_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$VERSION_REPO"
bash tools/server_v22_cumulative_u16.sh "$@"

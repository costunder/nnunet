#!/usr/bin/env bash
# Environment-only navigation wrapper; the shared launcher owns execution.
if [[ $# -ne 0 ]]; then
  printf '%s\n' 'This wrapper accepts no positional arguments; set CP_GPU and existing CP_* environment variables.' >&2
  false
elif [[ -n "${CP_ARM:-}" && "${CP_ARM}" != "native" ]]; then
  printf '%s\n' 'Conflicting CP_ARM: this wrapper is fixed to native.' >&2
  false
elif [[ -z "${CP_GPU:-}" ]]; then
  printf '%s\n' 'Set CP_GPU to the physical GPU allocated to this arm before running.' >&2
  false
else
  if arm_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"; then
    CP_ARM="native" CP_GPU="$CP_GPU" bash "$arm_repo_root/tools/server_comparison_cached.sh"
  else
    printf '%s\n' 'Cannot locate the repository containing this wrapper.' >&2
    false
  fi
fi

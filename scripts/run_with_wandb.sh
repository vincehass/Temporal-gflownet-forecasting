#!/bin/bash
# Wrapper script to ensure W&B is enabled for all commands
#
# Usage: ./scripts/run_with_wandb.sh [original command and args]
#
# Example: ./scripts/run_with_wandb.sh python scripts/train.py --config-name base_config
#
# This script will ensure that W&B is always enabled with appropriate settings

set -e

# Script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Make ensure_wandb.py executable if it isn't already
chmod +x "$SCRIPT_DIR/ensure_wandb.py" 2>/dev/null || true

# Parse any wandb-specific args
WANDB_ARGS=()
CMD_ARGS=()
SET_DEFAULTS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wandb-entity=*)
      WANDB_ARGS+=("--wandb-entity" "${1#*=}")
      shift
      ;;
    --wandb-project=*)
      WANDB_ARGS+=("--wandb-project" "${1#*=}")
      shift
      ;;
    --wandb-name=*)
      WANDB_ARGS+=("--wandb-name" "${1#*=}")
      shift
      ;;
    --offline)
      WANDB_ARGS+=("--offline")
      shift
      ;;
    --disable-wandb)
      WANDB_ARGS+=("--disable-wandb")
      shift
      ;;
    --set-defaults)
      SET_DEFAULTS=true
      WANDB_ARGS+=("--set-defaults")
      shift
      ;;
    *)
      CMD_ARGS+=("$1")
      shift
      ;;
  esac
done

# If no command is provided, show usage
if [ ${#CMD_ARGS[@]} -eq 0 ]; then
  echo "Usage: ./scripts/run_with_wandb.sh [command and args]"
  echo ""
  echo "W&B options:"
  echo "  --wandb-entity=NAME     W&B entity name"
  echo "  --wandb-project=NAME    W&B project name"
  echo "  --wandb-name=NAME       W&B run name"
  echo "  --offline               Use W&B in offline mode"
  echo "  --disable-wandb         Don't use W&B for this run"
  echo "  --set-defaults          Save provided W&B settings as defaults"
  echo ""
  echo "Example:"
  echo "  ./scripts/run_with_wandb.sh --wandb-project=my-project python scripts/train.py"
  exit 1
fi

# Only set defaults if requested without running a command
if [ "$SET_DEFAULTS" = true ] && [ ${#CMD_ARGS[@]} -eq 0 ]; then
  python "$SCRIPT_DIR/ensure_wandb.py" "${WANDB_ARGS[@]}" echo "Setting defaults only"
  exit $?
fi

# Run the command with wandb enabled
python "$SCRIPT_DIR/ensure_wandb.py" "${WANDB_ARGS[@]}" "${CMD_ARGS[@]}" 
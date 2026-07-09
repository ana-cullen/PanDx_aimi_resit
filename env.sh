SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/venv/bin/activate"

export nnUNet_raw="$SCRIPT_DIR/workspace/workspace/nnUNet_raw"
export nnUNet_preprocessed="$SCRIPT_DIR/workspace/workspace/nnUNet_preprocessed"
export nnUNet_results="$SCRIPT_DIR/workspace/workspace/nnUNet_results"

SCRIPT_DIR="/vol/csedu-nobackup/course/IMC037_aimi/group09/resit"

source "/vol/csedu-nobackup/course/IMC037_aimi/group09/resit/venv2/bin/activate"

export PYTHONPATH="$(dirname "${BASH_SOURCE[0]}"):$PYTHONPATH"

export nnUNet_raw="$SCRIPT_DIR/nnUNet_raw"
export nnUNet_preprocessed="$SCRIPT_DIR/nnUNet_preprocessed"
export nnUNet_results="$SCRIPT_DIR/nnUNet_results"

export SABN_COND_MAP_PATH="$SCRIPT_DIR/PanDx_aimi_resit/patient_sex_map.json"

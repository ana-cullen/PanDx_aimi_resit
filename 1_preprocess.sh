source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

nnUNetv2_extract_fingerprint -d 101 --verify_dataset_integrity
nnUNetv2_plan_experiment -d 101
nnUNetv2_preprocess -d 101 -c 2d -np 2
nnUNetv2_preprocess -d 101 -c 3d_fullres -np 2

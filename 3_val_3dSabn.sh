source "$(dirname "${BASH_SOURCE[0]}")/env.sh"


#for FOLD in 0 1 2 3 4; do
for FOLD in 0; do
    CUDA_LAUNCH_BLOCKING=1 nnUNetv2_train 101 3d_fullres $FOLD -tr nnUNetTrainerCELossLesionSplitSaBN --npz -p resEncUNetPlansSabn -num_gpus 2 --val
done

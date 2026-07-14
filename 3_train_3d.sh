source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

for FOLD in 0 1 2 3 4; do
    nnUNetv2_train 101 3d_fullres $FOLD -tr nnUNetTrainerCELossLesionSplit -p nnUNetPlans_v3 --npz
done

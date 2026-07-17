source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# Stage 2 of the PanDx paper (arXiv:2503.10068, Sec 2.1): fine-scale segmentation
# of the 6 PDAC-related structures on the cropped high-res ROI.
#   - ResU-Net backbone -> nnUNetPlans_v3 (generated via ResEncUNetPlanner)
#   - CE-only loss, DASE lesion-size-stratified folds -> nnUNetTrainerCELossLesionSplit
#     (picks up nnUNet_preprocessed/Dataset101_PDAC/splits_final.json automatically)
#   - 5-fold ensemble: final prediction averages softmax across all 5 folds
for FOLD in 0 1 2 3 4; do
    nnUNetv2_train 101 3d_fullres $FOLD -tr nnUNetTrainerCELossLesionSplitSaBN -p nnUNetPlans_v3 --npz --c
done 
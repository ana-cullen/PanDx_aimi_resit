source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# NOTE: the PanDx paper (arXiv:2503.10068) does not describe a 2D training
# stage at all -- Stage 1 reuses a pretrained low-res baseline checkpoint and
# Stage 2 is 3D-only (see 3_train_3d.sh for the faithful reproduction). This
# script is kept only as an extra nnU-Net configuration, using the same
# CE-loss/DASE-split trainer as Stage 2 for consistency, on the default plans.
nnUNetv2_train 101 2d 0 -tr nnUNetTrainerCELossLesionSplit --npz --c 
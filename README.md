# PanDx: AI-assisted Pancreatic Ductal Adenocarcinoma Detection 
[![arXiv](https://img.shields.io/badge/preprint-2503.10068-blue)](https://arxiv.org/abs/2503.10068) [![cite](https://img.shields.io/badge/cite-BibTex-red)](xx) [![leaderboard](https://img.shields.io/badge/Leaderboard-yellow)](https://panorama.grand-challenge.org/evaluation/testing-phase/leaderboard/) [![website](https://img.shields.io/badge/Challenge%20website-50d13d)](https://panorama.grand-challenge.org/)

# Sandwich BatchNorm extension of PanDx
This repo is built on off of the PanDx repo with an additional module to train with SandwichBatchNorm (SaBN) instead of regular InstanceNorm or BatchNorm and minor changes to the original nnunetv2 package.

## SaBN folder
This folder contains a custom nnUNetTrainerSaBN.py that inherits from nnunetv2's nnUNetTrainer class with additional logic for passing the clinical information through the network for SaBN. It also contains a
SaBNPredictor that passes conditional information for inference and
an entry point for this predictor in predict_sabn.py

The SaResNet folder contains the actual SandwichBatchNorm module (SimpleSaBN.py) as well as a modified version of the ResUNet used in the PanDx network that passes the conditional clinical information through the network. Additionally there are some helper functions to set up the network in helpers.py

## Changes to nnunetv2
Minor changes were made to PanDx's nnunetv2 package to integrate our SaBN implementation

PanDx_aimi_resit/packages/nnunetv2/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerCELoss.py
- Added nnUNetTrainerCELossLesionSplitSaBN class that uses PanDx's CE Loss as the loss function but inherits our custom nnUNetTrainerSaBN class.

PanDx_aimi_resit/packages/nnunetv2/nnunetv2/run/run_training.py
- Commented out nnUNetTrainer subclass assertion from get_trainer_from_args() 

PanDx_aimi_resit/packages/nnunetv2/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
-  Uncommented self.batch_size = batch_sizes[my_rank] to allow the use of multiple GPUs
-  Final validation uses non-ddp wrapped module to avoid crashing on multiple GPUs

## Conditional Clinical information
The clinical information to be used as the conditional input should be supplied in a json file of case_id -> int mappings with ints contiguous starting at 0. Conditional clinical information should be int >= 1. Cases with unknown information should be mapped to 0 as this will skip the independent affine transformation and only run regular batchnorm

patient_sex_map.json is an example file. clinical.py has basic code to create a mapping json from a csv file with clinical information (hardcoded to patient sex)

### Training
Before training set SABN_COND_MAP_PATH as an environment variable that points to the pre-computed conditional mapping. 
 
The nnUNetTrainerSaBN class sets it's COND_MAP_PATH to the SABN_COND_MAP_PATH path on import.

### Inference
SaBNPredictor can be given a new path to a conditonal map (via --cond-map-path arg) for standalone inference. If this isn't supplied, the nnUNetTrainerSaBN will fall back on the path set as an environment variable and will set the condition of any keys not present in the map to 0/Unknown.

### Plans
When the conditonal map is loaded in by a nnUNetTrainerSaBN trainer instance it derives the number of conditons from the SABN_COND_MAP_PATH and saves the number of conditions as an additonal key in the plans files (e.g. "num_conditions": 3).

During inference the network architecture is reconstructed using the number of conditions given in the plans file to make the conditional embedding.


## Training + Plans
resEncUNetPlansSabn.json is an example plans file that sets the correct UNet class to run with SaBN

To train a network with SaBN:
1. Create the clinical information map json file and set SABN_COND_MAP_PATH as an environment variable to point to it.
2. Run roi.py on full scan data (PanDx Stage 1)
3. Preprocess ROIs extracted in previous step:
```
nnUNetv2_preprocess -d 101 -c 3d_fullres -np 2 -plans_name resEncUNetPlansSabn
```
4. Make training/validation split by running make_dase_splits.py
5. Train network: 
```
for FOLD in 0 1 2 3 4; do
    CUDA_LAUNCH_BLOCKING=1 nnUNetv2_train 101 3d_fullres $FOLD -tr nnUNetTrainerCELossLesionSplitSaBN --npz --c -p resEncUNetPlansSabn -num_gpus 2
done
```

## Inference
Use the SaBN.predict_sabn entrypoint for inference. It takes the same args as nnUNetv2_predict as well as an additional --cond-map-path arg that should point to the condtional mapping for inference cases.

```
python -m SaBN.predict_sabn --cond-map-path [inference_cond_map] -d [task_id] -i [inout_dir] -o [output_dir] -tr nnUNetTrainerCELossLesionSplitSaBN -p resEncUNetPlansSabn --continue_prediction -f 0 1 2 3 4 -chk checkpoint_final.pth --save_probabilities 
```

main.py runs the full pipeline and computes the condtional map for patient_sex using the "clinical-information-pancreatic-ct.json" supplied in the input directory.

#nnUNetv2_plan_experiment -d 101
#nnUNetv2_preprocess -d 101 -c 2d -np 2
#nnUNetv2_preprocess -d 101 -c 3d_fullres -np 2 --num_processes 8
# nnUNet_resume_preprocessing=1 nnUNetv2_preprocess -d 101 -c 3d_fullres -np 2 --num_processes 8


# nnUNetv2_extract_fingerprint -d 101 --verify_dataset_integrity
# nnUNetv2_plan_experiment -d 101 -pl ResEncUNetPlanner -overwrite_plans_name resEncUNetPlans
nnUNetv2_preprocess -d 101 -c 3d_fullres -np 2 --num_processes 8 -plans_name resEncUNetPlans --verbose
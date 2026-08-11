import subprocess
import shutil
import os
import json
from glob import glob
import SimpleITK as sitk

INPUT_IMAGES_DIR = "/input/images/venous-ct"
INPUT_CLINICAL_JSON = "/input/clinical-information-pancreatic-ct.json"
STAGING_DIR = "/opt/app/staged_input"
OUTPUT_LIKELIHOOD = "/output/pdac-likelihood.json"
OUTPUT_MAP_DIR = "/output/images/pdac-detection-map"
NNUNET_RESULTS_DIR = "/opt/algorithm/nnunet/nnUNet_results"

os.makedirs(OUTPUT_MAP_DIR, exist_ok=True)
os.makedirs(STAGING_DIR, exist_ok=True)

for f in glob(os.path.join(INPUT_IMAGES_DIR, "*")):
    os.symlink(f, os.path.join(STAGING_DIR, os.path.basename(f)))
os.symlink(INPUT_CLINICAL_JSON, os.path.join(STAGING_DIR, os.path.basename(INPUT_CLINICAL_JSON)))


subprocess.run([
    "python3", "main.py",
    "-i", STAGING_DIR,
    "-o", "/output/images",
    "-m", NNUNET_RESULTS_DIR
], check=True)

produced = glob(os.path.join(OUTPUT_MAP_DIR, "*.nii.gz"))
assert len(produced) == 1, f"expected exactly one detection map, got {produced}"
uuid_stem = os.path.splitext(os.path.basename(produced[0]))[0]
sitk.WriteImage(sitk.ReadImage(produced[0]), os.path.join(OUTPUT_MAP_DIR, f"{uuid_stem}.mha"))
os.remove(produced[0])


with open(os.path.join("/output/images", "pdac-likelihood.json")) as f:
    scores = json.load(f)
assert len(scores) == 1, f"expected exactly one likelihood score, got {scores}"
with open(OUTPUT_LIKELIHOOD, "w") as f:
    json.dump(list(scores.values())[0], f)

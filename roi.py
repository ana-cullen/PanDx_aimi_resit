import json
import os
import os.path as osp
from glob import glob

import SimpleITK as sitk
from tqdm import tqdm

from main import crop_roi, downsample_panorama_dataset, predict

BASE = "/vol/csedu-nobackup/course/IMC037_aimi/group09/resit"
RAW = BASE + "/nnUNet_raw"
RESULTS = BASE + "/nnUNet_results"

SRC = RAW + "/Dataset100_PDAC_full"
DST = RAW + "/Dataset101_PDAC"
WORK = BASE + "/itm"

LOCALIZER_TASK = 103
MARGINS = [100, 50, 15]

def main():
    os.makedirs(DST + "/imagesTr", exist_ok=True)
    os.makedirs(DST + "/labelsTr", exist_ok=True)
    os.makedirs(WORK + "/Staged", exist_ok=True)

    for image in sorted(glob(SRC + "/imagesTr/*_0000.nii.gz")):
        case_id = osp.basename(image)[: -len("_0000.nii.gz")]
        link = WORK + "/Staged/" + case_id + ".nii.gz"
        if not osp.exists(link):
            os.symlink(image, link)

    print("1. downsample")
    downsample_panorama_dataset(WORK + "/Staged", WORK + "/LowImagesTr")

    print("2. pretrained model")

    predict(
        nnunet_model_dir=RESULTS,
        input_dir=WORK + "/LowImagesTr",
        output_dir=WORK + "/LowPred",
        task=LOCALIZER_TASK,
        tta=False)

   
    skipped = []
    for link in sorted(glob(WORK + "/Staged/*.nii.gz")):
        case_id = osp.basename(link)[: -len(".nii.gz")]
        msk = sitk.GetArrayFromImage(sitk.ReadImage(WORK + "/LowPred/" + case_id + ".nii.gz"))
        if (msk == 1).sum() == 0:
            skipped.append(case_id)
            os.remove(link)
    print("skipped " + str(len(skipped)) + " cases with an empty pancreas mask: " + str(skipped))

    print("3. crop roi")
    coords = crop_roi(WORK + "/Staged", WORK + "/LowPred", DST + "/imagesTr", margins=MARGINS)

    new_image = open(DST + "/crop_coordinates.json", "w")
    new_image.write(json.dumps(coords, indent=2))
    new_image.close()

    print("4. crop labels")
    for id in tqdm(coords):
        c = coords[id]
        lbl = sitk.ReadImage(SRC + "/labelsTr/" + id + ".nii.gz")
        cropped = lbl[c["x_start"]:c["x_finish"], c["y_start"]:c["y_finish"], c["z_start"]:c["z_finish"]]
        sitk.WriteImage(cropped, DST + "/labelsTr/" + id + ".nii.gz")

    dataset_json = json.load(open(SRC + "/dataset.json"))
    dataset_json["numTraining"] = len(coords)
    json.dump(dataset_json, open(DST + "/dataset.json", "w"), indent=4)

    print("")
    print("done, " + str(len(coords)) + " cases written to " + DST)


if __name__ == "__main__":
    main()

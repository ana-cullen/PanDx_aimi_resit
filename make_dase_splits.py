#Same split as PanDx paper (arXiv:2503.10068), but with 5 folds instead of 4
import glob
import os

import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_json

PDAC_LABEL = 1
N_FOLDS = 4
N_QUARTILE_BINS = 4

RAW_LABELS_DIR = "/vol/csedu-nobackup/course/IMC037_aimi/group09/resit/nnUNet_raw/Dataset101_PDAC/labelsTr"
PREPROCESSED_DIR = "/vol/csedu-nobackup/course/IMC037_aimi/group09/resit/nnUNet_preprocessed/Dataset101_PDAC"


def lesion_voxel_count(label_path: str) -> int:
    data = nib.load(label_path).get_fdata()
    return int(np.sum(data == PDAC_LABEL))


def main():
    label_files = sorted(glob.glob(os.path.join(RAW_LABELS_DIR, "*.nii.gz")))
    case_ids = [os.path.basename(f)[: -len(".nii.gz")] for f in label_files]

    lesion_sizes = {cid: lesion_voxel_count(f) for cid, f in zip(case_ids, label_files)}
    positive_ids = [cid for cid, sz in lesion_sizes.items() if sz > 0]
    negative_ids = [cid for cid, sz in lesion_sizes.items() if sz == 0]

    print(f"{len(positive_ids)} PDAC-positive cases, {len(negative_ids)} PDAC-negative cases")

    # bin PDAC-positive cases into quartiles by lesion size
    positive_sorted = sorted(positive_ids, key=lambda cid: lesion_sizes[cid])
    bins = np.array_split(positive_sorted, min(N_QUARTILE_BINS, len(positive_sorted)))
    for i, b in enumerate(bins):
        print(f"  size bin {i}: {list(b)} "
              f"(sizes: {[lesion_sizes[c] for c in b]})")

    # distribute each quartile bin round-robin across folds
    fold_val_positive = [[] for _ in range(N_FOLDS)]
    for b in bins:
        for i, cid in enumerate(b):
            fold_val_positive[i % N_FOLDS].append(cid)

    # distribute negatives round-robin to preserve the global PDAC ratio per fold
    fold_val_negative = [[] for _ in range(N_FOLDS)]
    for i, cid in enumerate(sorted(negative_ids)):
        fold_val_negative[i % N_FOLDS].append(cid)

    all_ids = set(case_ids)
    splits = []
    for f in range(N_FOLDS):
        val_keys = sorted(fold_val_positive[f] + fold_val_negative[f])
        train_keys = sorted(all_ids - set(val_keys))
        splits.append({"train": train_keys, "val": val_keys})
        print(f"fold {f}: train={len(train_keys)}, val={len(val_keys)} "
              f"(val positives={len(fold_val_positive[f])})")

    out_path = os.path.join(PREPROCESSED_DIR, "splits_final.json")
    save_json(splits, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

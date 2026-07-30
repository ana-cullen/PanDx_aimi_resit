"""Standalone inference entry point for SaBN-conditioned trainers, mirroring
nnUNetv2_predict (nnunetv2.inference.predict_from_raw_data.predict_entry_point)
but using SaBNPredictor so the network's required `cond` argument gets set
per case instead of crashing with a missing-argument TypeError.

Usage matches nnUNetv2_predict for the flags it supports, plus:
    --cond-map-path PATH   case_id -> cond_id JSON for cases being predicted
                            on right now (they won't be in the trainer's
                            training-time COND_MAP_PATH). Omit to fall back
                            to that training-time map (cond=0/"unknown" for
                            any case not in it).
"""
import argparse

import torch

from nnunetv2.utilities.file_path_utilities import get_output_folder
from batchgenerators.utilities.file_and_folder_operations import isdir, maybe_mkdir_p

from .SaBNPredictor import SaBNPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', type=str, required=True, help='input folder')
    parser.add_argument('-o', type=str, required=True, help='output folder')
    parser.add_argument('-d', type=str, required=True, help='dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans', help='plans identifier')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainerSaBN', help='trainer class used for training')
    parser.add_argument('-c', type=str, required=True, help='configuration')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4), help='folds to use')
    parser.add_argument('-step_size', type=float, required=False, default=0.5)
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_probabilities', action='store_true')
    parser.add_argument('--continue_prediction', action='store_true')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3)
    parser.add_argument('-nps', type=int, required=False, default=3)
    parser.add_argument('-num_parts', type=int, required=False, default=1)
    parser.add_argument('-part_id', type=int, required=False, default=0)
    parser.add_argument('-device', type=str, default='cuda', required=False)
    parser.add_argument('--cond-map-path', type=str, required=False, default=None,
                        help='case_id -> cond_id JSON for the cases being predicted on now')
    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.part_id < args.num_parts
    assert args.device in ('cpu', 'cuda', 'mps')
    if args.device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = SaBNPredictor(tile_step_size=args.step_size,
                              use_gaussian=True,
                              use_mirroring=not args.disable_tta,
                              perform_everything_on_gpu=True,
                              device=device,
                              verbose=args.verbose,
                              verbose_preprocessing=False,
                              cond_map_path=args.cond_map_path)
    predictor.initialize_from_trained_model_folder(model_folder, args.f, checkpoint_name=args.chk)

    predictor.predict_from_files(
        args.i, args.o, save_probabilities=args.save_probabilities,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
        num_parts=args.num_parts,
        part_id=args.part_id,
    )


if __name__ == "__main__":
    main()

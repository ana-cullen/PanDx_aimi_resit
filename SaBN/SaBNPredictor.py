import multiprocessing
import os
from time import sleep
from typing import Dict, Optional

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.file_path_utilities import check_workers_busy
from nnunetv2.utilities.helpers import empty_cache

from .nnUNetTrainerSaBN import _CondAdapter, cond_id_for_case, load_cond_map, nnUNetTrainerSaBN


class SaBNPredictor(nnUNetPredictor):
    """nnUNetPredictor for SaBN-conditioned trainers (nnUNetTrainerSaBN and
    subclasses). The stock nnUNetPredictor calls network(x) with no
    knowledge of `cond`, which is a required forward() argument for these
    networks -- see nnUNetTrainerSaBN.perform_actual_validation for the
    same problem during training-time validation.

    cond_map_path defaults to the trainer's own COND_MAP_PATH (the map used
    during training), but standalone inference is normally run on cases
    that aren't in that map -- e.g. main.py's per-patient pipeline knows a
    single patient's clinical info at run time and has no reason to be in
    the static training-time map. Pass cond_map_path to point at a JSON
    built for this run instead (same case_id -> cond_id format).
    """

    def __init__(self, *args, cond_map_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cond_map_path = cond_map_path
        self.cond_map: Dict[str, int] = {}

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds, checkpoint_name: str = 'checkpoint_final.pth'):
        super().initialize_from_trained_model_folder(model_training_output_dir, use_folds, checkpoint_name)
        cond_map_path = self._cond_map_path if self._cond_map_path is not None else nnUNetTrainerSaBN.COND_MAP_PATH
        self.cond_map = load_cond_map(cond_map_path)
        self.network = _CondAdapter(self.network)

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes,
                                   *args, **kwargs):
        """Copy of nnUNetPredictor.predict_from_data_iterator with cond set
        on self.network (a _CondAdapter) per case, derived from that case's
        output filename -- the same identifier used for the map keys."""
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:

            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                properties = preprocessed['data_properites']

                case_id = os.path.basename(ofile) if ofile is not None else None
                cond_id = cond_id_for_case(case_id, self.cond_map) if case_id is not None else 0
                self.network.cond = torch.tensor([cond_id], dtype=torch.long, device=self.device)

                proceed = not check_workers_busy(export_pool, r, allowed_num_queued=2 * len(export_pool._pool))
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_busy(export_pool, r, allowed_num_queued=2 * len(export_pool._pool))

                prediction = self.predict_logits_from_preprocessed_data(data, *args, **kwargs).cpu()

                if ofile is not None:
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        compute_gaussian.cache_clear()
        empty_cache(self.device)
        return ret

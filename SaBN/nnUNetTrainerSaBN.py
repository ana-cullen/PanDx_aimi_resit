import json
from typing import Dict

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from .SaResNet.helpers import get_network_from_plans
from torch import autocast, nn
import torch
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


class nnUNetTrainerSaBN(nnUNetTrainer):

    # COND_MAP_PATH should point to a json file of case_id -> int mappings
    # with ints contiguous starting at 0. Conditional clinical information 
    # should be int >= 1. Cases with unknown information should be mapped 
    # to 0 as this will skip the sandwich affine layer and only run regular
    # batchnorm
    #
    #   {"PANORAMA_0001": 0, "PANORAMA_0002": 1, "PANORAMA_0350": 0, ...}

    COND_MAP_PATH = "/vol/csedu-nobackup/course/IMC037_aimi/group09/resit/PanDx_aimi_resit/patient_sex_map.json"

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.cond_map: Dict[str, int] = self._load_cond_map()
        self.num_conditions = (max(self.cond_map.values()) + 1) if self.cond_map else 1

    def _load_cond_map(self) -> Dict[str, int]:
        if self.COND_MAP_PATH is None:
            self.print_to_log_file(
                "WARNING: COND_MAP_PATH is not set. All cases will be "
                "treated as a single condition (num_conditions=1), which "
                "makes SaBN degenerate to plain BatchNorm. Set the var "
                "to your case_id -> cond_id JSON before training."
            )
            return {}
        with open(self.COND_MAP_PATH, 'r') as f:
            return json.load(f)

    def _cond_ids_for_keys(self, keys) -> torch.Tensor:
        ids = []
        for k in keys:
            if k in self.cond_map:
                ids.append(self.cond_map[k])
            else:
                # prefix match, in case nnU-Net appended a patch/crop suffix
                matched = next((v for ck, v in self.cond_map.items() if k.startswith(ck)), None)
                ids.append(matched if matched is not None else 0)
        return torch.tensor(ids, dtype=torch.long, device=self.device)


    def build_network_architecture(self, plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return get_network_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=enable_deep_supervision,
            num_conditions=self.num_conditions,
        )

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        cond = self._cond_ids_for_keys(batch['keys'])

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, cond)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        cond = self._cond_ids_for_keys(batch['keys'])

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, cond)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0]

        axes = [0] + list(range(2, len(output.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
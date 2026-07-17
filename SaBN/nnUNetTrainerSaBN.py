import json
import os
from typing import Dict

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from torch import autocast, nn
import torch
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss


class nnUNetTrainerSaBN(nnUNetTrainer):

    # ------------------------------------------------------------------
    # A case_id -> cond_id map has to come from somewhere. There's no field
    # like this anywhere in the base nnUNetTrainer/dataloader, so it's
    # loaded here from a JSON file you provide, and num_conditions is
    # derived from it rather than being passed in from outside (see the fix
    # to build_network_architecture below for why).
    #
    # `cond` is deliberately generic: it's whatever categorical label you
    # want SaBN's independent affine layers indexed by -- scanner/site,
    # sex, age bracket, vendor, acquisition protocol, task id in a
    # multi-task setup, etc. This class has no opinion on what it represents,
    # it just needs a case_id -> int mapping, ints contiguous starting at 0:
    #
    #   {"PANORAMA_0001": 0, "PANORAMA_0002": 1, "PANORAMA_0350": 0, ...}
    # ------------------------------------------------------------------
    COND_MAP_PATH = "patient_sex_map.json"

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.cond_map: Dict[str, int] = self._load_cond_map()
        self.num_conditions = (max(self.cond_map.values()) + 1) if self.cond_map else 1

    def _load_cond_map(self) -> Dict[str, int]:
        if self.COND_MAP_PATH is None:
            self.print_to_log_file(
                "WARNING: SABN_COND_MAP_PATH is not set. All cases will be "
                "treated as a single condition (num_conditions=1), which "
                "makes SaBN degenerate to plain BatchNorm. Set the env var "
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
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        arch_kwargs = dict(configuration_manager.network_arch_init_kwargs)
        arch_kwargs['norm_op_kwargs'] = dict(arch_kwargs.get('norm_op_kwargs') or {})
        arch_kwargs['norm_op_kwargs']['num_conditions'] = self.num_conditions

        return get_network_from_plans(
            configuration_manager.network_arch_class_name,
            arch_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
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
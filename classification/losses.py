# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.utils.clip_grad import dispatch_clip_grad

class TempNormSoftCrossEntropy(nn.Module):

    def __init__(self, eps=1e-12, nan_replacement_value=0.0):
        super(TempNormSoftCrossEntropy, self).__init__()

        self.eps = eps
        self.nan_replacement_value = nan_replacement_value

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # any nan value in x is replaced by 0.0
        x = torch.nan_to_num(x, nan=self.nan_replacement_value)
        # if torch.isnan(x).any():
        #     print("NaN detected in x. Investigate input x.")

        x_mean = x.mean(dim=-1, keepdim=True)
        # if torch.isnan(x_mean).any():
        #     print("NaN detected in x_mean. Investigate x_mean.")

        x_std = x.std(dim=-1, keepdim=True) + self.eps
        # if torch.isnan(x_std).any():
        #     print("NaN detected in x_std. Investigate x_std.")

        x = (x - x_mean) / x_std
        # if torch.isnan(x).any():
        #     print("NaN detected in normalized x. Investigate normalized x.")
        # any nan value in normalized x is replaced by 0.0
        x = torch.nan_to_num(x, nan=self.nan_replacement_value)

        temperature = 2
        log_probs = F.log_softmax(x/temperature, dim=-1)
        loss = torch.sum(-target * log_probs, dim=-1)
        loss = loss.mean()
        return loss

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class Grad_Accumulate_Scaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, idx, lr_scheduler, epoch, num_steps, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False,
                 accumulation_steps=1):
        loss = loss / accumulation_steps
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        if (idx + 1) % accumulation_steps==0:
            self._scaler.step(optimizer)
            optimizer.zero_grad()
            self._scaler.update()
            lr_scheduler.step_update(epoch * num_steps + idx)

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
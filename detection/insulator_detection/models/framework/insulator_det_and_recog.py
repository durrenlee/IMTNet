import math
import os
import sys
import json

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Optimizer
import torch.nn as nn
from timm.optim import create_optimizer_v2
from contextlib import contextmanager

# fix 'DetDataPreprocessor is not in the mmengine::model registry'.
from mmdet.utils import register_all_modules
register_all_modules()
o_path = os.getcwd()
sys.path.append(o_path)

from insulator_detection.models.lvt_str.str_model import InsulatorSTRModel
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, overload, Sequence, Tuple, Union
from pytorch_lightning.utilities.types import (
    EPOCH_OUTPUT,
    STEP_OUTPUT,
)

from mmengine.config import Config
from mmengine.runner import Runner
import os.path as osp
from mmengine.utils import is_list_of
from mmengine.runner.loops import _update_losses, _parse_losses
from mmengine.runner.amp import autocast

from insulator_detection.data.module import SceneTextDataModule
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.utilities import rank_zero_info

class InsulatorDetAndReco(pl.LightningModule):
    def __init__(self, batch_size, img_size, max_label_length, charset_train, charset_test,
                 det_lr, str_lr, str_weight_decay, str_coef_lr, str_coef_wd, str_warmup_pct,
                 code_path, train_dataset_size, val_dataset_size,
                 str_dataset_root_dir, str_dataset_output_url,
                 str_dataset_train_dir, str_dataset_remove_whitespace,
                 str_dataset_normalize_unicode, str_dataset_augment,
                 str_dataset_num_workers, str_dataset_openai_meanstd,
                 str_map_file_train, str_map_file_val, str_batch_size,
                 detection_config_path, **kwargs: Any) -> None:
        super().__init__()
        rank_zero_info("=====================InsulatorDetAndReco init=========================")
        # disable automatic optimization for manual optimization
        self.automatic_optimization = False

        self.save_hyperparameters()

        self.batch_size = batch_size
        self.img_size = img_size
        self.max_label_length = max_label_length
        self.charset_train = charset_train
        self.charset_test = charset_test
        self.det_lr = det_lr
        self.str_lr = str_lr
        self.str_weight_decay = str_weight_decay
        self.str_coef_lr = str_coef_lr
        self.str_coef_wd = str_coef_wd
        self.str_warmup_pct = str_warmup_pct  # warmup epochs=warmup_pct*total_epochs
        self.train_dataset_size = train_dataset_size
        self.val_dataset_size = val_dataset_size
        self.str_bs = str_batch_size
        self.kwargs = kwargs

        # init str model
        self.str_model = InsulatorSTRModel(img_size=img_size,
                                           charset_train=charset_train, charset_test=charset_test,
                                           max_label_length=max_label_length,
                                           batch_size=batch_size, lr=self.str_lr, warmup_pct=self.str_warmup_pct,
                                           weight_decay=self.str_weight_decay, **kwargs)
        self.str_data_module = SceneTextDataModule(
            root_dir=str_dataset_root_dir, output_url=str_dataset_output_url,
            train_dir=str_dataset_train_dir, batch_size=batch_size,
            img_size=img_size, charset_train=charset_train,
            charset_test=charset_test, max_label_length=max_label_length,
            remove_whitespace=str_dataset_remove_whitespace,
            normalize_unicode=str_dataset_normalize_unicode,
            augment=str_dataset_augment, num_workers=str_dataset_num_workers,
            openai_meanstd=str_dataset_openai_meanstd
        )

        # init mmdetection runner
        # load config
        det_cfg = Config.fromfile(detection_config_path)
        det_cfg.launcher = 'none'

        # work_dir: running config, log and checkpoint  files
        work_dir = None
        if work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            det_cfg.work_dir = work_dir
        elif det_cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            det_cfg.work_dir = osp.join('./work_dirs',
                                        osp.splitext(osp.basename(detection_config_path))[0])

        # enable automatic-mixed-precision training
        det_cfg.optim_wrapper.type = 'AmpOptimWrapper'
        det_cfg.optim_wrapper.loss_scale = 'dynamic'

        det_cfg.resume = True
        det_cfg.load_from = None

        # override batch size in det configuration file
        det_cfg.train_dataloader.batch_size = self.batch_size

        # build the runner from config
        self.det_runner = Runner.from_cfg(det_cfg)

        # build train_loop
        self.det_runner._train_loop = self.det_runner.build_train_loop(
            self.det_runner._train_loop)  # type: ignore

        # `build_optimizer` should be called before `build_param_scheduler`
        #  because the latter depends on the former
        self.det_runner.optim_wrapper = self.det_runner.build_optim_wrapper(self.det_runner.optim_wrapper)

        # Automatically scaling lr by linear scaling rule
        self.det_runner.scale_lr(self.det_runner.optim_wrapper, self.det_runner.auto_scale_lr)

        # param_schedulers
        if self.det_runner.param_schedulers is not None:
            self.det_runner.param_schedulers = self.det_runner.build_param_scheduler(  # type: ignore
                self.det_runner.param_schedulers)  # type: ignore

        # build val_loop
        if self.det_runner._val_loop is not None:
            self.det_runner._val_loop = self.det_runner.build_val_loop(
                self.det_runner._val_loop)  # type: ignore

        # call 'before_run' hook
        self.det_runner.call_hook('before_run')

        # initialize the model weights
        self.det_runner._init_model_weights()

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.det_runner.load_or_resume()

        # load map files
        with open(str_map_file_train, "r") as infile_train:
            self.map_data_train = json.load(infile_train)
        with open(str_map_file_val, "r") as infile_val:
            self.map_data_val = json.load(infile_val)

        # Initiate inner count of `optim_wrapper`.
        self.det_runner.optim_wrapper.initialize_count_status(
            self.det_runner.model,
            self.det_runner._train_loop.iter,  # type: ignore
            self.det_runner._train_loop.max_iters)  # type: ignore

        # optimizer settings
        self._accumulative_counts = 1
        self._inner_count = 0
        self._max_counts = -1
        self.clip_grad_kwargs = self.det_runner.optim_wrapper.clip_grad_kwargs

        # accumulate cropped images and labels
        self.str_train_accumulated_images = []
        self.str_train_accumulated_labels = []
        self.str_validation_accumulated_images = []
        self.str_validation_accumulated_labels = []

    def on_fit_start(self) -> None:
        rank_zero_info("*******************on_fit_start(self)*******************")
        """Launch training."""
        self.det_runner._train_loop.runner.call_hook('before_train')

    def on_train_epoch_start(self):
        rank_zero_info("*******************on_train_epoch_start(self)*******************")
        self.det_runner._train_loop.runner.call_hook('before_train_epoch')

        """Iterate one epoch."""

        # make sure sanity check is disabled in python lightning(num_sanity_val_steps: 0)
        # assert self.current_epoch == self.det_runner._train_loop._epoch

        # enable train()
        self.det_runner._train_loop.runner.model.train()

    @contextmanager
    def optim_context(self, model: nn.Module):
        """A Context for gradient accumulation and automatic mix precision
        training.

        If subclasses need to enable the context for mix precision training,
        e.g., ``:class:`AmpOptimWrapper``,  the corresponding context should be
        enabled in `optim_context`. Since ``OptimWrapper`` uses default fp32
        training, ``optim_context`` will only enable the context for
        blocking the unnecessary gradient synchronization during gradient
        accumulation

        If model is an instance with ``no_sync`` method (which means
        blocking the gradient synchronization) and
        ``self._accumulative_counts != 1``. The model will not automatically
        synchronize gradients if ``cur_iter`` is divisible by
        ``self._accumulative_counts``. Otherwise, this method will enable an
        empty context.

        Args:
            model (nn.Module): The training model.
        """
        # During gradient accumulation process, the gradient synchronize
        # should only happen before updating parameters.
        if not self.should_sync() and hasattr(model, 'no_sync'):
            with model.no_sync():
                yield
        else:
            yield

    def should_update(self) -> bool:
        """Decide whether the parameters should be updated at the current
        iteration.

        Called by :meth:`update_params` and check whether the optimizer
        wrapper should update parameters at current iteration.

        Returns:
            bool: Whether to update parameters.
        """
        return (self._inner_count % self._accumulative_counts == 0
                or self._inner_count == self._max_counts)

    def should_sync(self) -> bool:
        """Decide whether the automatic gradient synchronization should be
        allowed at the current iteration.

        It takes effect when gradient accumulation is used to skip
        synchronization at the iterations where the parameter is not updated.

        Since ``should_sync`` is called by :meth:`optim_context`, and it is
        called before :meth:`backward` which means ``self._inner_count += 1``
        has not happened yet. Therefore, ``self._inner_count += 1`` should be
        performed manually here.

        Returns:
            bool: Whether to block the automatic gradient synchronization.
        """
        return ((self._inner_count + 1) % self._accumulative_counts == 0
                or (self._inner_count + 1) == self._max_counts)

    # def forward(self, x):
    #     return self.str_model(x)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # access optimizers (one or multiple)
        det_optimizer, str_optimizer = self.optimizers()
        # det_optimizer = self.optimizers()
        det_scheduler, str_scheduler = self.lr_schedulers()
        # det_scheduler = self.lr_schedulers()

        ##########################
        #      detection task    #
        ##########################

        # call 'before_train_iter' hook
        self.det_runner._train_loop.runner.call_hook(
            'before_train_iter', batch_idx=batch_idx, data_batch=batch)

        # Enable automatic mixed precision training context.
        with self.optim_context(self):
            data = self.det_runner._train_loop.runner.model.data_preprocessor(batch, True)
            losses = self.det_runner._train_loop.runner.model._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.det_runner._train_loop.runner.model.parse_losses(losses)  # type: ignore

        # loss factor
        if self._accumulative_counts == 1:
            loss_factor = 1
        loss = parsed_losses / loss_factor

        # use python lightning manual optimization manual_backward to replace self.backward(loss)
        self.manual_backward(loss)
        self._inner_count += 1

        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.should_update():
            if self.clip_grad_kwargs:
                # clip val and algorithm need to be determined
                self.clip_gradients(det_optimizer, gradient_clip_val=0.5,
                                    gradient_clip_algorithm=self.clip_grad_kwargs.pop('type', 'norm'))
            # update model parameters
            det_optimizer.step()
            # clear the gradients from the previous training step
            det_optimizer.zero_grad()

            # Step the scheduler
            det_scheduler.step()

        # call 'after_train_iter' hook
        self.det_runner._train_loop.runner.call_hook(
            'after_train_iter',
            batch_idx=batch_idx,
            data_batch=batch,
            outputs=log_vars)
        self.det_runner._train_loop._iter += 1

        ##########################
        #      str task          #
        ##########################
        # clear the gradients from the previous training step
        # str_optimizer.zero_grad()

        # get indexes(not img/label key) in map file according to image id
        img_ids = [data_sample.img_id for data_sample in batch['data_samples']]
        for img_id in img_ids:
            try:
                keys = self.map_data_train[str(img_id)]
                for key in keys:
                    image, label = self.str_data_module.train_dataset[key]  # Unpack the tuple
                    self.str_train_accumulated_images.append(image)
                    self.str_train_accumulated_labels.append(label)
            except KeyError:
                continue

        str_loss = torch.tensor(0.0, device=self.device, requires_grad=True)  # Default zero loss
        accumulated_size = len(self.str_train_accumulated_images)

        if accumulated_size >= self.str_bs:
            cropped_images = torch.stack(self.str_train_accumulated_images[:self.str_bs], dim=0).to(self.device)
            labels = self.str_train_accumulated_labels[:self.str_bs]
            str_batch = (cropped_images, labels)
            str_loss = self.str_model.training_step(str_batch, batch_idx)
            # use python lightning manual optimization manual_backward to replace self.backward(loss)
            self.manual_backward(str_loss)
            # Remove processed samples from the buffer
            self.str_train_accumulated_images = self.str_train_accumulated_images[self.str_bs:]
            self.str_train_accumulated_labels = self.str_train_accumulated_labels[self.str_bs:]

            # update model parameters
            str_optimizer.step()
            # clear the gradients from the previous training step
            str_optimizer.zero_grad()
            str_scheduler.step()
        # str task end

        return log_vars

    def on_validation_start(self) -> None:
        rank_zero_info("*******************on_validation_start(self)*******************")
        self.det_runner._train_loop.runner.call_hook('after_train_epoch')

        #  fixed val epoch no. display issue: make sure sanity check is disabled in
        #  python lightning(num_sanity_val_steps: 0)
        # self.det_runner._train_loop._epoch += 1

        """Launch validation."""
        self.det_runner._train_loop._decide_current_val_interval()
        self.det_runner._val_loop.runner.call_hook('before_val')

    def on_validation_epoch_start(self) -> None:
        self.det_runner.logger.info("*******************on_validation_epoch_start(self)*******************")

        self.det_runner._val_loop.runner.call_hook('before_val_epoch')
        self.det_runner._val_loop.runner.model.eval()

        # clear val loss
        self.det_runner._val_loop.val_loss.clear()

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        ##########################
        #      detection task    #
        ##########################
        self.det_runner._val_loop.runner.call_hook(
            'before_val_iter', batch_idx=batch_idx, data_batch=batch)

        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.det_runner._val_loop.fp16):
            data = self.det_runner._val_loop.runner.model.data_preprocessor(batch, False)
            outputs = self.det_runner._val_loop.runner.model._run_forward(data, mode='predict')  # type: ignore

        outputs, self.det_runner._val_loop.val_loss = _update_losses(outputs, self.det_runner._val_loop.val_loss)

        self.det_runner._val_loop.evaluator.process(data_samples=outputs, data_batch=batch)
        self.det_runner._val_loop.runner.call_hook(
            'after_val_iter',
            batch_idx=batch_idx,
            data_batch=batch,
            outputs=outputs)

        ##########################
        #      str task          #
        ##########################
        # get indexes(not img/label key) in map file according to image id
        img_ids = [data_sample.img_id for data_sample in batch['data_samples']]
        for img_id in img_ids:
            try:
                keys = self.map_data_val[str(img_id)]
                for key in keys:
                    image, label = self.str_data_module.val_dataset[key]  # Unpack the tuple
                    self.str_validation_accumulated_images.append(image)
                    self.str_validation_accumulated_labels.append(label)
            except KeyError:
                continue

        # Check if enough data is accumulated for a full STR batch
        if len(self.str_validation_accumulated_images) >= self.str_bs:
            # Stack images into a single tensor: Shape (N, C, H, W)
            # cropped_images = torch.stack(cropped_images, dim=0).to(self.device)

            # Create STR batch
            cropped_images = torch.stack(self.str_validation_accumulated_images[:self.str_bs], dim=0).to(self.device)
            labels = self.str_validation_accumulated_labels[:self.str_bs]
            # Remove processed samples
            self.str_validation_accumulated_images = self.str_validation_accumulated_images[self.str_bs:]
            self.str_validation_accumulated_labels = self.str_validation_accumulated_labels[self.str_bs:]

            str_batch = (cropped_images, labels)
            outputs = self.str_model.validation_step(str_batch, batch_idx)
            return outputs

        # If batch size is not yet 128, skip STR validation for this step
        return None

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        rank_zero_info("*******************validation_epoch_end(self, outputs)*******************")
        ##########################
        #      detection task    #
        ##########################
        # # if (self.det_runner._train_loop.runner.val_loop is not None
        # #         and self.det_runner._train_loop._epoch >= self.det_runner._train_loop.val_begin
        # #         and (self.det_runner._train_loop._epoch % self.det_runner._train_loop.val_interval == 0
        # #              or self.det_runner._train_loop._epoch == self.det_runner._train_loop._max_epochs)):
        #
        # compute metrics
        bbox_mAP = 0.0
        bbox_mAP_50 = 0.0
        bbox_mAP_75 = 0.0
        try:
            metrics = self.det_runner._val_loop.evaluator.evaluate(len(self.det_runner._val_loop.dataloader.dataset))
        except:  # sanity check for validation steps
            metrics = {}

        if metrics != {}:
            if self.det_runner._val_loop.val_loss:
                loss_dict = _parse_losses(self.det_runner._val_loop.val_loss, 'val')
                metrics.update(loss_dict)

            self.det_runner._val_loop.runner.call_hook('after_val_epoch', metrics=metrics)
            bbox_mAP = metrics['coco/bbox_mAP']
            bbox_mAP_50 = metrics['coco/bbox_']
            bbox_mAP_75 = metrics['coco/bbox_mAP_75']
        else:
            self.det_runner.logger.warn('metrics is {} during sanity check before starting train.')

        # self.log('bbox_mAP', bbox_mAP)
        ##########################
        #      str task          #
        ##########################
        # Process any remaining data
        # Handle remaining accumulated STR samples
        if len(self.str_validation_accumulated_images) > 0:
            cropped_images = torch.stack(self.str_validation_accumulated_images, dim=0).to(self.device)
            labels = self.str_validation_accumulated_labels
            str_batch = (cropped_images, labels)

            # Final STR validation step
            final_outputs = self.str_model.validation_step(str_batch, -1)
            outputs.append(final_outputs)  # Add final batch outputs to overall outputs
            self.str_validation_accumulated_images.clear()
            self.str_validation_accumulated_labels.clear()

        acc, ned, loss = self.str_model.validation_epoch_end(outputs)
        print(f'val_accuracy:{str(100 * acc)}')
        print(f'val_NED:{100 * ned}')
        print(f'val_loss:{loss}')
        print(f'hp_metric:{acc}')

        self.log('val_accuracy', 100 * acc, sync_dist=True)
        self.log('val_NED', 100 * ned, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        self.log('hp_metric', acc, sync_dist=True)

    def on_validation_end(self) -> None:
        self.det_runner.logger.info("*******************on_validation_end(self)*******************")
        self.det_runner._val_loop.runner.call_hook('after_val')

    # notice: on_train_epoch_end() is executed after on_validation_end()
    def on_train_epoch_end(self) -> None:
        self.det_runner.logger.info("*******************on_train_epoch_end(self)*******************")

        # below two line are moved to on_validation_start()
        self.det_runner._train_loop.runner.call_hook('after_train_epoch')
        self.det_runner._train_loop._epoch += 1

    def on_fit_end(self) -> None:
        self.det_runner.logger.info("*******************on_fit_end(self)*******************")
        self.det_runner.call_hook('after_run')

    def configure_optimizers(self):
        ##########################
        #      det task          #
        ##########################
        # # optimizer and schedulers from experimented on mmdetection
        # # Define the AdamW optimizer for detector
        det_optimizer = create_optimizer_v2(
            self.det_runner._train_loop.runner.model.parameters(),
            opt='adamw',  # Optimizer type
            lr=self.det_lr,  # Learning rate
            foreach=None,
            # weight_decay=0.05,  # Weight decay
            weight_decay=0.05,
            eps=1e-8,  # Epsilon for numerical stability, passed through kwargs
            amsgrad=False,
            betas=(0.9, 0.999),
            capturable=False,
            differentiable=False,
            fused=None,
            maximize=False
        )
        # # Linear warmup for the first 30% of steps
        linear_scheduler = LinearLR(
            optimizer=det_optimizer,
            start_factor=0.1,
            total_iters=self.trainer.estimated_stepping_batches * 0.2,
            last_epoch=-1
        )
        #  Cosine annealing for the remaining 70% of steps
        cosine_scheduler = CosineAnnealingLR(
            optimizer=det_optimizer,
            T_max=self.trainer.estimated_stepping_batches * 0.8,
            eta_min=1e-6,
            last_epoch=-1,
        )

        # Chain both schedulers with SequentialLR
        det_scheduler = SequentialLR(
            optimizer=det_optimizer,
            schedulers=[linear_scheduler, cosine_scheduler],
            milestones=[int(self.trainer.estimated_stepping_batches * 0.2)]
        )

        ##########################
        #      str task          #
        ##########################
        self.str_model.weight_decay = self.str_weight_decay
        self.str_model.coef_lr = self.str_coef_lr
        self.str_model.coef_wd = self.str_coef_wd


        agb = self.trainer.accumulate_grad_batches
        # Linear scaling so that the effective learning rate is constant regardless of the number of GPUs used with DDP.
        lr_scale = agb * math.sqrt(self.trainer.num_devices) * self.str_bs / 256.
        lr = lr_scale * self.str_model.lr

        # https://github.com/mlfoundations/open_clip/blob/b4cf9269b0b11c0eea47cb16039369a46bd67449/src/training/main.py#L171
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n \
                               or "pos_queries" in n or "text_embed" in n
        include = lambda n, p: not exclude(n, p)

        # encoder parameters
        encoder_params = list(self.str_model.clip_model.named_parameters())
        enc_gain_or_bias_params = [p for n, p in encoder_params if exclude(n, p) and p.requires_grad]
        enc_rest_params = [p for n, p in encoder_params if include(n, p) and p.requires_grad]

        # decoder parameters
        decoder_params = [(n, p) for n, p in list(self.str_model.named_parameters()) if "clip_model" not in n]
        dec_gain_or_bias_params = [p for n, p in decoder_params if exclude(n, p) and p.requires_grad]
        dec_rest_params = [p for n, p in decoder_params if include(n, p) and p.requires_grad]

        rank_zero_info(
            "[InsulatorSTRModel] The length of encoder params with and without weight decay is {} and {}, respectively.".format(
                len(enc_rest_params), len(enc_gain_or_bias_params)
            ))
        rank_zero_info(
            "[InsulatorSTRModel] The length of decoder params with and without weight decay is {} and {}, respectively.".format(
                len(dec_rest_params), len(dec_gain_or_bias_params)
            ))

        str_optimizer = torch.optim.AdamW(
            [
                {"params": enc_gain_or_bias_params, "weight_decay": 0., 'lr': lr},
                {"params": enc_rest_params, "weight_decay": self.str_model.weight_decay, 'lr': lr},
                {"params": dec_gain_or_bias_params, "weight_decay": 0., 'lr': lr * self.str_model.coef_lr},
                {"params": dec_rest_params, "weight_decay": self.str_model.weight_decay * self.str_model.coef_wd,
                 'lr': lr * self.str_model.coef_lr},
            ],
            lr=lr, betas=(0.9, 0.98), eps=1.0e-6,
        )

        str_sched = OneCycleLR(str_optimizer, [lr, lr, lr * self.str_model.coef_lr, lr * self.str_model.coef_lr],
                           self.str_model.trainer.estimated_stepping_batches, pct_start=self.str_model.warmup_pct,
                           cycle_momentum=False)

        return [
                {
                    "optimizer": det_optimizer,
                    "lr_scheduler": {
                        "scheduler": det_scheduler,
                        "interval": "step",  # Update the LR every step
                        "frequency": 1
                    }
                },
                {
                    "optimizer": str_optimizer,
                    "lr_scheduler": {
                        "scheduler": str_sched,
                        "interval": "step",
                        "frequency": 1
                         }
                }
            ]

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
        optimizer.zero_grad(set_to_none=True)

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        rank_zero_info("*******************test_step()*******************")


def parse_losses(
        self, losses: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Parses the raw outputs (losses) of the network.

    Args:
        losses (dict): Raw output of the network, which usually contain
            losses and other necessary information.

    Returns:
        tuple[Tensor, dict]: There are two elements. The first is the
        loss tensor passed to optim_wrapper which may be a weighted sum
        of all losses, and the second is log_vars which will be sent to
        the logger.
    """
    log_vars = []
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars.append([loss_name, loss_value.mean()])
        elif is_list_of(loss_value, torch.Tensor):
            log_vars.append(
                [loss_name,
                 sum(_loss.mean() for _loss in loss_value)])
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    loss = sum(value for key, value in log_vars if 'loss' in key)
    log_vars.insert(0, ['loss', loss])
    log_vars = OrderedDict(log_vars)  # type: ignore

    return loss, log_vars  # type: ignore

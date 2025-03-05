import os
import time
import math
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict, OmegaConf
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.model_summary import summarize
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, LearningRateMonitor
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from models.lvt_str.base import BaseSystem
from models.lvt_str.utils import create_model
from data.data_loader_module import InsulatorDataModule

from mmdet.utils import setup_cache_size_limit_of_dynamo
from dist_utils import copy_remote, is_main_process
from insulator_detection.data.create_det_json import create_coco_json
from insulator_detection.data.create_lmdb_from_merged_ann import parse_merged_json, create_lmdb


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    print('=======main function start in train.py===========')
    torch.autograd.set_detect_anomaly(True)
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()
    trainer_strategy = None

    with open_dict(config):
        # create coco-based train json file for data module based on mmdetection
        if not os.path.exists(config.model.detection_train_json):
            print(f'config.model.detection_merged_train_json:{config.model.detection_merged_train_json}')
            print(f'config.model.detection_train_json:{config.model.detection_train_json}')
            create_coco_json(config.model.detection_merged_train_json, config.model.detection_train_json)
        while not os.path.exists(config.model.detection_train_json):
            print('sleeping a few of seconds for detection train json file creation completion...')
            time.sleep(10)

        # create coco-based val json file for data module based on mmdetection
        if not os.path.exists(config.model.detection_val_json):
            print(f'config.model.detection_merged_val_json:{config.model.detection_merged_val_json}')
            print(f'config.model.detection_val_json:{config.model.detection_val_json}')
            create_coco_json(config.model.detection_merged_val_json, config.model.detection_val_json)
        while not os.path.exists(config.model.detection_val_json):
            print('sleeping a few of seconds for detection val json file creation completion...')
            time.sleep(10)

        # *create lmdb and map files*
        # map file root
        if not os.path.exists(config.model.str_map_dir):
            os.makedirs(config.model.str_map_dir)

        # create train lmdb and train map json file
        if not os.path.exists(config.model.str_lmdb_train):
            os.makedirs(config.model.str_lmdb_train)
            img_label_list_train = parse_merged_json(config.model.detection_merged_train_json,
                                                     config.model.str_merged_cropped_imgs_train)
            create_lmdb(img_label_list_train, config.model.str_lmdb_train, config.model.str_map_file_train)

        lmdb_train_file = os.path.join(config.model.str_lmdb_train, 'data.mdb')
        while not os.path.exists(lmdb_train_file):
            print('sleeping a few of seconds for lmdb train file creation completion...')
            time.sleep(10)
        while not os.path.exists(config.model.str_map_file_train):
            print('sleeping a few of seconds for str img id/keys map train file creation completion...')
            time.sleep(10)

        # create val lmdb and val map json file
        if not os.path.exists(config.model.str_lmdb_val):
            os.makedirs(config.model.str_lmdb_val)
            img_label_list_val = parse_merged_json(config.model.detection_merged_val_json,
                                                   config.model.str_merged_cropped_imgs_val)
            create_lmdb(img_label_list_val, config.model.str_lmdb_val, config.model.str_map_file_val)
        lmdb_val_file = os.path.join(config.model.str_lmdb_val, 'data.mdb')
        while not os.path.exists(lmdb_val_file):
            print('sleeping a few of seconds for lmdb val file creation completion...')
            time.sleep(10)
        while not os.path.exists(config.model.str_map_file_val):
            print('sleeping a few of seconds for str img id/keys map val file creation completion...')
            time.sleep(10)

        if not os.path.exists(config.model.str_synth_dir):
            os.makedirs(config.model.str_synth_dir)
        if not os.path.exists(config.model.str_lmdb_test):
            os.makedirs(config.model.str_lmdb_test)

        # Special handling for GPU-affected config
        gpus = config.trainer.get('gpus', 0)
        if gpus:
            # Use mixed-precision training
            # config.trainer.precision = 16

            config.trainer.precision = 32  # fix binary_cross_entropy() autocast to float32 issue

        if gpus > 1:
            # Use DDP
            config.trainer.strategy = 'ddp'
            # DDP optimizations
            trainer_strategy = DDPStrategy(
                find_unused_parameters=getattr(config.model, "find_unused_parameters", False),
                gradient_as_bucket_view=True)

            # # Scale steps-based config
            # config.trainer.val_check_interval //= gpus
            # if config.trainer.get('max_steps', -1) > 0:
            #     config.trainer.max_steps //= gpus

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    # print config
    rank_zero_info(OmegaConf.to_yaml(config))

    # create model
    # If specified, use pretrained weights to initialize the clip model
    if config.pretrained is not None:
        model: BaseSystem = create_model(config.pretrained, True)
    else:
        model: pl.LightningModule = hydra.utils.instantiate(config.model)
        print(f'model:{model}')
    print('*****************summarize model************************')
    rank_zero_info(summarize(model, max_depth=2))

    # data module - insulator defect detection dataset
    datamodule: InsulatorDataModule = hydra.utils.instantiate(config.data)
    datamodule._train_dataset = model.det_runner._train_loop.dataloader.dataset
    datamodule._val_dataset = model.det_runner._val_loop.dataloader.dataset
    # datamodule._test_dataset = model.det_runner._test_loop.dataloader.dataset
    datamodule.train_data_loader = model.det_runner._train_loop.dataloader
    datamodule.val_data_loader = model.det_runner._val_loop.dataloader
    # datamodule.test_data_loader = model.det_runner._test_loop.dataloader
    print(f'len(datamodule.train_dataset):{len(datamodule.train_dataset)}')
    config.model.train_dataset_size = len(datamodule.train_dataset)
    config.model.val_dataset_size = len(datamodule.val_dataset)

    # checkpoint
    if config.model.get('has_det_branch', True):
        monitor = 'bbox_mAP'
        set_filename = '{epoch}-{step}-{bbox_mAP:.4f}'
        print("%%%%%%%%%%%%%%%config.model.has_det_branch")
    if config.model.get('has_str_branch', True):
        monitor = 'val_accuracy'
        set_filename = '{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}'
        print("%%%%%%%%%%%%%%%config.model.has_str_branch")

    checkpoint = ModelCheckpoint(monitor=monitor, mode='max', save_top_k=1, save_last=True,
                                    filename=set_filename,
                                    every_n_epochs=1)
    # checkpoint = ModelCheckpoint(monitor='bbox_mAP', mode='max', save_top_k=1, save_last=True,
    #                                 filename='{epoch}-{step}-{bbox_mAP:.4f}',
    #                                 every_n_epochs=1)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor, checkpoint]

    if getattr(config, 'swa', False):
        # set swa lrs
        swa_epoch_start = 0.8
        lr_scale = math.sqrt(torch.cuda.device_count()) * config.data.batch_size / 256.
        lr = lr_scale * config.model.lr
        if "clip" in config.model.name:
            swa_lrs = [lr, lr, config.model.coef_lr * lr, config.model.coef_lr * lr]
        else:
            swa_lrs = [lr,]

        swa_lrs = [x * (1 - swa_epoch_start) for x in swa_lrs]
        swa = StochasticWeightAveraging(swa_lrs=swa_lrs, swa_epoch_start=swa_epoch_start)
        callbacks.append(swa)

    cwd = HydraConfig.get().runtime.output_dir if config.ckpt_path is None else \
            str(Path(config.ckpt_path).parents[1].absolute())
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=TensorBoardLogger(cwd, '', '.'),
                                               # resume_from_checkpoint='/root/output/insulatordetandstr_2024-12-21_19-21-02/checkpoints/last.ckpt',
                                               strategy=trainer_strategy, enable_model_summary=False,
                                               accumulate_grad_batches=config.trainer.accumulate_grad_batches,
                                               callbacks=callbacks)

    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)

    # copy data and perform test
    torch.distributed.barrier()
    if is_main_process():
        copy_remote(cwd, config.data.output_url)
        test_call(cwd, config.data.root_dir, config.model.code_path)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@rank_zero_only
def test_call(cwd, data_dir, code_path=None):
    file = os.path.join(code_path, 'test.py')
    assert os.path.exists(file)
    print("The execute file is {}".format(file))
    ckpts = [x for x in os.listdir(os.path.join(cwd, 'checkpoints')) if 'val' in x]
    val_acc = [float(x.split('-')[-2].split('=')[-1]) for x in ckpts]

    best_ckpt = os.path.join(os.path.join(cwd, 'checkpoints'), ckpts[val_acc.index(max(val_acc))])
    print("The best ckpt is {}".format(best_ckpt))
    best_epoch = int(best_ckpt.split('/')[-1].split('-')[0].split('=')[-1])
    print('The val accuracy is best {}-{}e'.format(max(val_acc), best_epoch))

    # test best
    # print("\n Test results with the best checkpoint")
    # os.system("python {} {} --data_root {} --new".format(file, best_ckpt, data_dir))
    # test last

    print("\n Test results with the last checkpoint")
    last_ckpt = os.path.join(os.path.join(cwd, 'checkpoints'), "last.ckpt")
    os.system("python {} {} --data_root {} --new".format(file, last_ckpt, data_dir))

    # test last with refinement
    # print("\n Test results with the last checkpoint")
    # last_ckpt = os.path.join(os.path.join(cwd, 'checkpoints'), "last.ckpt")
    # os.system("python {} {} --data_root {} --new --clip_refine".format(file, last_ckpt, data_dir))

if __name__ == '__main__':
    main()

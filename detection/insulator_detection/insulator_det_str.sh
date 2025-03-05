#!/bin/bash
python train.py     ++experiment=insulatordetandstr trainer.max_epochs=100 \
                                           model.det_lr=1e-4 model.str_lr=8.4e-5 model.batch_size=2  \
                                           trainer.gpus=1  \
                                           model.clip_pretrained=/root/pretrained/open_clip_pytorch_model.bin \
                                           trainer.accumulate_grad_batches=1




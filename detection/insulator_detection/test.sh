#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1  # gpu number id: 0 or 1..
ckpt_id=$2    # lgtvit4str model checkpoint file name

abs_root=/YOURROOT # root path

exp_path=${abs_root}/YOURPATH/checkpoints

runfile=${abs_root}/YOURPROJECT/test.py   # test.py path

clip_model_path=${abs_root}/pretrained/ViT-B-16.pt    # CLIP model checkpoint file path
#clip_model_path=${abs_root}/pretrained/clip/ViT-L-14.pt
data_root=${abs_root}/data/str_dataset

python ${runfile} ${exp_path} \
            --data_root ${data_root} \
            --clip_model_path ${clip_model_path} \
            --new
            # --clip_refine \
            # --sample_K 5 \
            # --sample_K2 3 \
            # --sample_total 50 \
            # --alpha 0.1 \

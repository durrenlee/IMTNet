#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1 # gpu number id: 0 or 1..
ckpt_id=$2  # # lgtvit4str model checkpoint file name
images_path=$3

abs_root=/root # root path

exp_path=${ckpt_id} # export path
# read.py path
runfile=${abs_root}/dilateformer-main-DwConv-dcn/detection/text_detection/read.py

python ${runfile} ${exp_path} --images_path ${images_path}

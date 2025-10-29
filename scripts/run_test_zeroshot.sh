#!/usr/bin/env bash
#bash scripts/run_test_zeroshot.sh 
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

weight=$2

export CUDA_VISIBLE_DEVICES=0
/opt/conda/envs/glm/bin/python -m torch.distributed.launch --master_port 1238 --nproc_per_node=1 --use_env \
    test_zeroshot.py --config ${config} --weights ${weight} ${@:3}
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$3
mkdir -p logs
python3 trainer.py \
    --arch $1 \
    --experiment-type $2 \
    --workers 1 \
    --epochs 160 \
    --start-epoch 0 \
    --batch-size 128 \
    --learning-rate 0.01 \
    --momentum 0.9  \
    --weight-decay 1e-4 \
    --print-freq 50 \
    --resume None \
    --save-dir checkpoints/$1-$2 \
    --save-every 159 \
    --lr-drop-factor 0.1 \
    --lr-drop-epochs 80 120 \
|& tee -a logs/log-$1-$2.txt

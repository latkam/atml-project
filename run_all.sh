#!/bin/bash
cuda_visible_devices=1
mkdir -p logs
for model in resnet14 resnet32 resnet56 resnet110 resnet218 resnet434 #resnet866
do
    ./run_one_model.sh $model $cuda_visible_devices #&
done
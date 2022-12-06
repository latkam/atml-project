#!/bin/bash
cuda_device=1
mkdir -p logs
for model in resnet14 resnet32 resnet56 resnet110 resnet218 resnet434 resnet866
do
    ./run_one_model.sh $model $cuda_device #&
done
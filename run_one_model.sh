#!/bin/bash
model=$1
cuda_visible_devices=$2
for experiment in full nobn bn random
do
    ./run_one.sh $model $experiment $cuda_visible_devices
done
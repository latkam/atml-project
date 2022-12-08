#!/bin/bash
for experiment in full #nobn bn random
do
    ./run_one.sh $1 $experiment $2
done
#!/bin/bash
for experiment in full #nobn bn
do
    ./run_one.sh $1 $experiment $2
done
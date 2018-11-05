#!/bin/bash

INSTR=./tools/memtime_wrapper.py
EXE=./tools/grb_blackbox_runner.py

MAX_MEM=$((20*1024*1024))
MAX_TIME=7200

trap "exit" INT
for prop in $(find ./planet/benchmarks/ACAS/ -name "*.rlv" | sort | awk 'NR % 4 == 3');
do
    target=$(echo $prop | gawk 'match($0, /(property[0-9]+\/.+)\.rlv/, arr) {print "results/ACAS/grbBBoptim/" arr[1] ".txt"}')
    if [ ! -f $target ]; then
        echo "$INSTR $EXE $MAX_MEM $MAX_TIME $target $prop --use_obj_function"
        $INSTR $EXE $MAX_MEM $MAX_TIME $target $prop --use_obj_function
    fi
done

coll_idx=1
for prop in $(find ./planet/benchmarks/collisionDetection/ -name "*.rlv"| sort);
do
    target_fname=$coll_idx-$(basename $prop .rlv)
    target="results/collisionDetection/grbBBoptim/$target_fname.txt"
    if [ ! -f $target ]; then
        echo "$INSTR $EXE $MAX_MEM $MAX_TIME $target $prop --use_obj_function"
        $INSTR $EXE $MAX_MEM $MAX_TIME $target $prop --use_obj_function
    fi
    coll_idx=$(($coll_idx + 1))
done

for prop in $(find ./planet/benchmarks/PCAMNIST-1000margin/ -name "*.rlv"| sort -r | awk 'NR % 4 == 3');
do
    target_fname=$(basename $prop .rlv)
    target="results/PCAMNIST-1000margin/grbBBoptim/$target_fname.txt"
    if [ ! -f $target ]; then
        echo "$INSTR $EXE $MAX_MEM $MAX_TIME $target $prop --use_obj_function"
        $INSTR $EXE $MAX_MEM $MAX_TIME $target $prop --use_obj_function
    fi
done

#!/bin/bash

INSTR=./tools/memtime_wrapper.py
EXE=./planet/src/planet

MAX_MEM=$((20*1024*1024))
MAX_TIME=7200

trap "exit" INT
for prop in $(find ./planet/benchmarks/ACAS/ -name "*.rlv" | sort);
do
    target=$(echo $prop | gawk 'match($0, /(property[0-9]+\/.+)\.rlv/, arr) {print "results/ACAS/planet/" arr[1] ".txt"}')
    if [ ! -f $target ]; then
        echo "$INSTR $EXE $MAX_MEM $MAX_TIME $target $prop "
        $INSTR $EXE $MAX_MEM $MAX_TIME $target $prop
    fi
done

coll_idx=1
for prop in $(find ./planet/benchmarks/collisionDetection/ -name "*.rlv"| sort);
do
    target_fname=$coll_idx-$(basename $prop .rlv)
    target="results/collisionDetection/planet/$target_fname.txt"
    if [ ! -f $target ]; then
        echo "$INSTR $EXE $MAX_MEM $MAX_TIME $target $prop"
        $INSTR $EXE $MAX_MEM $MAX_TIME $target $prop
    fi
    coll_idx=$(($coll_idx + 1))
done

for prop in $(find ./planet/benchmarks/twinLadder/ -name "*.rlv"| sort);
do
    target_fname=$(basename $prop .rlv)
    target="results/twinLadder/planet/$target_fname.txt"
    if [ ! -f $target ]; then
        echo "$INSTR $EXE $MAX_MEM $MAX_TIME $target $METHOD $prop"
        $INSTR $EXE $MAX_MEM $MAX_TIME $target $METHOD $prop
    fi
done

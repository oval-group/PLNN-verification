#!/bin/bash
TARGET_DIR='./ReluplexCav2017/benchmarks/'
mkdir -p $TARGET_DIR

trap "exit" INT
COLL_DETECT_DIR_NNET=$TARGET_DIR/collisionDetectionNNet

mkdir -p $COLL_DETECT_DIR_NNET
for rlv_file in ./planet/benchmarks/collisionDetection/*.rlv;
do
    filename=$(basename $rlv_file .rlv)
    echo "./tools/pytorch2nnet.py $rlv_file $COLL_DETECT_DIR_NNET/$filename.nnet"
    ./tools/rlv_2_nnet.py $rlv_file $COLL_DETECT_DIR_NNET/$filename.nnet
done

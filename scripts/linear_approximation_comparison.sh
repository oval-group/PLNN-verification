#!/bin/bash
trap "exit" INT
TARGET_DIR=./figures/linear_approximation

mkdir -p $TARGET_DIR
./tools/compare_relaxations.py planet/benchmarks/ACAS/property1/2_7.rlv 50 0.9 $TARGET_DIR/deep_net.eps
./tools/compare_relaxations.py planet/benchmarks/collisionDetection/reluBenchmark0.0715999603271s_UNSAT.rlv 50 0.9 $TARGET_DIR/shallow_net.eps

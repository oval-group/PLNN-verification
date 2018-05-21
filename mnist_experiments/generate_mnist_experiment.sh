#!/bin/bash
target_file=$1
target_dir=$(dirname $target_file)

inp=$(echo $target_dir | gawk 'match($0, /([0-9]+)_inp/, res) {print res[1]}')
width=$(echo $target_dir | gawk 'match($0, /([0-9]+)_width/, res) {print res[1]}')
depth=$(echo $target_dir | gawk 'match($0, /([0-9]+)_depth/, res) {print res[1]}')

mkdir -p $target_dir

../tools/mnist_trainer.py $inp $width $depth $target_dir --task OddOrEven

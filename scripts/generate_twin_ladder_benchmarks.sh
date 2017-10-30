#!/bin/bash
possible_width=(5 10 25)
possible_nb_layers=(2 4 5)
possible_nb_inputs=(5 10 25)
possible_margin=(1e-2 1 10)

RLV_TARGET_DIR=planet/benchmarks/twinLadder
NNET_TARGET_DIR=ReluplexCav2017/benchmarks/twinLadder

mkdir -p $RLV_TARGET_DIR
mkdir -p $NNET_TARGET_DIR

trap "exit" INT
for width in "${possible_width[@]}"
do
    for nb_layers in "${possible_nb_layers[@]}"
    do
        for nb_input in "${possible_nb_inputs[@]}"
        do
            for margin in "${possible_margin[@]}"
            do

                target_name="twin_ladder-${nb_input}_inp-${nb_layers}_layers-${width}_width-${margin}_margin"
                layer_pattern=$(for a in `seq $nb_layers`; do echo -n "$width "; done)

                rlv_file="${RLV_TARGET_DIR}/${target_name}.rlv"
                nnet_file="${NNET_TARGET_DIR}/${target_name}.nnet"

                echo "./tools/generate_twinladder_net.py rlv $rlv_file $margin $nb_input $layer_pattern"
                ./tools/generate_twinladder_net.py rlv $rlv_file $margin $nb_input $layer_pattern
                echo "./tools/generate_twinladder_net.py nnet $nnet_file $margin $nb_input $layer_pattern"
                ./tools/generate_twinladder_net.py nnet $nnet_file $margin $nb_input $layer_pattern
            done
        done
    done
done

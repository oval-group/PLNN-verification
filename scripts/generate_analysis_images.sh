#/bin/bash
trap "exit" INT
TARGET_DIR=./figures/analysis
mkdir -p $TARGET_DIR

BAB_RES=results/twinLadder/BaB
./tools/evaluate_twinladder.py $BAB_RES inp    --output_plot $TARGET_DIR/bab_inp.eps
./tools/evaluate_twinladder.py $BAB_RES layers --output_plot $TARGET_DIR/bab_layers.eps
./tools/evaluate_twinladder.py $BAB_RES width  --output_plot $TARGET_DIR/bab_width.eps
./tools/evaluate_twinladder.py $BAB_RES margin --output_plot $TARGET_DIR/bab_margin.eps

MIP_RES=results/twinLadder/MIP
./tools/evaluate_twinladder.py $MIP_RES inp    --output_plot $TARGET_DIR/mip_inp.eps
./tools/evaluate_twinladder.py $MIP_RES layers --output_plot $TARGET_DIR/mip_layers.eps
./tools/evaluate_twinladder.py $MIP_RES width  --output_plot $TARGET_DIR/mip_width.eps
./tools/evaluate_twinladder.py $MIP_RES margin --output_plot $TARGET_DIR/mip_margin.eps

PLANET_RES=results/twinLadder/planet
./tools/evaluate_twinladder.py $PLANET_RES inp    --output_plot $TARGET_DIR/planet_inp.eps
./tools/evaluate_twinladder.py $PLANET_RES layers --output_plot $TARGET_DIR/planet_layers.eps
./tools/evaluate_twinladder.py $PLANET_RES width  --output_plot $TARGET_DIR/planet_width.eps
./tools/evaluate_twinladder.py $PLANET_RES margin --output_plot $TARGET_DIR/planet_margin.eps

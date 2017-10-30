#!/bin/bash
TARGET_DIR='./planet/benchmarks/ACAS'
mkdir -p $TARGET_DIR

trap "exit" INT
PROP1_DIR=$TARGET_DIR/property1
mkdir -p $PROP1_DIR
for net in ./ReluplexCav2017/nnet/*.nnet;
do
    net_id=$(basename $net | gawk 'match($0, /run2a_(.+)_batch/, arr) {print arr[1]}')
    ./tools/acas_2_rlv.py $net $PROP1_DIR/$net_id.rlv --property ./ReluplexCav2017/check_properties/exps_config/property1/property.rlv
done

PROP2_DIR=$TARGET_DIR/property2
mkdir -p $PROP2_DIR
for net in ./ReluplexCav2017/nnet/*.nnet;
do
    net_id=$(basename $net | gawk 'match($0, /run2a_(.+)_batch/, arr) {print arr[1]}')
    x=$(echo $net_id | gawk 'match($net_id, /([0-9])_[0-9]/, arr) {print arr[1]}')
    if [ $x -gt 1 ]; then
        ./tools/acas_2_rlv.py $net $PROP2_DIR/$net_id.rlv --property ./ReluplexCav2017/check_properties/exps_config/property2/property.rlv
    fi
done

PROP3_DIR=$TARGET_DIR/property3
mkdir -p $PROP3_DIR
for net in ./ReluplexCav2017/nnet/*.nnet;
do
    net_id=$(basename $net | gawk 'match($0, /run2a_(.+)_batch/, arr) {print arr[1]}')
    x=$(echo $net_id | gawk 'match($net_id, /([0-9])_[0-9]/, arr) {print arr[1]}')
    y=$(echo $net_id | gawk 'match($net_id, /[0-9]_([0-9])/, arr) {print arr[1]}')
    if [ $x -ne 1 ] || [ $y -lt 7 ]; then
        ./tools/acas_2_rlv.py $net $PROP3_DIR/$net_id.rlv --property ./ReluplexCav2017/check_properties/exps_config/property3/property.rlv
    fi
done

PROP4_DIR=$TARGET_DIR/property4
mkdir -p $PROP4_DIR
for net in ./ReluplexCav2017/nnet/*.nnet;
do
    net_id=$(basename $net | gawk 'match($0, /run2a_(.+)_batch/, arr) {print arr[1]}')
    x=$(echo $net_id | gawk 'match($net_id, /([0-9])_[0-9]/, arr) {print arr[1]}')
    y=$(echo $net_id | gawk 'match($net_id, /[0-9]_([0-9])/, arr) {print arr[1]}')
    if [ $x -ne 1 ] || [ $y -lt 7 ]; then
        ./tools/acas_2_rlv.py $net $PROP4_DIR/$net_id.rlv --property ./ReluplexCav2017/check_properties/exps_config/property4/property.rlv
    fi
done

PROP5_DIR=$TARGET_DIR/property5
mkdir -p $PROP5_DIR
for prop in ./ReluplexCav2017/check_properties/exps_config/property5/*.rlv
do
    prop_name=$(basename $prop)
    ./tools/acas_2_rlv.py ./ReluplexCav2017/nnet/ACASXU_run2a_1_1_batch_2000.nnet $PROP5_DIR/$prop_name --property $prop
done

PROP6_DIR=$TARGET_DIR/property6
mkdir -p $PROP6_DIR
for prop in ./ReluplexCav2017/check_properties/exps_config/property6a/*.rlv
do
    prop_name=$(basename $prop)
    ./tools/acas_2_rlv.py ./ReluplexCav2017/nnet/ACASXU_run2a_1_1_batch_2000.nnet $PROP6_DIR/6a_$prop_name --property $prop
done
for prop in ./ReluplexCav2017/check_properties/exps_config/property6b/*.rlv
do
    prop_name=$(basename $prop)
    ./tools/acas_2_rlv.py ./ReluplexCav2017/nnet/ACASXU_run2a_1_1_batch_2000.nnet $PROP6_DIR/6b_$prop_name --property $prop
done

PROP7_DIR=$TARGET_DIR/property7
mkdir -p $PROP7_DIR
for prop in ./ReluplexCav2017/check_properties/exps_config/property7/*.rlv
do
    prop_name=$(basename $prop)
    ./tools/acas_2_rlv.py ./ReluplexCav2017/nnet/ACASXU_run2a_1_9_batch_2000.nnet $PROP7_DIR/$prop_name --property $prop
done

PROP8_DIR=$TARGET_DIR/property8
mkdir -p $PROP8_DIR
./tools/acas_2_rlv.py ./ReluplexCav2017/nnet/ACASXU_run2a_2_9_batch_2000.nnet $PROP8_DIR/property.rlv --property ./ReluplexCav2017/check_properties/exps_config/property8/property.rlv

PROP9_DIR=$TARGET_DIR/property9
mkdir -p $PROP9_DIR
for prop in ./ReluplexCav2017/check_properties/exps_config/property9/*.rlv
do
    prop_name=$(basename $prop)
    ./tools/acas_2_rlv.py ./ReluplexCav2017/nnet/ACASXU_run2a_3_3_batch_2000.nnet $PROP9_DIR/$prop_name --property $prop
done

PROP10_DIR=$TARGET_DIR/property10
mkdir -p $PROP10_DIR
for prop in ./ReluplexCav2017/check_properties/exps_config/property10/*.rlv
do
    prop_name=$(basename $prop)
    ./tools/acas_2_rlv.py ./ReluplexCav2017/nnet/ACASXU_run2a_4_5_batch_2000.nnet $PROP10_DIR/$prop_name --property $prop
done

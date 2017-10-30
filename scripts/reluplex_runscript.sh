#!/bin/bash
INSTR=./tools/memtime_wrapper.py
MAX_MEM=$((20*1024*1024))
MAX_TIME=7200

trap "exit" INT

EXE=./ReluplexCav2017/check_properties/bin/property1.elf
for net in ./ReluplexCav2017/nnet/*.nnet;
do
    target=$(basename $net | gawk 'match($0, /run2a_(.+)_batch/, arr) {print "results/ACAS/reluplex/property1/" arr[1] ".txt"}')
    if [ ! -f $target ]; then
        echo "$INSTR $EXE $MAX_MEM $MAX_TIME $target $net $target.final"
        $INSTR $EXE $MAX_MEM $MAX_TIME $target $net $target.final
    fi
done

EXE=./ReluplexCav2017/check_properties/bin/property2.elf
for net in ./ReluplexCav2017/nnet/*.nnet;
do
    net_id=$(basename $net | gawk 'match($0, /run2a_(.+)_batch/, arr) {print arr[1]}')
    x=$(echo $net_id | gawk 'match($net_id, /([0-9])_[0-9]/, arr) {print arr[1]}')
    if [ $x -gt 1 ]; then
        target=$(basename $net | gawk 'match($0, /run2a_(.+)_batch/, arr) {print "results/ACAS/reluplex/property2/" arr[1] ".txt"}')
        if [ ! -f $target ]; then
            echo "$INSTR $EXE $MAX_MEM $MAX_TIME $target $net $target.final"
            $INSTR $EXE $MAX_MEM $MAX_TIME $target $net $target.final
        fi
    fi
done

EXE=./ReluplexCav2017/check_properties/bin/property3.elf
for net in ./ReluplexCav2017/nnet/*.nnet;
do
    net_id=$(basename $net | gawk 'match($0, /run2a_(.+)_batch/, arr) {print arr[1]}')
    x=$(echo $net_id | gawk 'match($net_id, /([0-9])_[0-9]/, arr) {print arr[1]}')
    y=$(echo $net_id | gawk 'match($net_id, /[0-9]_([0-9])/, arr) {print arr[1]}')
    if [ $x -ne 1 ] || [ $y -lt 7 ]; then
        target=$(basename $net | gawk 'match($0, /run2a_(.+)_batch/, arr) {print "results/ACAS/reluplex/property3/" arr[1] ".txt"}')
        if [ ! -f $target ]; then
            echo "$INSTR $EXE $MAX_MEM $MAX_TIME $target $net $target.final"
            $INSTR $EXE $MAX_MEM $MAX_TIME $target $net $target.final
        fi
    fi
done

EXE=./ReluplexCav2017/check_properties/bin/property4.elf
for net in ./ReluplexCav2017/nnet/*.nnet;
do
    net_id=$(basename $net | gawk 'match($0, /run2a_(.+)_batch/, arr) {print arr[1]}')
    x=$(echo $net_id | gawk 'match($net_id, /([0-9])_[0-9]/, arr) {print arr[1]}')
    y=$(echo $net_id | gawk 'match($net_id, /[0-9]_([0-9])/, arr) {print arr[1]}')
    if [ $x -ne 1 ] || [ $y -lt 7 ]; then
        target=$(basename $net | gawk 'match($0, /run2a_(.+)_batch/, arr) {print "results/ACAS/reluplex/property4/" arr[1] ".txt"}')
        if [ ! -f $target ]; then
            echo "$INSTR $EXE $MAX_MEM $MAX_TIME $target $net $target.final"
            $INSTR $EXE $MAX_MEM $MAX_TIME $target $net $target.final
        fi
    fi
done

EXE=./ReluplexCav2017/check_properties/bin/property5.elf
for p_idx in {0..3};
do
    pushd ReluplexCav2017
    target="../results/ACAS/reluplex/property5/property_$p_idx.txt"
    if [ ! -f $target ]; then
        echo "../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target $p_idx $target.final"
        ../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target $p_idx $target.final
    fi
    popd
done

EXE=./ReluplexCav2017/check_properties/bin/property6a.elf
for p_idx in {1..4};
do
    pushd ReluplexCav2017
    target="../results/ACAS/reluplex/property6/6a_property_$p_idx.txt"
    if [ ! -f $target ]; then
        echo "../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target $p_idx $target.final"
        ../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target $p_idx $target.final
    fi
    popd
done

EXE=./ReluplexCav2017/check_properties/bin/property6b.elf
for p_idx in {1..4};
do
    pushd ReluplexCav2017
    target="../results/ACAS/reluplex/property6/6b_property_$p_idx.txt"
    if [ ! -f $target ]; then
        echo "../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target $p_idx $target.final"
        ../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target $p_idx $target.final
    fi
    popd
done

EXE=./ReluplexCav2017/check_properties/bin/property7.elf
for p_idx in {3..4};
do
    pushd ReluplexCav2017
    target="../results/ACAS/reluplex/property7/property_$p_idx.txt"
    if [ ! -f $target ]; then
        echo "../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target $p_idx $target.final"
        ../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target $p_idx $target.final
    fi
    popd
done

EXE=./ReluplexCav2017/check_properties/bin/property8.elf
pushd ReluplexCav2017
target="../results/ACAS/reluplex/property8/property.txt"
if [ ! -f $target ]; then
echo "../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target 3 $target.final"
../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target 3 $target.final
fi
popd


EXE=./ReluplexCav2017/check_properties/bin/property9.elf
for p_idx in {0..4};
do
    pushd ReluplexCav2017
    target="../results/ACAS/reluplex/property9/property_$p_idx.txt"
    if [ $p_idx -ne 3 ] ; then
        if [ ! -f $target ]; then
            echo "../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target $p_idx $target.final"
            ../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target $p_idx $target.final
        fi
    fi
    popd
done

EXE=./ReluplexCav2017/check_properties/bin/property10.elf
for p_idx in {1..4};
do
    pushd ReluplexCav2017
    target="../results/ACAS/reluplex/property10/property_$p_idx.txt"
    if [ ! -f $target ]; then
        echo "../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target $p_idx $target.final"
        ../$INSTR ../$EXE $MAX_MEM $MAX_TIME $target $p_idx $target.final
    fi
    popd
done


EXE=./ReluplexCav2017/check_properties/bin/generic_prover.elf
coll_idx=1
for nnet_file in $(find ./ReluplexCav2017/benchmarks/collisionDetectionNNet/*.nnet| sort);
do
    target_fname=$coll_idx-$(basename $nnet_file .nnet)
    target="results/collisionDetection/reluplex/$target_fname.txt"
    if [ ! -f $target ]; then
        echo "$INSTR $EXE $MAX_MEM $MAX_TIME $target $nnet_file $target.final"
        $INSTR $EXE $MAX_MEM $MAX_TIME $target $nnet_file $target.final
    fi
    coll_idx=$(($coll_idx + 1))
done

EXE=./ReluplexCav2017/check_properties/bin/generic_prover.elf
for nnet_file in $(find ./ReluplexCav2017/benchmarks/twinLadder/*.nnet| sort);
do
    target_fname=$(basename $nnet_file .nnet)
    target="results/twinLadder/reluplex/$target_fname.txt"
    if [ ! -f $target ]; then
        echo "$INSTR $EXE $MAX_MEM $MAX_TIME $target $nnet_file $target.final"
        $INSTR $EXE $MAX_MEM $MAX_TIME $target $nnet_file $target.final
    fi
done

#!/bin/bash

LOG_PATH=$1

for f in ${LOG_PATH}/checkpoint*; do 
    for y in $f/checkpoint-*; do 
        python replay_checkpint.py $(echo $y| grep -v tune) ; 
    done ; 
done
#!/bin/bash

for n in `ls *.txt`; do
    echo ${n}
    python3 dual_ma_optuna_save.py ${n} 0.6 > ${n}.log
done

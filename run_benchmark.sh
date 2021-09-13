#!/usr/bin/env bash

for me in {0..99};
do
    echo "Running benchmark $me";
    kernprof -l benchmark.py;
    python -m line_profiler benchmark.py.lprof > benchmark_fastGaussian_${me}.txt;
    rm testing_small*;
done
#!/bin/bash
./histogram -i ./data/$1/input.ppm -o ./data/$1/run.ppm -t image -e ./data/$1/output.ppm
#cuda-gdb --args ./histogram -i ./data/$1/input.ppm -o ./data/$1/run.ppm -t image -e ./data/$1/output.ppm

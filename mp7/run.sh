#!/bin/bash
echo "./sparse -i ./data/$1/col.raw,./data/$1/row.raw,./data/$1/data.raw,./data/$1/vec.raw -o ./data/$1/run.raw -t vector -e ./data/$1/output.raw"
./sparse -i ./data/$1/col.raw,./data/$1/row.raw,./data/$1/data.raw,./data/$1/vec.raw -o ./data/$1/run.raw -t vector -e ./data/$1/output.raw

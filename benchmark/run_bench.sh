#!/bin/bash
#
# Example script to cycle through benchmark scenariouse
#

log_file="bench.log"
data_folder="/home/ubuntu/benchmark/data"

# Clean log file
echo "" > $log_file

for CUDA in "9.2" "10.1" "10.2" "11.0"
do
    for batch_size in "10" "20" "30" "40"
    do
        echo "" &>> $log_file
        echo "----------------- Cuda ${CUDA} | Batch size - ${batch_size} --------------------" &>> $log_file
        echo"" &>> $log_file

        docker run -it -v ${data_folder}:/data bert-base-uncased:pt1.7.1-cu${CUDA} --dry-run ${batch_size} --data-processors 2 &>> $log_file

    done
done
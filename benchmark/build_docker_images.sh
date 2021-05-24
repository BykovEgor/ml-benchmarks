#!/bin/bash

for i in "9.2" "10.1" "10.2" "11.0" "cpu"
do
    if [ "$i" == "cpu" ]; then
        docker build --build-arg CUDA=${i} -t bert-base-uncased:pt1.7.1-cpu -f ./bert-base-uncased.Dockerfile .
    else
        docker build --build-arg CUDA=${i} -t bert-base-uncased:pt1.7.1-cu${i} -f ./bert-base-uncased.Dockerfile .
    fi
done
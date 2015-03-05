#!/usr/bin/env bash

master_ip=10.52.1.6

./bin/spark-submit \
    --class SVMClassifier \
    --master spark://${master_ip}:7077 \
    wdai/svm/target/svm-1.0.jar \

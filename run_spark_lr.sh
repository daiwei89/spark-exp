#!/usr/bin/env bash
niter=5
mini_batch_fraction=1

master_ip=10.53.1.7
data_path=data/mllib/sample_binary_classification_data.txt
#data_path=/tank/projects/biglearning/wdai/datasets/mlr_datasets/synth_bin/lr2sp_dim100000_s10000000_nnz1000
#data_path=/l0/synth_bin/lr2sp_dim100000_s10000000_nnz1000
data_path=/l0/url_reputation/url.train
#data_path=/l0/subset20/url.20.train
#data_path=/l0/url_reputation/url.test
#data_path=/tank/projects/biglearning/wdai/datasets/mlr_datasets/url_reputation/url.train
#data_path=/tank/projects/biglearning/wdai/petuum/exp_apps/new_mlr/datasets/binary_lr_small_libsvm1.train.0
#data_path=/l0/datasets/lr2_high_dim_tiny_libsvm1.train.0
#data_path=/l0/datasets/lr2_dim1k_s100k_libsvm1.train.0
#data_path=/tank/projects/biglearning/wdai/petuum/exp_apps/new_mlr/datasets/lr2_high_dim_tiny_libsvm1.train.0
#data_path=/tank/projects/biglearning/wdai/petuum/exp_apps/new_mlr/datasets/lr2_dim100_tiny_libsvm1.train.0
#data_path=/l0/datasets/lr2_dim100_tiny_libsvm1.train.0
#data_path=/tank/projects/biglearning/wdai/petuum/exp_apps/new_mlr/datasets/lr2sp_dim2000_s10000_nnz1000
#data_path=/l0/datasets/lr2sp_dim2000_s10000_nnz1000
#data_path=/l0/datasets/lr2sp_dim20000_s1000000_nnz1000
#data_path=/tank/projects/biglearning/wdai/petuum/exp_apps/new_mlr/datasets/kdda/kdda
step_size=2

cmd="time ./bin/spark-submit \
    --class LRClassifier \
    --master spark://${master_ip}:7077 \
    --driver-memory 100G \
    wdai/lr/target/lr-1.0.jar \
    ${data_path} \
    ${step_size} \
    ${niter} \
    ${mini_batch_fraction}"
echo $cmd
eval $cmd

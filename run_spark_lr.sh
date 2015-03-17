#!/usr/bin/env bash
num_iterations=1
step_size=0.5
reg_type=L2
lambda=1e-8
data_format=LibSVM
minibatch_fraction=1

master_ip=10.52.1.2
#data_path=data/mllib/sample_binary_classification_data.txt
data_path=/l0/kddb.small
#data_path=/tank/projects/biglearning/wdai/datasets/mlr_datasets/synth_bin/lr2sp_dim100000_s10000000_nnz1000
#data_path=/l0/synth_bin/lr2sp_dim100000_s10000000_nnz1000
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

script_path=`readlink -f $0`
script_dir=`dirname $script_path`
spark_dir=`dirname $script_dir`

cmd="time $spark_dir/bin/spark-submit \
    --class LogisticRegression \
    --master spark://${master_ip}:7077 \
    --driver-memory 100G \
    $script_dir/lr/target/lr-1.0.jar \
    --numIterations ${num_iterations} \
    --stepSize ${step_size} \
    --regType ${reg_type} \
    --regParam ${lambda} \
    --minibatchFraction ${minibatch_fraction} \
    --dataFormat ${data_format} \
    ${data_path}"
echo $cmd
eval $cmd

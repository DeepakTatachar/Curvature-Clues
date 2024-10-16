data_dir=""

for seed in {0..135..1}; do
    python train_cifar10.py --data_index $seed --data_dir=$data_dir --use_seed f --suffix shadow --train_batch_size 256
done
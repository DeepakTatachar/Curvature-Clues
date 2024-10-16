data_dir=""

for seed in {0..70..1}; do
    python precompute_cifar100_scores.py --root_path=$data_dir --exp_idx=$seed --load_from_azure_blob --h 0.001 
done

for seed in {0..70..1}; do
   python precompute_scores/precompute_cifar100_mentr_scores.py --root_path=$data_dir --exp_idx=$seed --load_from_azure_blob --h 0.001 
done
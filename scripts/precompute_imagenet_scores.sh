for seed in {0..70..1}; do
    python precompute_imagenet_scores.py --exp_idx=$seed --load_from_azure_blob --h 0.001 --train_batch_size 32 
done

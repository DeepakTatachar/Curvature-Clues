data_dir=""

for seed in {0..70..1}; do
    python precompute_cifar10_scores.py --root_path=$data_dir --container_name curvature-mi-models  --model_name cifar10/cifar10_resnet18_ds"$seed"_shadow.ckpt --load_from_azure_blob --h 0.001 
done

for seed in {0..70..1}; do
    python precompute_cifar10_mentr_scores.py --root_path=$data_dir --load_from_azure_blob --model_name cifar10/cifar10_resnet18_ds"$seed"_shadow.ckpt --h 0.001
done
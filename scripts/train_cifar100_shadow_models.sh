start=0
end=130
data_dir=""
GPU=0
NUM_GPUS=3

# Run the loop
for i in $(seq $start $end); do
    if [[ $GPU < $NUM_GPUS ]]; then
        # Set the CUDA_VISIBLE_DEVICES environment variable
        export CUDA_VISIBLE_DEVICES=$GPU
        # Run the evaluation script
        python train_cifar100.py --arch=resnet18 --data_index=${i} --suffix=shadow --use_seed=f --data_dir=$data_dir --paralle=f --train_batch_size=512 &
    fi
    # Increment the GPU number
    GPU=$((GPU + 1))
    # If the current GPU is equal to the number of GPUs, wait for all the processes to finish
    if [[ $GPU == $NUM_GPUS ]]; then
        wait
        GPU=0
    fi   
done
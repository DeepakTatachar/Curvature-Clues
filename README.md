# Curvature Clues: Decoding Deep Learning Privacy with Input Loss Curvature (NeurIPS '24 Spotlight)
This is the official implementation for the paper "Curvature Clues: Decoding Deep Learning Privacy with Input Loss Curvature". Paper accepted at NeurIPS 2024 (Spotlight). If you use this github repo consider citing our work
```bibtex
@inproceedings{
    ravikumar2024curvature,
    title={Curvature Clues: Decoding Deep Learning Privacy with Input Loss Curvature},
    author={Ravikumar, Deepak and Soufleri, Efstathia and Roy, Kaushik},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=ZEVDMQ6Mu5}
}
```

Read our [paper](https://arxiv.org/abs/2407.02747).

## Environment
This code is tested and validated with python 3.9 and 3.11. To replicate our environment please use the `environment.yml` file provided by running
```
conda env update -n py3.9_curv_clues --file environment.yml
```
## Code Flow
1. **Training Shadow Models** The first step is to train shadow models whose code can be found in `./train` directory.
2. **Compute Scores** Next we compute the scores for various MIA methods, this code is found in `./precompute_scores` directory. The code uses Azure blob storage to save the scores, modifications to the code maybe needed to save locally.
3. **Results** Next we fetch the precomputed scores to get the results. The code for which are in the root directory and correspond to the notebook (i.e. `*.ipynb` files) 

Note we have released the pretrained shadow models and the scores, thus you can skip step 1 and step 2 by downloading the precomputed scores from our project page [here](https://engineering.purdue.edu/NRL/projects/curvature-clues)

## Training Shadow Models

### Setup
Create a folder ```./pretrained/<dataset name>``` and ```./pretrained/<dataset name>/temp```
i.e. 
```
mkdir pretrained
mkdir pretrained/cifar100
mkdir pretrained/cifar100/temp
```
and copy the training file under question to the root directory.

### Training CIFAR10 shadow models
To train CIFAR10 shadow models set the `data_dir` variable in `./scripts/train_cifar10_shadow_models.sh` and run
```
sh train_cifar10_shadow_models.sh
``` 

### Training CIFAR100 shadow models
To train CIFAR100 shadow models set the `data_dir` variable in `./scripts/train_cifar10_shadow_models.sh` and run
```
sh train_cifar100_shadow_models.sh
``` 

### ImageNet shadow models
For ImageNet we use pre-trained models from [Feldman and Zhang](https://github.com/google-research/heldout-influence-estimation). We provide code to convert these models to pytorch in `./train/imagenet_shadow_models` directory. Set the path to ImageNet `libdata/indexed_tfrecords.py`. 

1. Please download `imagenet_index.npz` from [Feldman and Zhang](https://github.com/google-research/heldout-influence-estimation) and place it in `build_fz_imagenet/`
2. Use the `build_imagenet.py` in the `build_fz_imagenet` directory to convert to TFRecord dataset.
3. Place the datasets in the following directory structure
4. Download the models
5. Copy the files from `./train/imagenet_shadow_models` to the root directory and run

```
python convert_imagenet_models_tf_2_torch.py --model_dir <path to where models were downloaded from Feldman and Zhang>
```

The `train` directory also has the code to train dp (`train/train_dp.py`) models, random subsets (`train/train_cifar100_random_samples.py`) and curvature subsets (`train/train_cifar100_low_curv_samples.py`) on CIFAR100.

## Compute Scores

### CIFAR100
To calculate the scores on CIFAR100 set the `data_dir` variable in `./scripts/precompute_cifar100_scores.sh` copy the files to the root directory and run
```
sh precompute_cifar100_scores.sh
```

### CIFAR10
To calculate the scores on CIFAR10 set the `data_dir` variable in `./scripts/precompute_cifar10_scores.sh` copy the files to the root directory and run
```
sh precompute_cifar10_scores.sh
```
### ImageNet
To calculate the scores on ImageNet set the copy the files to the root directory and run
```
sh precompute_imagenet_scores.sh
```

## Reproducing our Results

### Setup
To reproduce our results we have provided the assets [here](https://engineering.purdue.edu/NRL/projects/curvature-clues). 

1. Extract `"precomputed scores"` such that it has the folwing structure and set the `"precomputed_scores_dir"` in `config.json` to <path_to_score>.

    ```
    <path_to_score>/precomputed_scores/
    ├── cifar10
    ├── cifar100
    ├── cifar100_dp
    ├── cifar100_random
    ├── cifar100_top
    └── imagenet
    ```
2. Download the `CIFAR susbet indices` to <path_to_susbet_indices> and set `"subset_idxs_dir"` in config.json.

3. For Imagenet results download the imagenet models from [Feldman and Zhang](https://github.com/google-research/heldout-influence-estimation) and set <path_to_fz_imagenet_models> for `"imagenet_models_dir"` and the extracted models should follow the folder structure below.
    ```
    <path_to_fz_imagenet_models>
    └── imagenet-resnet50
        └── 0.7
            └── 0
            .   ├── aux_arrays.npz
            .   └── checkpoints
            .
            .
            └── 999
                ├── aux_arrays.npz
                └── checkpoints
    ```
4. Download [imagenet_index.npz](https://pluskid.github.io/influence-memorization/data/imagenet_index.npz) from [Feldman and Zhang's website](https://pluskid.github.io/influence-memorization/) and set the `<path_to_imagenet_index.npz>` for `"imagenet_index_dir"` in `config.json`. 

We describe the files and implementations below

| File     | Description         |
|----------|---------------------|
| `conditonal_mia_aug_cifar10.ipynb` | Provides the results for MIA attack using various methods and reproduces results from Table 1 for CIFAR10 |
| `conditonal_mia_aug_cifar100.ipynb` | Provides the results for MIA attack using various methods and reproduces results from Table 1 for CIFAR100 |
| `conditonal_mia_aug_imagenet.ipynb` | Provides the results for MIA attack using various methods and reproduces results from Table 1 for ImageNet |
| `conditonal_mia_aug_v_m_random.ipynb` | Provides the results for experiments under `Effect of Dataset Size` section of the paper when models are trained on random subsets |
| `conditonal_mia_aug_v_m_top.ipynb` | Provides the results for experiments under `Effect of Dataset Size` section of the paper when models are trained on most memorized subsets according to [Feldman and Zhang](https://github.com/google-research/heldout-influence-estimation) |
| `conditonal_mia_aug_dp.ipynb` | Provides the results for experiments under `Effect of Privacy` section of the paper |

For ease of reproducibility we have released our pretrained shadow models and precomputed scores used by the ipynb files on our [project page linked here](https://engineering.purdue.edu/NRL/projects/curvature-clues) where you can download these models and scores files. You can extract the precomputed scores files and set the locations in `config.json`.


"""
Author: Deepak Ravikumar Tatachar
Copyright © 2024 Deepak Ravikumar, Nano(Neuro) Electronics Research Lab, Purdue University

Description:
    Calculate the ImageNet curvature score by converting tf model to pytorch and using 
    the same ImageNet order and index as Feldman and Zhang[1].

Reference:
[1] Feldman, V. and Zhang, C. What neural networks memorize and why: Discovering the long tail via influence estimation. 
Advances in Neural Information Processing Systems, 33:2881-2891, 2020.
"""

import os
import multiprocessing


def main():
    import argparse
    import torch
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    from utils.str2bool import str2bool
    from utils.inference import inference
    from libdata.indexed_tfrecords import IndexedImageDataset
    from utils.load_dataset import load_dataset
    from models.torch_resnet50 import ResNet50
    import random
    import numpy as np
    import logging
    from tqdm import tqdm
    from azure_blob_storage import upload_numpy_as_blob, get_model_from_azure_blob
    import torch.nn.functional as F
    import tensorflow as tf

    parser = argparse.ArgumentParser(description="Collect Scores", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Curvature cal parameters
    parser.add_argument("--dataset", default="imagenet", type=str, help="Set dataset to use")
    parser.add_argument("--test", action="store_true", help="Calculate curvature on Test Set")
    parser.add_argument("--lr", default=0.1, type=float, help="Learning Rate")
    parser.add_argument("--test_accuracy_display", default=True, type=str2bool, help="Test after each epoch")
    parser.add_argument("--resume", default=False, type=str2bool, help="Resume training from a saved checkpoint")
    parser.add_argument("--momentum", "--m", default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, metavar="W", help="Weight decay (default: 1e-4)")
    parser.add_argument("--h", default=1e-4, type=float, help="h for curvature calculation")

    # Dataloader args
    parser.add_argument("--train_batch_size", default=512, type=int, help="Train batch size")
    parser.add_argument("--test_batch_size", default=512, type=int, help="Test batch size")
    parser.add_argument("--val_split", default=0.00, type=float, help="Fraction of training dataset split as validation")
    parser.add_argument("--augment", default=False, type=str2bool, help="Random horizontal flip and random crop")
    parser.add_argument("--padding_crop", default=4, type=int, help="Padding for random crop")
    parser.add_argument("--shuffle", default=False, type=str2bool, help="Shuffle the training dataset")
    parser.add_argument("--random_seed", default=0, type=int, help="Initializing the seed for reproducibility")
    parser.add_argument("--root_path", default="", type=str, help="Where to load the dataset from")

    # Model parameters
    parser.add_argument("--save_seed",  action="store_true", help="Save the seed")
    parser.add_argument("--use_seed", action="store_true", help="For Random initialization")
    parser.add_argument("--suffix", default="wd1", type=str, help="Appended to model name")
    parser.add_argument("--parallel", action="store_true", help="Device in  parallel")
    parser.add_argument("--dist", action="store_true", help="Use distributed computing")
    parser.add_argument("--model_save_dir", default="./pretrained/", type=str, help="Where to load the model")
    parser.add_argument("--load_from_azure_blob", action='store_true', help="Load pre trained model from azure blob storage")
    parser.add_argument("--exp_idx", default=37, type=int, help="Model Experiment Index")
    parser.add_argument("--gpu_id", default=0, type=int, help="Absolute GPU ID given by multirunner")

    global args
    args = parser.parse_args()
    args.arch = "resnet50"

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        log_dev_conf = tf.config.LogicalDeviceConfiguration(memory_limit=200)  # 100 MB

        # Apply the logical device configuration to the first GPU
        tf.config.set_logical_device_configuration(gpus[0], [log_dev_conf])

    # Reproducibility settings
    if args.use_seed:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['OMP_NUM_THREADS'] = '4'

        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            version_list = list(map(float, torch.__version__.split(".")))
            if  version_list[0] <= 1 and version_list[1] < 8: ## pytorch 1.8.0 or below
                torch.set_deterministic(True)
            else:
                torch.use_deterministic_algorithms(True)
        except:
            torch.use_deterministic_algorithms(True)

    # Setup right device to run on
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger(f"Compute Logger")
    logger.setLevel(logging.INFO)

    if not args.load_from_azure_blob:
        model_name = f"{args.dataset.lower()}_{args.arch}_{args.suffix}"
    else:
        model_name = f"{args.dataset.lower()}_{args.exp_idx}"

    handler = logging.FileHandler(os.path.join("./logs", f"score_{model_name}_zo_aug_curv_scorer_gpu9.log"))
    formatter = logging.Formatter(fmt=f"%(asctime)s %(levelname)-8s %(message)s ", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(args)
    logger.info(model_name)
    dataset = IndexedImageDataset("imagenet")

    net = ResNet50(dataset.num_classes)
    if args.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0, 1, 2])

    def modified_prediction_entropy(outs, labels):
        """
        Compute the modified prediction entropy metric for a batch of predictions.
        https://arxiv.org/pdf/2003.10595.pdf

        Parameters:
        - outs: Tensor of shape (batch_size, num_classes) representing the prediction probabilities for each class.
        - labels: Tensor of shape (batch_size,) representing the true class indices.

        Returns:
        - A tensor of shape (batch_size,) containing the modified prediction entropy for each instance in the batch.
        """
        batch_size, num_classes = outs.shape

        # Convert labels to one-hot encoding
        labels_one_hot = F.one_hot(labels, num_classes).float()

        # Compute the term for the correct labels
        correct_label_terms = -(1 - outs[range(batch_size), labels]) * torch.log(outs[range(batch_size), labels])

        # Compute the sum for the incorrect labels
        incorrect_probs = outs * (1 - labels_one_hot)  # Zero out the probabilities for the correct class
        incorrect_labels_sums = -incorrect_probs * torch.log(1 - incorrect_probs + 1e-9)  # Add a small epsilon to prevent log(0)
        incorrect_labels_sums = incorrect_labels_sums.sum(dim=1)  # Sum over all classes for each instance in the batch

        # Combine the two terms
        mentr = correct_label_terms + incorrect_labels_sums

        return mentr

    def get_rademacher(data, h):
        v = torch.randint_like(data, high=2).to(device)
        # Generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1

        v = h * (v + 1e-7)

        return v
    
    def get_curvature_for_batch_zo(
        net, 
        batch_data, 
        batch_labels,
        h=1e-3, 
        niter=10, 
        temp=1, 
        device='cpu', 
        compute_curvature=True):

        num_samples = batch_data.shape[0]
        curv = torch.zeros(num_samples).to(batch_data.device)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            logits = net(batch_data)
            loss = criterion(logits / temp, batch_labels)
            prob = torch.nn.functional.softmax(logits, 1)
            max_prob = torch.max(prob, 1)[0]
            m_entropy = modified_prediction_entropy(prob, batch_labels)
            if compute_curvature:
                for _ in range(niter):
                    v = get_rademacher(batch_data, h)
                    u = get_rademacher(batch_data, h)
                    t1 = criterion(net(batch_data + v + u), batch_labels)
                    t2 = criterion(net(batch_data - v + u), batch_labels)
                    t3 = criterion(net(batch_data + v - u), batch_labels)
                    t4 = criterion(net(batch_data - v - u), batch_labels)

                    H = (t1 - t2 - t3 + t4) / (4*h)
                    curv += torch.abs(H)

        curv_estimate = curv / niter
        return curv_estimate, loss, max_prob, m_entropy

    def add_gaussian_noise_with_seed(inputs, seed, mean=0.0, std=0.05):
        """
        Adds Gaussian noise to an input tensor, with noise generated based on a given seed.

        Parameters:
        - input: Tensor of shape [batch_size, channels, height, width].
        - seed: An integer seed for reproducible noise generation.
        - mean: Mean of the Gaussian noise.
        - std: Standard deviation of the Gaussian noise.

        Returns:
        - A tensor with the same shape as input, with Gaussian noise added.
        """
        torch.manual_seed(seed)
        # Generate Gaussian noise
        noise = torch.randn_like(inputs) * std + mean
        # Add noise to the input
        noisy_input = inputs + noise
        return noisy_input
    
    def shift_flip(inputs, shift_x, shift_y, flip=False):
        """
        Apply specified shift in pixels for both width and height.
        Optionally flip the image horizontally.

        Parameters:
        - inputs: a batch of images, tensor of shape [batch_size, channels, height, width].
        - shift_x: horizontal shift value, can be negative or positive.
        - shift_y: vertical shift value, can be negative or positive.
        - flip: a boolean indicating whether to flip the image.

        Returns:
        - Transformed inputs.
        """
        # Ensure shift values are within the expected range
        shift_x = max(min(shift_x, 4), -4)
        shift_y = max(min(shift_y, 4), -4)

        # Manually create padding based on the shift directions
        if shift_x > 0:
            pad_left, pad_right = shift_x, 0
        else:
            pad_left, pad_right = 0, -shift_x
        if shift_y > 0:
            pad_top, pad_bottom = shift_y, 0
        else:
            pad_top, pad_bottom = 0, -shift_y

        # Apply padding accordingly and crop to maintain original size
        padded = torch.nn.functional.pad(inputs, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        cropped = padded[:, :, pad_top:inputs.shape[2]+pad_top, pad_left:inputs.shape[3]+pad_left]
        
        # Apply horizontal flip if required
        if flip:
            cropped = torch.flip(cropped, dims=[3])

        return cropped

    def score_true_labels_and_save(logger, model_name, global_args):
        net.eval()
        dataset_len = 1281167
        split = 'train'
        test = False
        
        transformations = [
            (0, shift_flip, [0, 0, False], True),
            (1, shift_flip, [0, 0, True], True),
        ]

        # Too much for ImageNet
        # for g_seed in range(101):
        #     if g_seed == 0:
        #         transform = (0, lambda x: x, [], False)
        #     else:
        #         transform = (g_seed, add_gaussian_noise_with_seed, [g_seed], False)

        #     transformations.append(transform)

        for t_idx, func, func_args, compute_curvature in transformations:
            scores = torch.zeros((dataset_len))
            losses = torch.zeros_like(scores)
            probs = torch.zeros_like(scores)
            m_entropy = torch.zeros_like(scores)
            total = 0
            dataset = IndexedImageDataset("imagenet")
            for data in tqdm(dataset.iterate(split, global_args.train_batch_size, shuffle=False, augmentation=False)):
                images = data['image'].numpy().transpose(0, 3, 1, 2)
                inputs = torch.from_numpy(images)
                targets = torch.from_numpy(data['label'].numpy())
                idxs = data['index'].numpy()
                inputs, targets = inputs.cuda(), targets.cuda()
                all_args = [inputs] + func_args
                inputs = func(*all_args)
                inputs.requires_grad = True
                net.zero_grad()

                curv_estimate, loss, max_prob, m_entr = get_curvature_for_batch_zo(
                    net, 
                    inputs, 
                    targets, 
                    h=args.h, 
                    niter=10,
                    compute_curvature=compute_curvature)
                
                scores[idxs] = curv_estimate.detach().clone().cpu()
                losses[idxs] = loss.detach().clone().cpu()
                probs[idxs] = max_prob.detach().clone().cpu()
                m_entropy[idxs] = m_entr.detach().clone().cpu()

            if compute_curvature:
                scores_file_name = (
                    f"curvature_scores_zo_{model_name}_{args.h}_tid{t_idx}.pt" 
                    if not test 
                    else f"curvature_scores_zo_{model_name}_{args.h}_tid{t_idx}_test.pt"
                )

                loss_file_name = (
                    f"losses_{model_name}_{args.h}_tid{t_idx}.pt"
                    if not test
                    else f"losses_{model_name}_{args.h}_tid{t_idx}_test.pt"
                )

                prob_file_name = (
                    f"prob_{model_name}_{args.h}_tid{t_idx}.pt"
                    if not test
                    else f"prob_{model_name}_{args.h}_tid{t_idx}_test.pt"
                )

                m_entropy_file_name = (
                    f"m_entropy_{model_name}_{args.h}_tid{t_idx}.pt" 
                    if not test 
                    else f"m_entropy_{model_name}_{args.h}_tid{t_idx}_test.pt"
                )

                logger.info(f"Saving {scores_file_name}")

                blob_container = "curvature-mi-scores"
                container_dir = args.dataset.lower()
                
                upload_numpy_as_blob(blob_container, container_dir, scores_file_name, scores.numpy(), True)
                upload_numpy_as_blob(blob_container, container_dir, loss_file_name, losses.numpy(), True)
                upload_numpy_as_blob(blob_container, container_dir, prob_file_name, probs.numpy(), True)
                upload_numpy_as_blob(blob_container, container_dir, m_entropy_file_name, m_entropy.numpy(), True)
            else:
                loss_file_name = (
                    f"loss_g_{model_name}_{args.h}_tid{t_idx}.pt" 
                    if not test 
                    else f"loss_g_{model_name}_{args.h}_tid{t_idx}_test.pt"
                )

                upload_numpy_as_blob(blob_container, container_dir, loss_file_name, losses.numpy(), True)
            
        logger.info('Done')
        return

    if args.load_from_azure_blob:
        logger.info('Getting model from cloud')
        model_state = get_model_from_azure_blob(dataset=args.dataset.lower(), seed=args.exp_idx)
        logger.info('Loaded model from cloud')
    else:
        model_state = torch.load(args.model_save_dir + args.dataset.lower() + "/" + model_name + ".ckpt")

    if args.parallel:
        net.module.load_state_dict(model_state)
    else:
        net.load_state_dict(model_state)

    net.to(device)

    # Calculate various scores for different mia methods
    score_true_labels_and_save(logger, model_name, args)
    logger.info('Done')
    return


if __name__ == "__main__":
    if os.name == "nt":
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()

    main()

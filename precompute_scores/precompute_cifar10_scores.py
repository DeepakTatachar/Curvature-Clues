"""
Author: Deepak Ravikumar Tatachar
Copyright © 2024 Deepak Ravikumar, Nano(Neuro) Electronics Research Lab, Purdue University
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
    from utils.load_dataset import load_dataset
    from models.resnet_K import resnet18_k
    import random
    import numpy as np
    import logging
    from tqdm import tqdm
    from azure_blob_storage import upload_numpy_as_blob, get_model_from_azure_blob_file
    import torchopt

    parser = argparse.ArgumentParser(description="Train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Curvature cal parameters
    parser.add_argument("--dataset", default="cifar10", type=str, help="Set dataset to use")
    parser.add_argument("--test", action="store_true", help="Calculate curvature on Test Set")
    parser.add_argument("--lr", default=0.1, type=float, help="Learning Rate")
    parser.add_argument("--test_accuracy_display", default=True, type=str2bool, help="Test after each epoch")
    parser.add_argument("--resume", default=False, type=str2bool, help="Resume training from a saved checkpoint")
    parser.add_argument("--momentum", "--m", default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, metavar="W", help="Weight decay (default: 1e-4)")
    parser.add_argument("--dedup", default=False, type=str2bool, help="Use de duplication")
    parser.add_argument("--h", default=0.001, type=float, help="h for curvature calculation")

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
    parser.add_argument("--save_seed", default=False, type=str2bool, help="Save the seed")
    parser.add_argument("--use_seed", default=True, type=str2bool, help="For Random initialization")
    parser.add_argument("--suffix", default="wd1", type=str, help="Appended to model name")
    parser.add_argument("--parallel", action="store_true", help="Device in  parallel")
    parser.add_argument("--dist", action="store_true", help="Use distributed computing")
    parser.add_argument("--model_save_dir", default="./pretrained/", type=str, help="Where to load the model")
    parser.add_argument("--load_from_azure_blob", action='store_true', help="Load pre trained model from azure blob storage")
    parser.add_argument("--exp_idx", default=37, type=int, help="Model Experiment Index")
    parser.add_argument("--gpu_id", default=0, type=int, help="Absolute GPU ID given by multirunner")
    parser.add_argument("--model_name", default="", type=str, help="Model Name")
    parser.add_argument("--container_name", default="", type=str, help="Azure Container Name")

    global args
    args = parser.parse_args()
    args.arch = "resnet18_cifar10"

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

    index = torch.load("./curv_scores/data_index_cifar100.pt")
    dataset_len = len(index)

    # Use the following transform for training and testing
    dataset = load_dataset(
        dataset=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=args.val_split,
        augment=args.augment,
        padding_crop=args.padding_crop,
        shuffle=args.shuffle,
        index=index,
        root_path=args.root_path,
        random_seed=args.random_seed,
        distributed=args.dist,
    )

    net = resnet18_k(dataset.num_classes)
    if args.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0, 1, 2])
    
    def get_rademacher(data, h):
        v = torch.randint_like(data, high=2).to(device)
        # Generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1

        v = h * (v + 1e-7)

        return v
    
    def get_curvature_for_batch_zo_v2(net, batch_data, batch_labels, h=1e-3, niter=10, temp=1, device='cpu'):
        num_samples = batch_data.shape[0]
        curv = torch.zeros(num_samples).to(batch_data.device)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            logits = net(batch_data)
            loss_pos = criterion(logits / temp, batch_labels)
            prob = torch.nn.functional.softmax(logits, 1)
            max_prob = torch.max(prob, 1)[0]
            for _ in range(niter):
                
                v = get_rademacher(batch_data, h)
                u = get_rademacher(batch_data, h)
                t1 = criterion(net(batch_data + v + u), batch_labels)
                t2 = criterion(net(batch_data - v + u), batch_labels)
                t3 = criterion(net(batch_data + v - u), batch_labels)
                t4 = criterion(net(batch_data - v - u), batch_labels)

                H = (t1 - t2 -t3 + t4) / (4*h)
                curv += torch.abs(H)

        curv_estimate = curv / niter
        return curv_estimate, loss_pos, max_prob
    
    def transform_input(inputs, shift_x, shift_y, flip=False):
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

    def score_true_labels_and_save(epoch, test, logger, model_name, num_classes):
        net.eval()
        dataloader = dataset.train_loader if not test else dataset.test_loader
        transformations = [
            (0, 0, True),
            (0, 0, False),
            # (0, 4, False),
            # (-4, 0, False),
            # (0, -4, False),
            # (-4, -4, False),
            # (4, 4, True),
            # (4, 0, True),
            # (0, 4, True),
            # (-4, 0, True),
            # (0, -4, True),
        ]

        for t_idx, (shift_x, shift_y, flip) in enumerate(transformations):
            scores = torch.zeros((dataset_len))
            losses = torch.zeros_like(scores)
            probs = torch.zeros_like(scores)
            total = 0
            for inputs, targets in tqdm(dataloader):
                start_idx = total
                stop_idx = total + len(targets)
                idxs = index[start_idx:stop_idx]
                total = stop_idx

                inputs, targets = inputs.cuda(), targets.cuda()
                inputs = transform_input(inputs, shift_x, shift_y, flip)
                inputs.requires_grad = True
                net.zero_grad()

                curv_estimate, loss_pos, max_prob = get_curvature_for_batch_zo_v2(net, inputs, targets, h=args.h, niter=10)
                scores[idxs] = curv_estimate.detach().clone().cpu()
                losses[idxs] = loss_pos.detach().clone().cpu()
                probs[idxs] = max_prob.detach().clone().cpu()

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

            logger.info(f"Saving {scores_file_name}")
            save_dir = os.path.join("curv_scores", args.arch)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            blob_container = "curvature-mi-scores"
            container_dir = "cifar10"
            
            upload_numpy_as_blob(blob_container, container_dir, scores_file_name, scores.numpy(), True)
            upload_numpy_as_blob(blob_container, container_dir, loss_file_name, losses.numpy(), True)
            upload_numpy_as_blob(blob_container, container_dir, prob_file_name, probs.numpy(), True)
    
    logger.info(f"Loading model for epoch {epoch}")
    if args.load_from_azure_blob:
        # model_state = get_model_from_azure_blob(seed=args.exp_idx)
        model_state = get_model_from_azure_blob_file(args.container_name, args.model_name)
    else:
        model_state = torch.load(args.model_save_dir + args.dataset.lower() + "/" + model_name + f"{epoch}.ckpt")

    if args.parallel:
        net.module.load_state_dict(model_state)
    else:
        net.load_state_dict(model_state)

    net.to(device)

    test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.test_loader, device=device)
    logger.info(" Test set: Accuracy: {}/{} ({:.2f}%)".format(test_correct, test_total, test_accuracy))

    # Calculate curvature score
    score_true_labels_and_save(epoch, args.test, logger, args.model_name.split("/")[-1], dataset.num_classes)

    if args.dist:
        destroy_process_group()


if __name__ == "__main__":
    if os.name == "nt":
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()

    main()

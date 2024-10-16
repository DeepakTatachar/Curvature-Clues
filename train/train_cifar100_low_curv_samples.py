"""
@author: Anonymized
@copyright: Anonymized
"""

import os
import multiprocessing

def main():
    import argparse
    import torch
    from torch.distributed import init_process_group, destroy_process_group, barrier
    from torch.nn.parallel import DistributedDataParallel as DDP
    from utils.str2bool import str2bool
    from utils.inference import inference
    from utils.load_dataset import load_dataset
    from utils.averagemeter import AverageMeter
    from utils.instantiate_model import instantiate_model
    from azure_blob_storage import cloud_save, get_model_from_azure_blob_file
    import random
    import numpy as np
    import logging
    import json

    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument('--epochs',                 default=300,            type=int,       help='Set number of epochs')
    parser.add_argument('--dataset',                default='CIFAR100',     type=str,       help='Set dataset to use')
    parser.add_argument('--lr',                     default=0.1,            type=float,     help='Learning Rate')
    parser.add_argument('--test_accuracy_display',  default=True,           type=str2bool,  help='Test after each epoch')
    parser.add_argument('--resume',                 default=False,          type=str2bool,  help='Resume training from a saved checkpoint')
    parser.add_argument('--momentum', '--m',        default=0.9,            type=float,     help='Momentum')
    parser.add_argument('--weight-decay', '--wd',   default=1e-4,           type=float,     metavar='W', help='Weight decay (default: 1e-4)')
    parser.add_argument('--data_dir',               default='',
                                                                            type=str,       help='Where to load data from')
    parser.add_argument('--exp_idx',                default=1,              type=int,       help='Exp idx to save')

    # Dataloader args
    parser.add_argument('--train_batch_size',       default=256,            type=int,       help='Train batch size')
    parser.add_argument('--test_batch_size',        default=256,            type=int,       help='Test batch size')
    parser.add_argument('--val_split',              default=0.00,           type=float,     help='Fraction of training dataset split as validation')
    parser.add_argument('--augment',                default=True,           type=str2bool,  help='Random horizontal flip and random crop')
    parser.add_argument('--padding_crop',           default=4,              type=int,       help='Padding for random crop')
    parser.add_argument('--shuffle',                default=True,           type=str2bool,  help='Shuffle the training dataset')
    parser.add_argument('--random_seed',            default=0,              type=int,       help='Initializing the seed for reproducibility')
    parser.add_argument('--top_k',                  default=5000,           type=int,       help='The top K lowest curvature samples to train on')

    # Model parameters
    parser.add_argument('--save_seed',              default=False,          type=str2bool,  help='Save the seed')
    parser.add_argument('--use_seed',               default=False,          type=str2bool,  help='For Random initialization')
    parser.add_argument('--suffix',                 default='',             type=str,       help='Appended to model name')
    parser.add_argument('--parallel',               default=False,          type=str2bool,  help='Use data parallel')
    parser.add_argument('--dist',                   default=False,          type=str2bool,  help='Distributed Training')
    parser.add_argument('--model_save_dir',         default='./pretrained/',type=str,       help='Where to store the model')
    parser.add_argument('--arch',                   default='resnet18',     type=str,       help='Network architecture')
    parser.add_argument('--gpu_id',                 default=0,              type=int,       help='GPU ID from multirunner.sh')
    parser.add_argument('--reprod',                 action='store_true',                    help='Set the reproducibility setting')
    
    global args
    args = parser.parse_args()

    # Distributed setting
    if args.dist:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        global_rank = 0

    # Reproducibility settings
    if args.reprod:
        print("Enabling Reproducibility settings")
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

    # Parameters
    num_epochs = args.epochs
    args.suffix = f"{args.exp_idx}_top_{args.top_k}"

    # Setup right device to run on
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    logger = logging.getLogger(f'Train Logger')
    logger.setLevel(logging.INFO)

    model_name = args.dataset.lower() + "_" + args.arch + "_"  + args.suffix
    handler = logging.FileHandler(os.path.join('./logs', f'{args.dataset.lower()}_{model_name}_node{global_rank}.log'))
    formatter = logging.Formatter(
        fmt=f'%(asctime)s [{global_rank}] %(levelname)-8s %(message)s ',
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.info(args)

    epochs = range(1, 300, 1)
    h = 1e-3
    score_file_name = f'regr_scores_{0}_cifar100_resnet18_cifar100_wd1_{h}.pt'
    curv_scores = torch.abs(torch.load(f"path to curvature score/{score_file_name}"))

    for epoch in epochs:
        score_file_name = f'regr_scores_{epoch}_cifar100_resnet18_cifar100_wd1_{h}.pt'
        curv_scores += torch.abs(torch.load(f"path to curvature score/{score_file_name}"))

    sorted_score, indices = torch.sort(curv_scores, stable=True, descending=False)
    # np.save("./dataset_idxs/cifar100/curvature_scorted.npy", indices.numpy())
    index = indices[:args.top_k].numpy().tolist()

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
        root_path=args.data_dir,
        random_seed=args.random_seed,
        distributed=args.dist)

    # Instantiate model 
    net, model_name = instantiate_model(
        dataset=dataset,
        arch=args.arch,
        suffix=args.suffix,
        load=args.resume,
        torch_weights=False,
        device=device,
        model_args={})

    if args.use_seed:  
        if args.save_seed:
            logger.info("Saving Seed")
            torch.save(net.state_dict(),'./seed/' + args.dataset.lower() + '_' + args.arch + ".seed")
        else:
            logger.info("Loading Seed")
            net.load_state_dict(torch.load('./seed/'+ args.dataset.lower() +'_' + args.arch + ".seed"))
    else:
        logger.info("Random Initialization")

    # Optimizer
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    if args.resume:
        saved_training_state = torch.load(args.model_save_dir + args.dataset.lower()+'/temp/' + model_name  + '.temp')
        start_epoch =  saved_training_state['epoch']
        optimizer.load_state_dict(saved_training_state['optimizer'])
        net.load_state_dict(saved_training_state['model'])
        best_val_accuracy = saved_training_state['best_val_accuracy']
        best_val_loss = saved_training_state['best_val_loss']
    else:
        start_epoch = 0
        best_val_accuracy = 0.0
        best_val_loss = float('inf')

    net = net.to(device)
    if args.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0,1,2])
    if args.dist:
        net = DDP(net, device_ids=[0,1,2])

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.6*args.epochs), int(0.8*args.epochs)],
        gamma=0.1)

    # Train model
    for epoch in range(start_epoch, num_epochs, 1):
        net.train()
        train_correct = 0.0
        train_total = 0.0
        save_ckpt = False
        losses = AverageMeter('Loss', ':.4e')
        for batch_idx, (data, labels) in enumerate(dataset.train_loader):
            data = data.to(device)
            labels = labels.to(device)
            
            # Clears gradients of all the parameter tensors
            optimizer.zero_grad()
            out = net(data)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            losses.update(loss.item())

            train_correct += (out.max(-1)[1] == labels).sum().long().item()
            train_total += labels.shape[0]

            if batch_idx % 48 == 0:
                trainset_len = (1 - args.val_split) * len(dataset.train_loader.dataset)
                curr_acc = 100. * train_total / trainset_len
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                               train_total,
                                                                               trainset_len,
                                                                               curr_acc,
                                                                               losses.avg))

        train_accuracy = float(train_correct) * 100.0 / float(train_total)
        logger.info('Train Epoch: {} Accuracy : {}/{} [ {:.2f}%)]\tLoss: {:.6f}'.format(
            epoch,
            train_correct,
            train_total,
            train_accuracy,
            losses.avg))
        
        # Step the scheduler by 1 after each epoch
        scheduler.step()
        
        if args.val_split > 0.0: 
            val_correct, val_total, val_accuracy, val_loss = inference(
                net=net,
                data_loader=dataset.val_loader,
                device=device,
                loss=criterion)
            
            logger.info("Validation loss {:.4f}".format(val_loss))

            if val_loss <= best_val_loss:
                best_val_accuracy = val_accuracy 
                best_val_loss = val_loss
                save_ckpt = True
        else:
            val_correct = -1
            val_total = -1
            val_accuracy = float('inf')
            if (epoch + 1) % 10 == 0:
                save_ckpt = True

        if args.parallel:
            saved_training_state = {    'epoch'     : epoch + 1,
                                        'optimizer' : optimizer.state_dict(),
                                        'model'     : net.module.state_dict(),
                                        'best_val_accuracy' : best_val_accuracy,
                                        'best_val_loss' : best_val_loss
                                    }
        else:
            saved_training_state = {    'epoch'     : epoch + 1,
                                        'optimizer' : optimizer.state_dict(),
                                        'model'     : net.state_dict(),
                                        'best_val_accuracy' : best_val_accuracy,
                                        'best_val_loss' : best_val_loss
                                    }

        if global_rank == 0:
            # torch.save(saved_training_state, args.model_save_dir + args.dataset.lower() + '/temp/' + model_name  + '.temp')
            if args.parallel:
                cloud_save(net.module.state_dict(), model_name + '.ckpt', args.gpu_id)
            else:
                cloud_save(net.state_dict(), model_name + '.ckpt', args.gpu_id)
        
        if save_ckpt and global_rank == 0:
            logger.info("Saving checkpoint...")
            if args.parallel:
                cloud_save(net.module.state_dict(), model_name + '.ckpt', args.gpu_id)
            else:
                cloud_save(net.state_dict(), model_name + '.ckpt', args.gpu_id)
            if args.test_accuracy_display:
                # Test model
                # Set the model to eval mode
                test_correct, test_total, test_accuracy = inference(
                    net=net,
                    data_loader=dataset.test_loader,
                    device=device)

                logger.info(
                    " Training set accuracy: {}/{}({:.2f}%) \n" 
                    " Validation set accuracy: {}/{}({:.2f}%)\n"
                    " Test set: Accuracy: {}/{} ({:.2f}%)".format(
                        train_correct,
                        train_total,
                        train_accuracy,
                        val_correct,
                        val_total,
                        val_accuracy,
                        test_correct,
                        test_total,
                        test_accuracy))

    # Test model
    # Set the model to eval mode
    logger.info("End of training without reusing Validation set")
    if args.val_split > 0.0:
        logger.info('Loading the best model on validation set')
        model_state = get_model_from_azure_blob_file("curvature-mi-models", f"{args.dataset.lower()}/{model_name}.ckpt")
        if args.parallel:
            net.module.load_state_dict(model_state)
        else:
            net.load_state_dict(model_state)
        net = net.to(device)
        val_correct, val_total, val_accuracy = inference(net=net, data_loader=dataset.val_loader, device=device)
        logger.info('Validation set: Accuracy: {}/{} ({:.2f}%)'.format(val_correct, val_total, val_accuracy))
    else:
        logger.info('Saving the final model')
        if global_rank == 0:
            if args.parallel:
                cloud_save(net.module.state_dict(), model_name + '.ckpt', args.gpu_id)
            else:
                cloud_save(net.state_dict(), model_name + '.ckpt', args.gpu_id)

    test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.test_loader, device=device)
    logger.info(' Test set: Accuracy: {}/{} ({:.2f}%)'.format(test_correct, test_total, test_accuracy))

    train_correct, train_total, train_accuracy = inference(net=net, data_loader=dataset.train_loader, device=device)
    logger.info(' Train set: Accuracy: {}/{} ({:.2f}%)'.format(train_correct, train_total, train_accuracy))

    if args.dist:
        destroy_process_group()

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()

    main()
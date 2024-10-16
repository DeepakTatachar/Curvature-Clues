"""
Author: Anonymized
Copyright: Anonymized
Code to train deep learning vision classification models using differential privacy using DP-SGD
"""

import os
import multiprocessing

def main():
    import argparse
    import torch
    import logging
    from utils.str2bool import str2bool
    from utils.inference import inference
    from utils.load_dataset import load_dataset
    from utils.averagemeter import AverageMeter
    from utils.instantiate_model import instantiate_model
    from opacus.validators import ModuleValidator
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    import numpy as np
    import random

    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument('--epochs',                 default=20,              type=int,       help='Set number of epochs')
    parser.add_argument('--dataset',                default='CIFAR100',     type=str,       help='Set dataset to use')
    parser.add_argument('--lr',                     default=0.001,           type=float,     help='Learning Rate')
    parser.add_argument('--test_accuracy_display',  default=True,           type=str2bool,  help='Test after each epoch')
    parser.add_argument('--optimizer',              default='SGD',          type=str,       help='Optimizer for training')
    parser.add_argument('--loss',                   default='crossentropy', type=str,       help='Loss function for training')
    parser.add_argument('--resume',                 default=False,          type=str2bool,  help='Resume training from a saved checkpoint')
    parser.add_argument('--gpu_id',                    default=0,              type=int,       help='Which GPU to use')

    # Dataloader args
    parser.add_argument('--train_batch_size',       default=128,            type=int,       help='Train batch size')
    parser.add_argument('--test_batch_size',        default=256,            type=int,       help='Test batch size')
    parser.add_argument('--val_split',              default=0.0,            type=float,     help='Fraction of training dataset split as validation')
    parser.add_argument('--augment',                default=False,           type=str2bool,  help='Random horizontal flip and random crop')
    parser.add_argument('--padding_crop',           default=4,              type=int,       help='Padding for random crop')
    parser.add_argument('--shuffle',                default=True,           type=str2bool,  help='Shuffle the training dataset')
    parser.add_argument('--random_seed',            default=0,              type=int,       help='Initializing the seed for reproducibility')
    parser.add_argument('--data_index',             default=1,              type=int,       help='Dataset sub sample index')
    parser.add_argument('--data_dir',               default='',
                                                                            type=str,       help='Where to load data from')

    # Model parameters
    parser.add_argument('--save_seed',              default=False,          type=str2bool,  help='Save the seed')
    parser.add_argument('--use_seed',               default=False,          type=str2bool,  help='For Random initialization')
    parser.add_argument('--suffix',                 default='',             type=str,       help='Appended to model name')
    parser.add_argument('--arch',                   default='resnet18',     type=str,       help='Network architecture')

    # Differential Privacy Parameters
    parser.add_argument('--target_epsilon',         default=20,             type=float,     help='Target privacy epsilon')
    parser.add_argument('--target_delta',           default=1e-5,           type=float,     help='Target privacy delta')
    parser.add_argument('--max_norm',               default=1.0,            type=float,     help='How much clip grad')

    global args
    args = parser.parse_args()

    DELTA = args.target_delta
    MAX_PHYSICAL_BATCH_SIZE = 64

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    version_list = list(map(float, torch.__version__.split(".")[:2]))
    if  version_list[0] <= 1 and version_list[1] < 8: ## pytorch 1.8.0 or below
        torch.set_deterministic(True)
    else:
        torch.use_deterministic_algorithms(True)

    # Create a logger
    logger = logging.getLogger(f'Train Logger')
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(
        os.path.join(
        './logs', 
        f'train_dp_{args.dataset}_{args.arch}_eps_{args.target_epsilon}_{args.suffix}_{args.data_index}.log'), encoding="UTF-8")
    formatter = logging.Formatter(
        fmt=u"%(asctime)s %(levelname)-8s \t %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(args)

    # Parameters
    num_epochs = args.epochs
    learning_rate = args.lr

    # Setup right device to run on
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

    # Use the following transform for training and testing
    index = np.load(os.path.join(f"./dataset_idxs/{args.dataset.lower()}/{args.data_index}.npy"))

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
        distributed=False,
        logger=logger,
        num_workers=0)
    
    args.suffix = f"eps_{args.target_epsilon}_seed_{args.data_index}_lr_{args.lr}{args.suffix}"

    # Instantiate model 
    net, model_name = instantiate_model(
        dataset=dataset,
        arch=args.arch,
        suffix=args.suffix,
        load=args.resume,
        torch_weights=False,
        device=device,
        model_args={},
        logger=logger)

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    if args.resume:
        start_epoch = saved_training_state['epoch']
        optimizer.load_state_dict(saved_training_state['optimizer'])
        net.load_state_dict(saved_training_state['model'])
        best_val_accuracy = saved_training_state['best_val_accuracy']
        best_val_loss = saved_training_state['best_val_loss']
    else:
        start_epoch = 0
        best_val_accuracy = 0.0
        best_val_loss = float('inf')

    net = net.to(device)

    net = ModuleValidator.fix(net)
    ModuleValidator.validate(net, strict=False)
    privacy_engine = PrivacyEngine()

    # Optimizer
    optimizer = torch.optim.RMSprop(
        net.parameters(),
        lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.6*args.epochs), int(0.8*args.epochs)],
        gamma=0.1)

    net, optimizer, dataset.train_loader = privacy_engine.make_private_with_epsilon(
        module=net,
        optimizer=optimizer,
        data_loader=dataset.train_loader,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        max_grad_norm=args.max_norm,
        epochs=args.epochs,
    )

    # Train model
    for epoch in range(start_epoch, num_epochs, 1):
        net.train()
        train_correct = 0.0
        train_total = 0.0
        save = False
        losses = AverageMeter('Loss', ':.4e')
        logger.info('')
        with BatchMemoryManager(
            data_loader=dataset.train_loader, 
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for batch_idx, (data, labels) in enumerate(memory_safe_data_loader):
                data = data.to(device)
                labels = labels.to(device)
                
                # Clears gradients of all the parameter tensors
                optimizer.zero_grad()
                out = net(data)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                losses.update(loss.item())

                with torch.no_grad():
                    train_correct += (out.max(-1)[1] == labels).sum().long().item()
                    train_total += labels.shape[0]

                if (batch_idx + 1) % 100 == 0:
                    curr_acc = 100. * train_correct / train_total
                    epsilon = privacy_engine.get_epsilon(DELTA)
                    logger.info(
                        f"Train Epoch: {epoch} \t"
                        f"Loss: {losses.avg:.6f} "
                        f"Acc@1: {curr_acc:.6f} "
                        f"(ε = {epsilon:.2f}, δ = {DELTA})"
                    )
        
        train_accuracy = float(train_correct) * 100.0 / float(train_total)
        logger.info(
            'Train Epoch: {} Accuracy : {}/{} [ {:.2f}%)]\tLoss: {:.6f}'.format(
                epoch,
                train_correct,
                train_total,
                train_accuracy,
                losses.avg))
       
        # Step the scheduler by 1 after each epoch
        scheduler.step()
        
        val_correct, val_total, val_accuracy, val_loss = -1, -1, -1, -1
        val_accuracy= float('inf')
        save = True

        def save_model_with_dp_accountant(model, accountant, args, model_name):
            '''
            Save the model with the corresponding privacy accountant to be able to validate the privacy after training
            '''
            save_dict = {
                'model': model.state_dict(),
                'dp_accountant': accountant
            }

            torch.save(save_dict, './pretrained/'+ args.dataset.lower()+'/' + model_name + '_and_accountant.ckpt')

        saved_training_state = {    
            'epoch'     : epoch + 1,
            'optimizer' : optimizer.state_dict(),
            'model'     : net.state_dict(),
            'dp_accountant': privacy_engine.accountant,
            'best_val_accuracy' : best_val_accuracy,
            'best_val_loss' : best_val_loss }

        torch.save(saved_training_state, './pretrained/'+ args.dataset.lower() + '/temp/' + model_name  + '.temp')
        
        if save:
            logger.info("Saving checkpoint...")
            save_model_with_dp_accountant(net, privacy_engine.accountant, args, model_name)
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

    logger.info("End of training without reusing Validation set")
       

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()
    
    main()


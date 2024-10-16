
'''
Author: Anonymized
Copyright: Anonymized

Reference:
[1] Feldman, V. and Zhang, C. What neural networks memorize and why: Discovering the long tail via influence estimation. 
Advances in Neural Information Processing Systems, 33:2881-2891, 2020.
'''
import os
from libdata.indexed_tfrecords import IndexedImageDataset
import torch
from models.torch_resnet50 import ResNet50 as TorchResNet50
from sonnet.nets import ResNet50 as TFResNet50
import torch
from convert_tf_2_torch import load_checkpoint, copy_tf_2_torch_ResNet50
from tqdm import tqdm
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser(
    description="Calculate the ImageNet curvature score by converting tf model to pytorch", 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--model_dir", default=None, type=str, help="Where to load fz models from")

global args
args = parser.parse_args()

split = 'test'
imagenet = IndexedImageDataset(name='imagenet')
exp_idx = 55
net_py = TorchResNet50(num_classes=1000)
net_tf = TFResNet50(num_classes=1000)
checkpoint_dir = os.path.join(args.model_dir, f"{0.7}", f"{exp_idx}", "checkpoints")
ckpt_list = glob.glob(os.path.join(checkpoint_dir, "ckpt-*.index"))
ckpt_path = ckpt_list[0][:-6]
load_results = load_checkpoint(net_tf, checkpoint_dir)
copy_tf_2_torch_ResNet50(net_tf, net_py)

device = 'cuda'
net_py.to(device)
net_py.eval()
correct_all = []
index_all = []
batch_size = 32
net_py = torch.nn.DataParallel(net_py, device_ids=[0,1,2])
with torch.no_grad():
    for inputs in tqdm(
        imagenet.iterate(split, batch_size, shuffle=False, augmentation=False),
        total=int(imagenet.get_num_examples(split) / batch_size)):
        
        images = inputs['image'].numpy().transpose(0, 3, 1, 2)
        images = torch.from_numpy(images).to(device)
        labels = torch.from_numpy(inputs['label'].numpy()).to(device)
        pred = net_py(images)
        correct = torch.eq(torch.argmax(pred, 1), labels)
        correct_all.extend(correct.cpu().numpy().tolist())
        index_all.append(inputs['index'].numpy())

correct_all = np.array(correct_all)
index_all = np.concatenate(index_all, axis=0)

print(f"Torch Acc: {sum(correct_all) / len(correct_all)}")

array = np.load(os.path.join(args.model_dir, f"{0.7}/{exp_idx}/aux_arrays.npz"), allow_pickle=True)
subset_idx = array["index_train"]
correct_test = array['correctness_test']
acc_test = sum(correct_test) / len(correct_test)
print(f"FZ Acc: {sum(correct_test)} / {len(correct_test)} = {acc_test}")
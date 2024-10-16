'''
@author: Anonymized
@copyright: Anonymized
Description:
    Code to copy tensorflow model weights to pytorch model. Used by other code in the repo.
'''

import os
import glob
import torch
import tensorflow as tf
from sonnet import Conv2D, BatchNorm, Linear

def load_checkpoint(model, checkpoint_dir):
    """
        This function 'load_checkpoint' is borrowed from Feldman and Zhang's repo:
        https://github.com/google-research/heldout-influence-estimation/blob/master/demo.py
        See line 45 onwards.

        Reference:
        [1] Feldman, V. and Zhang, C. 
        What neural networks memorize and why: Discovering the long tail via influence estimation. 
        Advances in Neural Information Processing Systems, 33:2881-2891, 2020.
        
        Loads the latest checkpoint.
    """
    v_epoch = tf.Variable(0, dtype=tf.int64, name="epoch", trainable=False)
    v_gs = tf.Variable(0, dtype=tf.int64, name="global_step", trainable=False)
    checkpoint = tf.train.Checkpoint(model=model, epoch=v_epoch, global_step=v_gs)

    ckpt_list = glob.glob(os.path.join(checkpoint_dir, "ckpt-*.index"))
    assert len(ckpt_list) == 1
    print(ckpt_list)
    ckpt_path = ckpt_list[0][:-6]
    checkpoint.restore(ckpt_path).expect_partial()
    return dict(epoch=int(v_epoch.numpy()), global_step=int(v_gs.numpy()), path=ckpt_path)


def copy_tf_2_torch(net_tf, net_py, input_shape=(1, 32, 32, 3)):
    inp = tf.zeros(shape=input_shape)
    _ = net_tf(inp, is_training=False)
    for name, param in net_py.named_parameters():
        name_orig = name
        name = ".".join(name.split(".")[:-1])
        x = net_tf

        index = None
        next_name = None
        if "_body" in name:
            name = name.split(".")
            index = int(name[2])
            next_name = ".".join(name[3:])
            name = name[:1] + ["submodules"]
            name = ".".join(name)

        if index is None:
            for attr_name in name.split("."):
                x = getattr(x, attr_name)

        else:
            for attr_name in name.split("."):
                x = getattr(x, attr_name)

            x = x[index]
            for attr_name in next_name.split("."):
                x = getattr(x, attr_name)

        if attr_name == "bn":
            bn = x
        elif attr_name == "conv":
            x = x.w.numpy().transpose(3, 2, 0, 1)
        else:
            if "weight" in name_orig:
                x = x.w.numpy().T
            else:
                x = x.b.numpy()

        if attr_name != "bn" and x.shape != param.data.shape:
            raise Exception

        if attr_name == "bn":
            if "weight" in name_orig:
                x = bn.scale.numpy()
            else:
                x = bn.offset.numpy()

            name = ".".join(name_orig.split(".")[:-1])
            py_bn = net_py
            for attr_name in name.split("."):
                py_bn = getattr(py_bn, attr_name)

            py_bn.running_mean = torch.from_numpy(bn.moving_mean.value.numpy()).squeeze()
            py_bn.running_var = torch.from_numpy(bn.moving_variance.value.numpy()).squeeze()
            py_bn.momentum = 1 - bn.moving_mean._decay
            py_bn.num_batches_tracked.data = torch.Tensor(bn.moving_mean._counter.numpy())

        param.data = torch.from_numpy(x).to(param.data.device)

def get_object_for_param(name, module):
    name = name.split(".")
    if len(name) > 2: 
        sub_name = '.'.join(name[2:])
        sub_module = getattr(module, name[0])
        
        return get_object_for_param(sub_name, sub_module[int(name[1])])
    elif len(name) == 1:
        return module
    else:
        if name[0].isdigit():
            return module[int(name[0])]
        
        return getattr(module, name[0])

def copy_tf_2_torch_ResNet50(net_tf, net_py, input_shape=(2, 224, 224, 3)):
    inp = tf.zeros(shape=input_shape)
    _ = net_tf(inp, is_training=False)
    for name, param in net_py.named_parameters():
        tf_layer = get_object_for_param(name, net_tf)
        py_layer = get_object_for_param(name, net_py)

        if isinstance(tf_layer, Conv2D):
            if 'weight' in name:
                tf_model_weights = tf_layer.w.numpy().transpose(3, 2, 0, 1)
            else:
                raise NotImplementedError

        elif isinstance(tf_layer, BatchNorm):
            if 'weight' in name:
                tf_model_weights = tf_layer.scale.numpy()
            if 'bias' in name:
                tf_model_weights = tf_layer.offset.numpy()

            py_layer.running_mean = torch.from_numpy(tf_layer.moving_mean.value.numpy()).squeeze()
            py_layer.running_var = torch.from_numpy(tf_layer.moving_variance.value.numpy()).squeeze()
            py_layer.momentum = 1 - tf_layer.moving_mean._decay
            py_layer.num_batches_tracked.data = torch.Tensor(tf_layer.moving_mean._counter.numpy())
        elif isinstance(tf_layer, Linear):
            if "weight" in name:
                tf_model_weights = tf_layer.w.numpy().T
            if 'bias' in name:
                tf_model_weights = tf_layer.b.numpy()
        else:
            raise NotImplementedError

        tf_weights = torch.from_numpy(tf_model_weights).to(param.data.device)
        assert param.data.shape == tf_weights.shape
        param.data = tf_weights
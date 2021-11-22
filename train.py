import argparse
import json
import os

import mindspore.common.initializer as weight_init
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model

from config import config
from src.dataset import create_dataset as create_dataset
from src.lr_generator import warmup_cosine_annealing_lr
from src.resnet import resnet101 as resnet

set_seed(1)


class LossCallBack(LossMonitor):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    """

    def __init__(self, has_trained_epoch=0):
        super(LossCallBack, self).__init__()
        self.has_trained_epoch = has_trained_epoch

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num + int(self.has_trained_epoch),
                                                      cur_step_in_epoch, loss), flush=True)


def init_weight(net):
    """init_weight"""
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))


def init_group_params(net):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params


def train_net():
    """train net"""
    # init context
    print('init context')
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
    context.set_context(enable_graph_kernel=True)
    context.set_context(graph_kernel_flags="--enable_parallel_fusion --enable_expand_ops=Conv2D")
    # make dataset rp2k
    print('make dataset rp2k')
    dataset = create_dataset(dataset_path=config.train_data_path, do_train=True, repeat_num=1,
                             batch_size=config.batch_size,
                             train_image_size=config.train_image_size, eval_image_size=config.eval_image_size,
                             class_indexing=json.loads(open(config.class_indexing, 'r', encoding='utf-8').read()))
    step_size = dataset.get_dataset_size()
    # init net
    print('init net')
    net = resnet(class_num=config.class_num)
    init_weight(net=net)
    lr = Tensor(warmup_cosine_annealing_lr(config.lr, step_size, config.warmup_epochs, config.epoch_size))
    # define opt
    print('define opt')
    group_params = init_group_params(net)
    opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    metrics = {"acc"}
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                  amp_level="O2", boost_level=config.boost_mode, keep_batchnorm_fp32=False)
    # define callbacks
    print('define callbacks')
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossCallBack(0)
    cb = [time_cb, loss_cb]
    ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path)
    # save checkpoint
    print('save checkpoint')
    ckpt_append_info = [{"epoch_num": 0, "step_num": config.has_trained_step}]
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                 keep_checkpoint_max=config.keep_checkpoint_max,
                                 append_info=ckpt_append_info)
    ckpt_cb = ModelCheckpoint(prefix="resnet_rp2k_", directory=ckpt_save_dir, config=config_ck)
    cb += [ckpt_cb]
    # train model
    print('train model')
    model.train(config.epoch_size, dataset, callbacks=cb, sink_size=dataset.get_dataset_size(), dataset_sink_mode=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="enter the path of workspace")
    parser.add_argument("-w", "--workspace", default=os.getcwd())
    args = parser.parse_args()
    workspace = args.workspace
    if os.getcwd() != workspace:
        os.chdir(workspace)
    train_net()

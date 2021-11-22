import argparse
import json
import os

from mindspore import context
from mindspore.common import set_seed
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from config import config
from src.dataset import create_dataset
from src.resnet import resnet101 as resnet

set_seed(1)


def eval_net(checkpoint_file_path):
    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
    # create dataset
    dataset = create_dataset(dataset_path=config.eval_data_path, do_train=False,
                             batch_size=config.batch_size, eval_image_size=config.eval_image_size,
                             class_indexing=json.loads(open(config.class_indexing, 'r', encoding='utf-8').read()))
    # define net
    net = resnet(class_num=config.class_num)
    # load checkpoint
    param_dict = load_checkpoint(checkpoint_file_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    # define loss, model
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # define model
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy'})
    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", checkpoint_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="enter the path of checkpoint file and workspace")
    parser.add_argument("-cfp", "--checkpoint_file_path")
    parser.add_argument("-w", "--workspace", default=os.getcwd())
    args = parser.parse_args()
    checkpoint_file_path = args.checkpoint_file_path
    workspace = args.workspace
    if os.getcwd() != workspace:
        os.chdir(workspace)
    eval_net(checkpoint_file_path)

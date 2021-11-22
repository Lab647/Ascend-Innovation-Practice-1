import argparse

import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from config import config
from src.resnet import resnet101 as resnet

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


def run_export(checkpoint_file_path, file_name, file_format="ONNX"):
    net = resnet(config.class_num)
    assert checkpoint_file_path is not None, "checkpoint_path is None."
    param_dict = load_checkpoint(checkpoint_file_path)
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.zeros([config.batch_size, 3, config.height, config.width], np.float32))
    export(net, input_arr, file_name=file_name, file_format=file_format)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="enter the path of checkpoint file and name of export file")
    parser.add_argument("-cfp", "--checkpoint_file_path")
    parser.add_argument("-fn", "--file_name")
    args = parser.parse_args()
    checkpoint_file_path = args.checkpoint_file_path
    file_name = args.file_name
    run_export(checkpoint_file_path, file_name)

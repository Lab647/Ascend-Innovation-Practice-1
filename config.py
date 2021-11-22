from easydict import EasyDict as D

config = D({
    'device_target': "GPU",
    'optimizer': "Momentum",
    'lr': 0.1,
    'epoch_size': 120,
    'weight_decay': 0.0001,
    'batch_size': 16,
    'train_image_size': 512,
    'eval_image_size': 512,
    'class_num': 2388,
    'momentum': 0.9,
    'loss_scale': 1024,
    'boost_mode': 'O0',
    'has_trained_step': 0,
    'save_checkpoint_epochs': 5,
    'keep_checkpoint_max': 10,
    'warmup_epochs': 0,
    'height': 512,
    'width': 512,
    'output_path': './output/',  # 输出路径
    'checkpoint_path': './checkpoint/',  # 检查点路径（位于输出路径下）
    'train_data_path': '/opt/data/private/workspace/eamon/resnet101/src/rp2k/raw/train',  # 训练集数据位置
    'eval_data_path': '/opt/data/private/workspace/eamon/resnet101/src/rp2k/raw/test',  # 测试集数据位置
    'class_indexing': '/opt/data/private/workspace/eamon/resnet101/src/rp2k/class_index.json',  # class indexing 文件位置
})


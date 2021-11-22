# [Ascend Innovation Practice 1](https://github.com/Lab647/Ascend-Innovation-Practice-1)

- 比赛链接：https://www.datafountain.cn/competitions/528
- 模型：ResNet-101 [paper](https://arxiv.org/pdf/1512.03385)
- 数据集：[PR2K](https://gas.graviti.cn/dataset/holger/RP2K) （零售商品数据集），总共有2388个类别；其中训练集344854张图片，验证集39457张图片
- 训练：`$ python train.py [--workspace /path/to/workspace_folder]    # […]  表示可选参数`
- 验证：`$ python eval.py -cfp /path/to/ckpt_file [--workspace /path/to/workspace_folder]`
- 参考：
  - https://gitee.com/mindspore/models/tree/master/official/cv/resnet
  - https://github.com/shuokay/resnet

> 本次提交的代码，仅支持使用GPU加速（单卡）的训练，如使用Ascend需对代码做简单的适配。
> 
> 另，单卡训练的时间可能会超过72h，需要使用cut.py脚本对数据集进行切分。

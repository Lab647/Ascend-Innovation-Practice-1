import multiprocessing

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C


def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, train_image_size=512, eval_image_size=512,
                   class_indexing=None):
    print('create_dataset ImageFolderDataset')
    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=get_num_parallel_workers(8), shuffle=True,
                                     class_indexing=class_indexing)
    print('create_dataset ImageFolderDataset after')
    mean = [0.4716334 * 255,  0.42496682 * 255, 0.35393123 * 255]
    std = [0.21809808 * 255, 0.20809866 * 255, 0.19378628 * 255]
    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(600),
            C.CenterCrop(eval_image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    type_cast_op = C2.TypeCast(mstype.int32)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=get_num_parallel_workers(8))
    print('create_dataset map1')

    data_set = data_set.map(operations=type_cast_op, input_columns="label",
                            num_parallel_workers=get_num_parallel_workers(8))
    print('create_dataset map1')

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)
    print('create_dataset finish')

    return data_set


def get_num_parallel_workers(num_parallel_workers):
    """
    Get num_parallel_workers used in dataset operations.
    If num_parallel_workers > the real CPU cores number, set num_parallel_workers = the real CPU cores number.
    """
    cores = multiprocessing.cpu_count()
    if isinstance(num_parallel_workers, int):
        if cores < num_parallel_workers:
            print("The num_parallel_workers {} is set too large, now set it {}".format(num_parallel_workers, cores))
            num_parallel_workers = cores
    else:
        print("The num_parallel_workers {} is invalid, now set it {}".format(num_parallel_workers, min(cores, 8)))
        num_parallel_workers = min(cores, 8)
    return num_parallel_workers

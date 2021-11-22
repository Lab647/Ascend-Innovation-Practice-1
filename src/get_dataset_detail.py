import json
import os
from io import BytesIO

import cv2
import numpy
from PIL import Image

os.chdir('/opt/data/private/workspace/eamon/resnet101/src/rp2k/')


def get_mean_std(folder):
    means = [0, 0, 0]
    std = [0, 0, 0]
    count = 0
    for path, dirs, fs in os.walk(folder):
        for f in fs:
            file_path = os.path.join(path, f)
            print(file_path)
            # calculate mean and std
            img = cv2.imread(file_path)
            img = numpy.asarray(img)
            img = img.astype(numpy.float32) / 255.0
            count += 1
            for i in range(3):
                means[i] += img[:, :, i].mean()
                std[i] += img[:, :, i].std()
    means.reverse()
    std.reverse()
    means = numpy.asarray(means) / count
    std = numpy.asarray(std) / count
    return means, std


mean, std = get_mean_std(dir_type)
print('mean =', mean)
print('std =', std)
# mean = [0.4716334,  0.42496682, 0.35393123]
# std = [0.21809808, 0.20809866, 0.19378628]

with open('class_index.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps({v: i for i, v in enumerate(os.listdir('raw/train'))}))
with open('class_index.json', 'r', encoding='utf-8') as f:
    print(json.loads(f.read()))

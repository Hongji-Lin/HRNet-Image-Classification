# encoding: utf-8
# @author: Evan
# @file: cutout.py
# @time: 2022/11/8 16:02
# @desc:
import torch
import numpy as np

"""
cutout采用的操作是随机裁剪掉图像中的一块正方形区域，并在原图中补0。
由于作者在cutout早期版本中使用了不规则大小区域的方式，但是对比发现，固定大小区域能达到同等的效果，因此就没必要这么麻烦去生成不规则区域了。

n_holes：表示裁剪掉的图像块的数目，默认都是设置为1；
length：每个正方形块的边长，不同数据集最优设置不同，CIFAR10为16，CIFAR100为8，SVHN为20；
# 这里觉得挺麻烦的，cutout调参很重要
"""
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

"""
看看在图像上cutout是什么效果，代码如下：
import cv2
from torchvision import transforms
from cutout import Cutout

# 执行cutout
img = cv2.imread('cat.png')
img = transforms.ToTensor()(img)
cut = Cutout(length=100)
img = cut(img)

# cutout图像写入本地
img = img.mul(255).byte()
img = img.numpy().transpose((1, 2, 0))
cv2.imwrite('cutout.png', img)

"""

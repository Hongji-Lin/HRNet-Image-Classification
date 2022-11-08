# encoding: utf-8
# @author: Evan
# @file: data_augment.py
# @time: 2022/11/7 14:06
# @desc: 数据增强
import math
from random import random

import cv2
import numpy as np
from utils.general import resample_segments, segment2box

# 颜色变化 = HSV + 噪声
# HSV变换
# 调用函数的文件位置：文件位置：utils/datasets.py
# 色域空间增强Augment colorspace：H色调、S饱和度、V亮度
# 通过一些随机值改变hsv，实现数据增强
def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

    return im_hsv


def Gaussian_noise(image):
    h,w,c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0,20,3)
            b = image[row,col,0]
            g = image[row,col,1]
            r = image[row,col,2]
            image[row,col,0] = np.clip(b+s[0],0,255)
            image[row,col,1] = np.clip(g+s[1],0,255)
            image[row,col,2] = np.clip(r+s[2],0,255)
    return image



# 随机旋转、平移、缩放、裁剪，错切/非垂直投影 、透视变换（从0开始）
# 调用函数地址：utils/datasets.py
# random_perspective Augment  随机透视变换 [1280, 1280, 3] => [640, 640, 3]
# 对mosaic整合后的图片进行随机旋转、平移、缩放、裁剪，透视变换，并resize为输入大小img_size
# 被调用的函数地址：utils/augmentations.py
def random_perspective(im, targets=(), segments=(),
                       degrees=10,  # 旋转角度
                       translate=.1,  # 平移
                       scale=.1,  # 缩放
                       shear=10,  # 错切/非垂直投影
                       perspective=0.0, # 透视变换
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective # 透视变换
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale 旋转+缩放
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear 错切/非垂直投影
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation 平移
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    # 将所有变换矩阵连乘得到最终的变换矩阵
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    # n = len(targets)
    # if n:
    #     use_segments = any(x.any() for x in segments)
    #     new = np.zeros((n, 4))
    #     if use_segments:  # warp segments
    #         segments = resample_segments(segments)  # upsample
    #         # 其中 segment.shape = [n, 2], 表示物体轮廓各个坐标点
    #         for i, segment in enumerate(segments):
    #             xy = np.ones((len(segment), 3))
    #             xy[:, :2] = segment
    #             xy = xy @ M.T  # transform 应用旋转矩阵
    #             xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine
    #
    #             # clip
    #             new[i] = segment2box(xy, width, height)
    #
    #     else:  # warp boxes 如果是box坐标, 这里targets每行为[x1,y1,x2,y2],n为行数,表示目标边框个数：
    #         xy = np.ones((n * 4, 3))
    #         xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    #         xy = xy @ M.T  # transform 应用旋转矩阵
    #         # 如果透视变换参数perspective不为0， 就需要做rescale，透视变换参数为0, 则无需做rescale。
    #         xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine
    #
    #         # create new boxes
    #         x = xy[:, [0, 2, 4, 6]]
    #         y = xy[:, [1, 3, 5, 7]]
    #         new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
    #
    #         # clip 将坐标clip到[0, width],[0,height]区间内
    #         new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
    #         new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
    #
    #     # filter candidates 进一步过滤,留下那些w,h>2,宽高比<20,变换后面积比之前比>0.1的那些xy
    #     i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
    #     targets = targets[i]
    #     targets[:, 1:5] = new[i]

    return im, targets


# 图像相互融合
# 调用函数地址：utils/datasets.py
def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels



# 垂直翻转
def Vertical(image):
    return cv2.flip(image, 0, dst=None)


def TestDir():
    root_path = "data/xxx"
    save_path = root_path
    for a, b, c in os.walk(root_path):
        for file_i in c:
            file_i_path = os.path.join(a, file_i)
            print(file_i_path)
            img_i = cv2.imread(file_i_path)

            img_scale = Scale(img_i, 1.5)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_scale.jpg"), img_scale)

            img_noise = Gaussian_noise(img_i)

            img_blur = cv2.GaussianBulr(img_i, (5, 5), 0)



if __name__ == "__main__":
    TestOneDir()
    AllDate()
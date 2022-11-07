# encoding: utf-8
# @author: Evan
# @file: datasets.py
# @time: 2022/11/7 15:17
# @desc: datasets


from random import random

import cv2
import numpy as np

from imagenet.imageSets.utils.general import xywhn2xyxy, xyn2xy


def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.imgs[i]
    if im is None:  # not cached in ram
        npy = self.img_npy[i]
        if npy and npy.exists():  # load npy
            im = np.load(npy)
        else:  # read image
            path = self.img_files[i]
            im = cv2.imread(path)  # BGR
            assert im is not None, f'Image Not Found {path}'
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    else:
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized

    # 代码位置：utils/datasets.py


def load_mosaic(self, index):
    """
    用在LoadImagesAndLabels模块的__getitem__函数 进行mosaic数据增强
    将四张图片拼接在一张马赛克图像中  loads images in a 4-mosaic
    param index: 需要获取的图像索引
    return: img4: mosaic和随机透视变换后的一张图片  numpy(640, 640, 3)
    labels4: img4对应的target  [M, cls+x1y1x2y2]
    """
    # labels4: 用于存放拼接图像（4张图拼成一张）的label信息(不包含segments多边形)
    # segments4: 用于存放拼接图像（4张图拼成一张）的label信息(包含segments多边形)
    labels4, segments4 = [], []
    s = self.img_size  # 一般的图片大小
    # 随机初始化拼接图像的中心点坐标  [0, s*2]之间随机取2个数作为拼接图像的中心坐标
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
    # 从dataset中随机寻找额外的三张图像进行拼接 [14, 26, 2, 16] 再随机选三张图片的index
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    # 遍历四张图像进行拼接 4张不同大小的图像 => 1张[1472, 1472, 3]的图像
    for i, index in enumerate(indices):
        # load image   每次拿一张图片 并将这张图片resize到self.size(h,w)
        img, _, (h, w) = self.load_image(index)

        # place img in img4
        if i == 0:  # top left  原图[375, 500, 3] load_image->[552, 736, 3]   hwc
            # 创建马赛克图像 [1472, 1472, 3]=[h, w, c]
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)   w=736  h = 552  马赛克图像：(x1a,y1a)左上角 (x2a,y2a)右下角
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)  图像：(x1b,y1b)左上角 (x2b,y2b)右下角
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 将截取的图像区域填充到马赛克图像的相应位置   img4[h, w, c]
        # 将图像img的【(x1b,y1b)左上角 (x2b,y2b)右下角】区域截取出来填充到马赛克图像的【(x1a,y1a)左上角 (x2a,y2a)右下角】区域
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # 计算pad(当前图像边界与马赛克边界的距离，越界的情况padw/padh为负值)  用于后面的label映射
        padw = x1a - x1b  # 当前图像与马赛克图像在w维度上相差多少
        padh = y1a - y1b  # 当前图像与马赛克图像在h维度上相差多少

        # labels: 获取对应拼接图像的所有正常label信息(如果有segments多边形会被转化为矩形label)
        # segments: 获取对应拼接图像的所有不正常label信息(包含segments多边形也包含正常gt)
        # 在新图中更新坐标值
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)  # 更新labels4
        segments4.extend(segments)  # 更新segments4

    # Concat/clip labels4 把labels4（[(2, 5), (1, 5), (3, 5), (1, 5)] => (7, 5)）压缩到一起
    labels4 = np.concatenate(labels4, 0)
    # 防止越界  label[:, 1:]中的所有元素的值（位置信息）必须在[0, 2*s]之间,小于0就令其等于0,大于2*s就等于2*s   out: 返回
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    return img4, labels4






# HSV变换
augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])





# 随机旋转、平移、缩放、裁剪，错切/非垂直投影 、透视变换（从0开始）
img4, labels4 = random_perspective(img4, labels4, segments4,
                                   degrees=self.hyp['degrees'],  # 旋转
                                   translate=self.hyp['translate'],  # 平移
                                   scale=self.hyp['scale'],  # 缩放
                                   shear=self.hyp['shear'],  # 错切/非垂直投影
                                   perspective=self.hyp['perspective'],  # 透视变换
                                   border=self.mosaic_border)  # border to remove

# 图像相互融合
if random.random() < hyp['mixup']:  # hyp['mixup']=0 默认为0则关闭 默认为1则100%打开
    # *load_mosaic(self, random.randint(0, self.n - 1)) 随机从数据集中任选一张图片和本张图片进行mixup数据增强
    # img:   两张图片融合之后的图片 numpy (640, 640, 3)
    # labels: 两张图片融合之后的标签label [M+N, cls+x1y1x2y2]
    img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))
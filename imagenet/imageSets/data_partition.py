# encoding: utf-8
# @author: Evan
# @file: data_partition.py
# @time: 2022/11/2 9:28
# @desc: 数据集的生成和读取

import os, random, shutil

def moveimg(full_fileDir, empty_fileDir, valDir):
    full_pathDir = os.listdir(full_fileDir)  # 取full图片的原始路径
    full_filenumber = len(full_pathDir)
    empty_pathDir = os.listdir(empty_fileDir)  # 取empty图片的原始路径
    empty_filenumber = len(empty_pathDir)

    rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    full_picknumber = int(full_filenumber * rate)  # 按照rate比例从full文件夹中取一定数量图片
    empty_picknumber = int(empty_filenumber * rate) # 按照rate比例从emoty文件夹中取一定数量图片
    full_sample = random.sample(full_pathDir, full_picknumber)  # 随机选取full数量的样本图片
    empty_sample = random.sample(empty_pathDir, empty_picknumber)  # 随机选取empty数量的样本图片

    for name in full_sample:
        shutil.move(full_fileDir + name, tarDir + "\\" + name)
    return


def movelabel(file_list, file_label_train, file_label_val):
    for i in file_list:
        if i.endswith('.jpg'):
            # filename = file_label_train + "\\" + i[:-4] + '.xml'  # 可以改成xml文件将’.txt‘改成'.xml'就可以了
            filename = file_label_train + "\\" + i[:-4] + '.txt'  # 可以改成xml文件将’.txt‘改成'.xml'就可以了
            if os.path.exists(filename):
                shutil.move(filename, file_label_val)
                print(i + "处理成功！")


if __name__ == '__main__':
    fileDir = r"E:\NEt\yolov5-hat\VOCdevkit\images\train" + "\\"  # 源图片文件夹路径
    tarDir = r'E:\NEt\yolov5-hat\VOCdevkit\images\val'  # 图片移动到新的文件夹路径
    moveimg(fileDir, tarDir)
    file_list = os.listdir(tarDir)
    file_label_train = r"E:\NEt\yolov5-hat\VOCdevkit\labels\train"  # 源图片标签路径
    file_label_val = r"E:\NEt\yolov5-hat\VOCdevkit\labels\val"  # 标签
    # 移动到新的文件路径
    movelabel(file_list, file_label_train, file_label_val)

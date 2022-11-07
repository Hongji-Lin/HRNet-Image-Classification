# encoding: utf-8
# @author: Evan
# @file: data_partition.py
# @time: 2022/11/2 9:28
# @desc: 数据集的划分

import os, random, shutil



def move_img(full_fileDir, empty_fileDir, full_valDir, empty_valDir):
    reful_list = os.listdir(full_valDir)
    reemp_list = os.listdir(empty_valDir)

    if reful_list is None:
        exit(0)
    else:
        for name in reful_list:
            shutil.move(full_valDir + name, full_fileDir)

    if reemp_list is None:
        exit(0)
    else:
        for name in reemp_list:
            shutil.move(empty_valDir + name, empty_fileDir)
    # return  # 本地调试时要return(清理照片，val保持空的上传），放到服务器上不需要

    full_list = os.listdir(full_fileDir)  # 取full图片的原始路径
    full_total = len(full_list)
    empty_list = os.listdir(empty_fileDir)  # 取empty图片的原始路径
    empty_total = len(empty_list)

    rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    full_nums = int(full_total * rate)  # 按照rate比例从full文件夹中取一定数量图片
    empty_nums = int(empty_total * rate)  # 按照rate比例从empty文件夹中取一定数量图片
    full_sample = random.sample(full_list, full_nums)  # 随机选取full数量的样本图片
    empty_sample = random.sample(empty_list, empty_nums)  # 随机选取empty数量的样本图片

    print("按照train:val = 0.9:0.1的比例划分：")
    print("train集中full的数量为：{}：".format(full_total - full_nums))
    print("train集中empty的数量为：{}".format(empty_total - empty_nums))
    print("val集中full的数量为：{}".format(full_nums))
    print("val集中empty的数量为：{}".format(empty_nums))
    print("训练集，验证集划分完成！")

    for full_name in full_sample:
        shutil.move(full_fileDir + full_name, full_valDir)
    for empty_name in empty_sample:
        shutil.move(empty_fileDir + empty_name, empty_valDir)
    # return


# def move_label(file_list, file_label_train, file_label_val):
#     for i in file_list:
#         if i.endswith('.jpg'):
#             # filename = file_label_train + "\\" + i[:-4] + '.xml'  # 可以改成xml文件将’.txt‘改成'.xml'就可以了
#             filename = file_label_train + "\\" + i[:-4] + '.txt'  # 可以改成xml文件将’.txt‘改成'.xml'就可以了
#             if os.path.exists(filename):
#                 shutil.move(filename, file_label_val)
#                 print(i + "处理成功！")


if __name__ == '__main__':
    # Linux/pycharm控制台相对路径
    full_fileDir = './imagenet/images/train/full/'  # full源图片文件夹路径
    empty_fileDir = './imagenet/images/train/empty/'  # empty源图片文件夹路径
    full_valDir = './imagenet/images/val/full/'  # full图片移动到新的文件夹路径
    empty_valDir = './imagenet/images/val/empty/'  # empty图片移动到新的文件夹路径

    # # pycharm直接运行的相对路径
    # full_fileDir = '../images/train/full/'  # full源图片文件夹路径
    # empty_fileDir = '../images/train/empty/'  # empty源图片文件夹路径
    # full_valDir = '../images/val/full/'  # full图片移动到新的文件夹路径
    # empty_valDir = '../images/val/empty/'  # empty图片移动到新的文件夹路径

    move_img(full_fileDir, empty_fileDir, full_valDir, empty_valDir)

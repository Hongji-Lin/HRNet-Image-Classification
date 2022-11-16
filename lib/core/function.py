# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging

import torch
import torchvision
from core.evaluate import accuracy
from torch import device
from torchvision.transforms import transforms
import sys
sys.path.append('../imagenet/imageSets/utils/data_utils.py')
from utils.data_utils import plot_class_preds
from data_utils import read_split_data

from lib import config

logger = logging.getLogger(__name__)
device = torch.device(list(config.GPUS) if torch.cuda.is_available() else "cpu")


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to full mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        images, labels = data

        # measure imagenet loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet

        # compute output
        output = model(images)
        labels = labels.cuda(non_blocking=True)

        loss = criterion(output, labels)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))
        prec1, prec5 = accuracy(output, labels, (1, 5))
        top1.update(prec1[0], images.size(0))
        top5.update(prec5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=images.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_top1', top1.val, global_steps)
                # 显示每个batch的图片
                images, labels = next(iter(train_loader))
                img_input = torchvision.utils.make_grid(images)
                writer.add_image('input_image1', img_input, global_steps)


def validate(config, val_loader, model, criterion, output_dir, tb_log_dir,
             writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 定义训练以及预测时的预处理方法
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, labels = data
            # compute output
            output = model(images)
            labels = labels.cuda(non_blocking=True)
            loss = criterion(output, labels)

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            prec1, prec5 = accuracy(output, labels, (1, 2))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Error@1 {error1:.3f}\t' \
              'Error@5 {error5:.3f}\t' \
              'Accuracy@1 {top1.avg:.3f}\t' \
              'Accuracy@5 {top5.avg:.3f}\t'.format(
                  batch_time=batch_time, loss=losses, top1=top1, top5=top5,
                  error1=100-top1.avg, error5=100-top5.avg)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_top1', top1.avg, global_steps)

            # add figure into tensorboard
            data_path = '../imagenet/images/train'
            train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)
            valdir = val_images_path
            fig = plot_class_preds(net=model,
                                   images_dir=valdir,
                                   transform=data_transform["val"],
                                   num_plot=5,
                                   device=device)
            if fig is not None:
                writer.add_figure("predictions vs. actuals",
                                  figure=fig,
                                  global_step=global_steps)

            writer_dict['train_global_steps'] = global_steps + 1

        print('class:{}'.format(output.argmax(1)))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

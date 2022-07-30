import utils.crl_utils
from utils import utils
import torch.nn as nn
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import random


def train(loader, model, criterion, criterion_ranking, optimizer, epoch, history, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    cls_losses = utils.AverageMeter()
    ranking_losses = utils.AverageMeter()
    end = time.time()
    model.train()
    for i, (input, target, idx) in enumerate(loader):
        data_time.update(time.time() - end)
        input, target = input.cuda(), target.long().cuda()

        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        if args.method == 'sam' or args.method == 'fmfp':
            optimizer.first_step(zero_grad=True)
            criterion(model(input), target).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        # record loss and accuracy
        prec, correct = utils.accuracy(output, target)

        total_losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec {top1.val:.2f}% ({top1.avg:.2f}%)'.format(
        #         epoch, i, len(loader), batch_time=batch_time,
        #         data_time=data_time, loss=total_losses, top1=top1))

    logger.write([epoch, total_losses.avg, top1.avg])

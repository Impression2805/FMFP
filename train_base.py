import utils.crl_utils
from utils import utils
import torch.nn as nn
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import random


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss
kdloss = KDLoss(2.0)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.05):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()
criterion_ls = LabelSmoothingCrossEntropy()


def mixup_data(x, y, alpha=0.3, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class focal_loss(nn.Module):
    def __init__(self, class_num=10, alpha=None, gamma=2.0, size_average=True):
        super(focal_loss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 2))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)
        probs = probs.clamp(min=0.0001, max=1.0)
        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def train(loader, model, criterion, criterion_ranking, optimizer, epoch, history, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    cls_losses = utils.AverageMeter()
    ranking_losses = utils.AverageMeter()
    end = time.time()
    focal_criterion = focal_loss(class_num=args.classnumber)
    model.train()

    for i, (input, target, idx) in enumerate(loader):
        data_time.update(time.time() - end)
        if args.method == 'Baseline':
            input, target = input.cuda(), target.long().cuda()
            output = model(input)
            loss = criterion(output, target)
        elif args.method == 'L1':
            input, target = input.cuda(), target.long().cuda()
            output = model(input)
            norm_loss = 0.01*output.abs().sum(dim=1).mean()
            loss = criterion(output, target) + norm_loss
        elif args.method == 'Mixup':
            input, target = input.cuda(), target.long().cuda()
            input, target_a, target_b, lam = mixup_data(input, target)
            output = model(input)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        elif args.method == 'LS':
            input, target = input.cuda(), target.long().cuda()
            output = model(input)
            loss = criterion_ls(output, target)
        elif args.method == 'focal':
            input, target = input.cuda(), target.long().cuda()
            output = model(input)
            if epoch < 10:
                loss = criterion(output, target)
            else:
                loss = focal_criterion(output, target)
        elif args.method == 'CRL':
            input, target = input.cuda(), target.long().cuda()
            output = model(input)
            conf = F.softmax(output, dim=1)
            confidence, _ = conf.max(dim=1)

            rank_input1 = confidence
            rank_input2 = torch.roll(confidence, -1)
            idx2 = torch.roll(idx, -1)

            rank_target, rank_margin = history.get_target_margin(idx, idx2)
            rank_target_nonzero = rank_target.clone()
            rank_target_nonzero[rank_target_nonzero == 0] = 1
            rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

            ranking_loss = criterion_ranking(rank_input1,rank_input2,rank_target)
            cls_loss = criterion(output, target)
            ranking_loss = args.rank_weight * ranking_loss
            loss = cls_loss + ranking_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec, correct = utils.accuracy(output, target)
        total_losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec {top1.val:.2f}% ({top1.avg:.2f}%)'.format(
        #            epoch, i, len(loader), batch_time=batch_time,
        #            data_time=data_time, loss=total_losses,top1=top1))
        if args.method == 'CRL':
            history.correctness_update(idx, correct, output)
    if args.method == 'CRL':
        history.max_correctness_update(epoch)
    logger.write([epoch, total_losses.avg, top1.avg])

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from scipy.spatial import distance
from scipy.stats import chi2
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import normalize
import argparse
import os
import csv
import math
import pandas as pd
import numpy as np
import resource
from collections import OrderedDict

from model import LeNet
from model import AlexNet
from model import resnet
from model import resnet18
from model import densenet_BC
from model import vgg
from model import mobilenet
from model import efficientnet
from model import wrn
from model import convmixer
from utils import data as dataset
from utils import crl_utils
from utils import metrics
from utils import utils
import train_base

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def csv_writter(path, dic, start):
    if os.path.isdir(path) == False: os.makedirs(path)
    os.chdir(path)
    # Write dic
    if start == 1:
        mode = 'w'
    else:
        mode = 'a'
    with open('logs.csv', mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        if start == 1:
            writer.writerow(dic.keys())
        writer.writerow([elem["string"] for elem in dic.values()])


class Counter(dict):
    def __missing__(self, key):
        return None


parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--epochs', default=201, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--plot', default=20, type=int, help='')
parser.add_argument('--run', default=3, type=int, help='')
parser.add_argument('--classnumber', default=10, type=int, help='class number for the dataset')
parser.add_argument('--data', default='cifar10', type=str, help='Dataset name to use [cifar10, cifar100]')
parser.add_argument('--model', default='res110', type=str, help='Models name to use [res110, dense, wrn, cmixer, efficientnet, mobilenet, vgg]')
parser.add_argument('--method', default='Baseline', type=str, help='[Baseline, Mixup, LS, L1, focal, CRL]')
parser.add_argument('--data_path', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--cwd_weight', default=0.1, type=float, help='Trianing time tempscaling')
parser.add_argument('--save_path', default='./output/', type=str, help='Savefiles directory')
parser.add_argument('--rank_weight', default=1.0, type=float, help='Rank loss weight')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N', help='print frequency (default: 10)')

args = parser.parse_args()

def main():
    acc_list=[]
    auroc_list=[]
    aupr_success_list=[]
    aupr_list=[]
    fpr_list=[]
    tnr_list = []
    aurc_list=[]
    eaurc_list=[]
    ece_list=[]
    nll_list=[]
    brier_list=[]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    save_path = args.save_path + args.data + '_' + args.model + '_' + args.method
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_loader, valid_loader, test_loader, \
    test_onehot, test_label = dataset.get_loader(args.data, args.data_path, args.batch_size, args)

    if args.data == 'cifar100':
        num_class = 100
        args.classnumber = 100
    else:
        num_class = 10
    model_dict = {
        "num_classes": num_class,
    }
    for r in range(args.run):
        print(100*'#')
        print(r)
        if args.model == 'resnet18':
            model = resnet18.ResNet18(**model_dict).cuda()
        elif args.model == 'res110':
            model = resnet.resnet110(**model_dict).cuda()
        elif args.model == 'dense':
            model = densenet_BC.DenseNet3(depth=100, num_classes=num_class,
                                          growth_rate=12, reduction=0.5,
                                          bottleneck=True, dropRate=0.0).cuda()
        elif args.model == 'vgg':
            model = vgg.vgg16(**model_dict).cuda()
        elif args.model == 'wrn':
            model = wrn.WideResNet(28, num_class, 10).cuda()
        elif args.model == 'efficientnet':
            model = efficientnet.efficientnet(**model_dict).cuda()
        elif args.model == 'mobilenet':
            model = mobilenet.mobilenet(**model_dict).cuda()
        elif args.model == "cmixer":
            model = convmixer.ConvMixer(256, 16, kernel_size=8, patch_size=1, n_classes=num_class).cuda()
        elif args.model == 'LeNet':
            model = LeNet.LeNet().cuda()
        elif args.model == 'AlexNet':
            model = AlexNet.AlexNet().cuda()

        cls_criterion = nn.CrossEntropyLoss().cuda()

        base_lr = 0.1  # Initial learning rate
        lr_strat = [80, 130, 170]
        lr_factor = 0.1  # Learning rate decrease factor
        custom_weight_decay = 5e-4  # Weight Decay
        custom_momentum = 0.9  # Momentum
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=custom_momentum,
                                    weight_decay=custom_weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=lr_strat, gamma=lr_factor)
        if args.model == "convmixer":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = None

        # make logger
        train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
        result_logger = utils.Logger(os.path.join(save_path, 'result.log'))

        correctness_history = crl_utils.History(len(train_loader.dataset))
        ranking_criterion = nn.MarginRankingLoss(margin=0.0).cuda()

        # start Train
        for epoch in range(1, args.epochs + 1):
            if scheduler != None:
                scheduler.step()
            train_base.train(train_loader,
                        model,
                        cls_criterion,
                        ranking_criterion,
                        optimizer,
                        epoch,
                        correctness_history,
                        train_logger,
                        args)

            # save model
            if epoch == args.epochs:
                torch.save(model.state_dict(),
                           os.path.join(save_path, 'model.pth'))
            # finish train

            # calc measure
            if epoch % args.plot == 0:
                print(100*'#')
                print(epoch)
                acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, ece, nll, brier = metrics.calc_metrics(args, test_loader,
                                                                                    test_label,
                                                                                    test_onehot,
                                                                                    model,
                                                                                    cls_criterion)
        acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, ece, nll, brier = metrics.calc_metrics(args, test_loader,
                                                                                                 test_label,
                                                                                                 test_onehot,
                                                                                                 model,
                                                                                                 cls_criterion)
        # result write
        result_logger.write([acc, auroc, aupr_success, aupr, fpr, tnr, aurc, eaurc, ece, nll, brier])

        acc_list.append(acc)
        auroc_list.append(auroc)
        aupr_success_list.append(aupr_success)
        aupr_list.append(aupr)
        fpr_list.append(fpr)
        tnr_list.append(tnr)
        aurc_list.append(aurc)
        eaurc_list.append(eaurc)
        ece_list.append(ece)
        nll_list.append(nll)
        brier_list.append(brier)


    acc_mean = np.mean(acc_list)
    auroc_mean = np.mean(auroc_list)
    aupr_success_mean = np.mean(aupr_success_list)
    aupr_mean = np.mean(aupr_list)
    fpr_mean = np.mean(fpr_list)
    tnr_mean = np.mean(tnr_list)
    aurc_mean = np.mean(aurc_list)
    eaurc_mean = np.mean(eaurc_list)
    ece_mean = np.mean(ece_list)
    nll_mean = np.mean(nll_list)
    brier_mean = np.mean(brier_list)

    acc_std = np.std(acc_list,ddof=1)
    auroc_std = np.std(auroc_list,ddof=1)
    aupr_success_std = np.std(aupr_success_list,ddof=1)
    aupr_std = np.std(aupr_list,ddof=1)
    fpr_std = np.std(fpr_list,ddof=1)
    tnr_std = np.std(tnr_list,ddof=1)
    aurc_std = np.std(aurc_list,ddof=1)
    eaurc_std = np.std(eaurc_list,ddof=1)
    ece_std = np.std(ece_list,ddof=1)
    nll_std = np.std(nll_list,ddof=1)
    brier_std = np.std(brier_list,ddof=1)

    logs_dict = OrderedDict(Counter(
        {
            "acc": {
                "value": round(acc_mean, 2),
                "string": f"{round(acc_mean, 2)}",
            },

            "aurc": {
                "value": round(aurc_mean, 2),
                "string": f"{round(aurc_mean, 2)}",
            },
            "eaurc": {
                "value": round(eaurc_mean, 2),
                "string": f"{round(eaurc_mean, 2)}",
            },
            "fpr": {
                "value": round(fpr_mean, 2),
                "string": f"{round(fpr_mean, 2)}",
            },
            "tnr": {
                "value": round(tnr_mean, 2),
                "string": f"{round(tnr_mean, 2)}",
            },
            "auroc": {
                "value": round(auroc_mean, 2),
                "string": f"{round(auroc_mean, 2)}",
            },
            "aupr_success": {
                "value": round(aupr_success_mean, 2),
                "string": f"{round(aupr_success_mean, 2)}",
            },
            "aupr": {
                "value": round(aupr_mean, 2),
                "string": f"{round(aupr_mean, 2)}",
            },

            "ece": {
                "value": round(ece_mean, 2),
                "string": f"{round(ece_mean, 2)}",
            },
            "nll": {
                "value": round(nll_mean, 2),
                "string": f"{(round(nll_mean, 2))}",
            },
            "brier": {
                "value": round(brier_mean, 2),
                "string": f"{(round(brier_mean, 2))}",
            },
        }
    ))

    # Print metrics
    csv_writter(path=save_path, dic=OrderedDict(logs_dict), start=r)
    os.chdir('../..')

    logs_dict = OrderedDict(Counter(
        {
            "acc": {
                "value": round(acc_std, 2),
                "string": f"{round(acc_std, 2)}",
            },

            "aurc": {
                "value": round(aurc_std, 2),
                "string": f"{round(aurc_std, 2)}",
            },
            "eaurc": {
                "value": round(eaurc_std, 2),
                "string": f"{round(eaurc_std, 2)}",
            },
            "fpr": {
                "value": round(fpr_std, 2),
                "string": f"{round(fpr_std, 2)}",
            },
            "tnr": {
                "value": round(tnr_std, 2),
                "string": f"{round(tnr_std, 2)}",
            },
            "auroc": {
                "value": round(auroc_std, 2),
                "string": f"{round(auroc_std, 2)}",
            },
            "aupr_success": {
                "value": round(aupr_success_std, 2),
                "string": f"{round(aupr_success_std, 2)}",
            },
            "aupr": {
                "value": round(aupr_std, 2),
                "string": f"{round(aupr_std, 2)}",
            },

            "ece": {
                "value": round(ece_std, 2),
                "string": f"{round(ece_std, 2)}",
            },
            "nll": {
                "value": round(nll_std, 2),
                "string": f"{(round(nll_std, 2))}",
            },
            "brier": {
                "value": round(brier_std, 2),
                "string": f"{(round(brier_std, 2))}",
            },
        }
    ))

    # Print metrics
    csv_writter(path=save_path, dic=OrderedDict(logs_dict), start=r)
    os.chdir('../..')


if __name__ == "__main__":
    main()




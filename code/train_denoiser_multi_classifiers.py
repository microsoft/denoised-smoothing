# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# File for training denoisers with multiple classifier alternatively attached to (a variant of the train_denoiser.py)

from architectures import DENOISERS_ARCHITECTURES, get_architecture
from datasets import get_dataset, DATASETS
from test_denoiser import test, test_with_classifier
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code, requires_grad_

import argparse
import datetime
import numpy as np
import os
import time
import torch

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, choices=DATASETS)
parser.add_argument('--arch', type=str, choices=DENOISERS_ARCHITECTURES)
parser.add_argument('--outdir', type=str, help='folder to save denoiser and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of noise distribution for data augmentation")
parser.add_argument('--objective', default='stability', type=str,
                    help="the objective that is used to train the denoiser",
                    choices=['classification', 'stability'])
parser.add_argument("--classifiers_path", type=str, default='../pretrained_models/cifar10_classifiers', 
                    help="Path to pretrained classifiers")
parser.add_argument("--classifier_idx", type=int, default=-1, help="Certify one model only")
parser.add_argument('--pretrained-denoiser', default='', type=str,
                    help='path to the pretrained denoiser')
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='SGD, Adam, or Adam then SGD', choices=['SGD', 'Adam','AdamThenSGD'])
parser.add_argument('--start-sgd-epoch', default=50, type=int,
                    help='[Relevent only to AdamThenSGD.] Epoch at which adam switches to SGD')
parser.add_argument('--start-sgd-lr', default=1e-3, type=float,
                    help='[Relevent only to AdamThenSGD.] LR at which SGD starts after Adam')
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from an existing checkpoint')
parser.add_argument('--azure_datastore_path', type=str, default='',
                    help='Path to imagenet on azure')
args = parser.parse_args()

if args.azure_datastore_path:
    os.environ['IMAGENET_DIR_AZURE'] = os.path.join(args.azure_datastore_path, 'datasets/imagenet_zipped')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

toPilImage = ToPILImage()

def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # args.outdir = os.path.join(os.getenv('PT_DATA_DIR', './'), args.outdir)
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Copy code to output directory
    copy_code(args.outdir)
    
    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)
    ## This is used to test the performance of the denoiser attached to a cifar10 classifier
    cifar10_test_loader = DataLoader(get_dataset('cifar10', 'test'), shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    if args.pretrained_denoiser:
        checkpoint = torch.load(args.pretrained_denoiser)
        assert checkpoint['arch'] == args.arch
        denoiser = get_architecture(checkpoint['arch'], args.dataset)
        denoiser.load_state_dict(checkpoint['state_dict'])
    else:
        denoiser = get_architecture(args.arch, args.dataset)

    if args.optimizer == 'Adam':
        optimizer = Adam(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = SGD(denoiser.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamThenSGD':
        optimizer = Adam(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    starting_epoch = 0
    logfilename = os.path.join(args.outdir, 'log.txt')

    ## Resume from checkpoint if exists and if resume flag is True
    denoiser_path = os.path.join(args.outdir, 'checkpoint.pth.tar')
    if args.resume and os.path.isfile(denoiser_path):
        print("=> loading checkpoint '{}'".format(denoiser_path))
        checkpoint = torch.load(denoiser_path,
                                map_location=lambda storage, loc: storage)
        assert checkpoint['arch'] == args.arch
        starting_epoch = checkpoint['epoch']
        denoiser.load_state_dict(checkpoint['state_dict'])
        if starting_epoch >= args.start_sgd_epoch and args.optimizer == 'AdamThenSGD ': # Do adam for few steps thaen continue SGD
            print("-->[Switching from Adam to SGD.]")
            args.lr = args.start_sgd_lr
            optimizer = SGD(denoiser.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                        .format(denoiser_path, checkpoint['epoch']))
    else:
        if args.resume: print("=> no checkpoint found at '{}'".format(args.outdir))
        init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttestAcc")

    # the first model is the model we test on, the other 14 models are surrogate models that we use 
    # (see the paper for more detail)
    models = [
    "ResNet110", # fixed model
    #surrogate models
    "WRN", "WRN40", "VGG16", "VGG19", "ResNet18","PreActResNet18","ResNeXt29_2x64d",
    "MobileNet","MobileNetV2","SENet18","ShuffleNetV2","EfficientNetB0","GoogLeNet","DenseNet121"
    ]

    path = os.path.join(args.classifiers_path, '{}/noise_0.00/checkpoint.pth.tar')

    base_classifiers_paths = [
        path.format(model) for model in models
    ]
    base_classifiers = []
    if args.classifier_idx == -1:
        for base_classifier in base_classifiers_paths:
            # load the base classifier
            checkpoint = torch.load(base_classifier)
            base_classifier = get_architecture(checkpoint["arch"], args.dataset)
            base_classifier.load_state_dict(checkpoint['state_dict'])

            requires_grad_(base_classifier, False)
            base_classifiers.append(base_classifier.eval().cuda())
    else:
        if args.classifier_idx not in range(len(models)): raise Exception("Unknown model")
        print("Model namse: {}".format(base_classifiers_paths[args.classifier_idx]))
        checkpoint = torch.load(base_classifiers_paths[args.classifier_idx])
        base_classifier = get_architecture(checkpoint["arch"], args.dataset)
        base_classifier.load_state_dict(checkpoint['state_dict'])

        requires_grad_(base_classifier, False)
        base_classifiers.append(base_classifier.eval().cuda())

    criterion = CrossEntropyLoss(size_average=None, reduce=None, reduction = 'mean').cuda()
    best_acc = 0

    for epoch in range(starting_epoch, args.epochs):
        before = time.time()
        train_loss = train(train_loader, denoiser, criterion, optimizer, epoch, args.noise_sd, base_classifiers[1:])
        test_loss, test_acc = test_with_classifier(cifar10_test_loader, denoiser, criterion, args.noise_sd, args.print_freq, base_classifiers[0])

        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, after - before,
            args.lr, train_loss, test_loss, test_acc))

        scheduler.step(epoch)
        args.lr = scheduler.get_lr()[0]

        # Switch from Adam to SGD
        if epoch == args.start_sgd_epoch and args.optimizer == 'AdamThenSGD ': # Do adam for few steps thaen continue SGD
            print("-->[Switching from Adam to SGD.]")
            args.lr = args.start_sgd_lr
            optimizer = SGD(denoiser.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': denoiser.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'checkpoint.pth.tar'))

        if args.objective in ['classification', 'stability'] and test_acc > best_acc:
            best_acc = test_acc
        else:
            continue

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': denoiser.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'best.pth.tar'))

def train(loader: DataLoader, denoiser: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float, classifiers: torch.nn.Module=None):
    """
    docstring here
        :param loader:DataLoader: training dataloader
        :param denoiser:torch.nn.Module: denoiser to be trained
        :param criterion: loss function
        :param optimizer:Optimizer: optimizer used during trainined
        :param epoch:int: current epoch (for logging)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param classifiers:torch.nn.Module=None: a list of ``freezed'' classifiers that will be attached successively to the denoiser,
                                                required for classifciation/stability objectives    
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    # switch to train mode
    denoiser.train()

    for i, (inputs, targets) in enumerate(loader):
        classifier = classifiers[i%len(classifiers)]
        classifier.eval()
        
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()

        # augment inputs with noise
        noise = torch.randn_like(inputs, device='cuda') * noise_sd

        # compute output
        outputs = denoiser(inputs + noise)
        if classifier:
            outputs = classifier(outputs)
        
        if isinstance(criterion, MSELoss):
            loss = criterion(outputs, inputs)
        elif isinstance(criterion, CrossEntropyLoss):
            if args.objective == 'stability':
                with torch.no_grad():
                    targets = classifier(inputs)
                    targets = targets.argmax(1).detach().clone()
            loss = criterion(outputs, targets)

        # record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    return losses.avg



if __name__ == "__main__":
    main()

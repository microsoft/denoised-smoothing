# evaluate a smoothed classifier on a dataset
from architectures import get_architecture, IMAGENET_CLASSIFIERS
from core import Smooth
from datasets import get_dataset, DATASETS, get_num_classes
from time import time

import argparse
import datetime
import os
import torch

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", choices=DATASETS, help="which dataset")
parser.add_argument("--base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--denoiser', type=str, default='',
                    help='Path to a denoiser to attached before classifier during certificaiton.')
parser.add_argument('--azure_datastore_path', type=str, default='',
                    help='Path to imagenet on azure')
parser.add_argument('--philly_imagenet_path', type=str, default='',
                    help='Path to imagenet on philly')
args = parser.parse_args()

if args.azure_datastore_path:
    os.environ['IMAGENET_DIR_AZURE'] = os.path.join(args.azure_datastore_path, 'datasets/imagenet_zipped')
elif args.philly_imagenet_path:
    os.environ['IMAGENET_DIR_PHILLY'] = os.path.join(args.philly_imagenet_path, './')
else:
    os.environ['IMAGENET_DIR_PHILLY'] = "/hdfs/public/imagenet/2012/"


if __name__ == "__main__":
    # load the base classifier
    if args.base_classifier in IMAGENET_CLASSIFIERS:
        assert args.dataset == 'imagenet'
        # loading pretrained imagenet architectures
        base_classifier = get_architecture(args.base_classifier ,args.dataset, pytorch_pretrained=True)
    else:
        checkpoint = torch.load(args.base_classifier)
        base_classifier = get_architecture(checkpoint['arch'], args.dataset)
        base_classifier.load_state_dict(checkpoint['state_dict'])

    if args.denoiser != '':
        checkpoint = torch.load(args.denoiser)
        if "off-the-shelf-denoiser" in args.denoiser:
            denoiser = get_architecture('orig_dncnn', args.dataset)
            denoiser.load_state_dict(checkpoint)
        else:
            denoiser = get_architecture(checkpoint['arch'] ,args.dataset)
            denoiser.load_state_dict(checkpoint['state_dict'])
        base_classifier = torch.nn.Sequential(denoiser, base_classifier)

    base_classifier = base_classifier.eval().cuda()

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    if not os.path.exists(args.outfile.split('sigma')[0]):
        os.makedirs(args.outfile.split('sigma')[0])

    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)
    f.close()

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        
        f = open(args.outfile, 'a')
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), flush=True)
        f.close()

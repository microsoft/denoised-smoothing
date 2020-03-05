# evaluate a smoothed classifier on a dataset
import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import pymongo

from IPython import embed

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--start", type=int, default=-1, help="start index of the samples to certify")
parser.add_argument("--end", type=int, default=-1, help="end index of the samlpes to certify")
parser.add_argument('--denoiser', type=str, default='',
					help='Path to a denoiser ')

# PGD-attack params
parser.add_argument('--random-start', default=True, type=bool)
parser.add_argument('--step-size', default=2.0, type=float)
parser.add_argument('--epsilon', default=8.0, type=float)
parser.add_argument('--num-steps', default=10, type=int)
parser.add_argument('--adv-training', action='store_true')

args = parser.parse_args()


#######################################################################################
# Azure's Mongo db Configuration
uri = "mongodb://smoothing-db:lxCEu3Dm66BS3TxxMgh8BwyYhBnhM8k2SR"\
	  "n3oNZa4e7vN0fb7n86wAmX6rahPzmwA7gjmWqDudDjocYmQzDuIg==@smo"\
	  "othing-db.documents.azure.com:10255/?ssl=true&replicaSet=globaldb"

client = pymongo.MongoClient(uri)
database = client['certification'] # Make sure that this database exists on Azure
results_db = database[args.dataset] # Make sure that this collection exists in the db
#######################################################################################

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

	print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)

	# iterate through the dataset
	dataset = get_dataset(args.dataset, args.split)
	results = []
	for i in range(args.start, args.end + 1):

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

		print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
			i, label, prediction, radius, correct, time_elapsed), flush=True)

		results.append({'model_path': args.base_classifier, 'denoiser': args.denoiser, 'dataset': args.dataset, 'sigma_test':args.sigma ,'idx':i, 'label':label, 'prediction':prediction, 'radius': radius, 'correct': correct, 'time_elapsed': time_elapsed, 'N': args.N})
	results_db.insert(results)

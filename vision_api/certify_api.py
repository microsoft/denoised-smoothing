# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import glob
import io
import numpy as np
import operator
import os
import pickle
import scipy.misc
import scipy.ndimage
import torch
import torch.nn as nn

from robust_api import RobustAPI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Certify online APIs")
    parser.add_argument("--gpu", default=None, type=str, help="id(s) for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--noise_sd", default=0.25, type=float,
                        help="standard deviation of noise distribution for data augmentation")
    parser.add_argument("--N0", type=int, default=100)
    parser.add_argument("--N", type=int, default=1000, help="number of samples to use")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument("--api", default="azure", help="which api to use", 
                        choices=["azure", "google", "clarifai", "aws"])
    parser.add_argument("--denoiser_checkpoint", type=str, default='', 
                        help="path to the trained denoisers. If '', no denoiser will be used.")
    parser.add_argument("--save", default="./logs", type=str, help="where to save the certification results")
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # The directory where test images are stored
    DATA_ROOT = "images/"

    import sys
    sys.path.append("../code/")
    from architectures import get_architecture

    if args.denoiser_checkpoint:
        checkpoint = torch.load(args.denoiser_checkpoint)
        denoiser = get_architecture(checkpoint['arch'] , 'imagenet')
        denoiser.load_state_dict(checkpoint['state_dict'])
    else:
        print('Not using a denoiser')
        denoiser = None

    online_robust_api_base = RobustAPI(args.api, online=True)
    online_robust_api_denoiser = RobustAPI(args.api, denoiser=denoiser, online=True)

    for i, filename in enumerate(glob.glob(os.path.join(DATA_ROOT, "*.png"))):
        print(filename)
        img = scipy.ndimage.imread(filename) / 255.

        # Predict the ground truth, i.e. the "true" class of the clean image predicted by the
        # API. Note that RobustAPI here has no denoiser, so it is not really robust, but we 
        # don't care since we are predicting clean images.
        true_label, pred, clean_log = online_robust_api_base.predict(img, N=1, noise_sd=0)
        print("Example %d: True class: %s"%(i, true_label))

        top_class, radius, noisy_logs = online_robust_api_denoiser.certify(img, args.noise_sd, args.N0, args.N, args.alpha)

        # Store the predictions
        with open(os.path.join(args.save, "pred_hist_%d.pkl"%i), "wb") as f:
            pickle.dump({"ground_truth": clean_log, "pred":noisy_logs}, f)
        f.close()

        correct = False
        if top_class == -1:
            print("Certification Abstained!\nTrue class: %s\n"%(true_label))
        else:
            if top_class == true_label:
                print("Certified correct with radius %f!\n"%(radius))
                correct = True
            else:
                print("Certification failed!\nTrue class: %s Certified class: %s\n"%(true_label, top_class))

        f = open(os.path.join(args.save, "log.txt"), 'a')
        print("{}\t{}\t{:.3f}\t{}\t{}".format(
            i, correct, radius, true_label, top_class), file=f, flush=True)
        f.close()
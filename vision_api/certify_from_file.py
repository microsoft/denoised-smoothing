# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
from robust_api import RobustAPI
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint

import argparse
import io
import numpy as np
import operator
import os
import pickle
import scipy.ndimage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Certify from query log files")
    parser.add_argument("--noise_sd", default=0.25, type=float,
                        help="standard deviation of noise distribution for data augmentation")
    parser.add_argument("--N0", type=int, default=20)
    parser.add_argument("--N", type=int, default=100, help="number of samples to use")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument("--api", default="azure", help="which api to use", 
                        choices=["azure", "google", "clarifai", "aws"])
    parser.add_argument("--log_dir", type=str, help="path to the dir where the query log files are stored")
    parser.add_argument("--save", default="./certification_output", type=str, help="where to save the certification results")
    args = parser.parse_args()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    FILE_ROOT = args.log_dir

    outfile = os.path.join(args.save, "log.txt")
    f = open(outfile, 'w')
    print("idx\tcorrect\tradius\tlabel\tpredict", file=f, flush=True)
    f.close()

    offline_robust_api = RobustAPI(args.api, online=False)
    for i in range(100):
        filepath = os.path.join(FILE_ROOT, "pred_hist_%d.pkl"%i)
        with open(filepath,"rb") as f:
            query_log = pickle.load(f)
        f.close()

        clean_log = query_log["ground_truth"]
        noisy_logs = query_log["pred"]

        true_label, pred = offline_robust_api.predict(clean_log)
        print("Example %d: True class: %s"%(i, true_label))

        top_class, radius = offline_robust_api.certify(noisy_logs, args.noise_sd, args.N0, args.N, args.alpha)

        correct = False
        if top_class == -1:
            print("Abstain!\nTrue class: %s, Prediction results: %s\n"%(true_label, str(pred)))
        else:
            if top_class == true_label:
                print("Certificatied correct with radius %f!\nPrediction results: %s\n"%(radius, str(pred)))
                correct = True
            else:
                print("Certification failed!\nTrue class: %s Certified class: %s Prediction results: %s\n"%(true_label, top_class, pred))

        f = open(os.path.join(args.save, "log.txt"), 'a')
        print("{}\t{}\t{:.3f}\t{}\t{}".format(
            i, correct, radius, true_label, top_class), file=f, flush=True)
        f.close()
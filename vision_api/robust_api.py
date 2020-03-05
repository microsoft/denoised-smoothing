# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict, OrderedDict
from random import random
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint

import argparse
import io
import io
import numpy as np
import operator
import os
import requests
import scipy.misc
import scipy.ndimage
import sys
import torch

# Import API dependency package
from google.cloud import vision
from google.cloud.vision import types
from clarifai.rest import ClarifaiApp
import boto3 # package for AWS API

class RobustAPI(object):
    def __init__(self, api_name, denoiser=None, online=False):
        """
            :param api="azure": which API we want to use for prediction
                                ("azure", "google", "aws", "clarifai")
            :param denoiser=None: Denoiser attached to the API. If None, the input is 
                                directly passed to the API without being denoised.
            :param online=False: A flag to use APIs online, or to use old queries that are
                                dumped in a given folder.
        """
        SUPPORTED_APIS = ['azure', 'google', 'aws', 'clarifai']
        if api_name not in SUPPORTED_APIS:
            raise Exception("Api %s not supported"%api_name)
        self.api_name = api_name
        self.denoiser = denoiser
        self.online = online
        if self.online:
            print("[Operating in the online setting. Initializing the API client]...")
            if api_name == "google":
                # Set up vision imageannotator
                self.client = vision.ImageAnnotatorClient()
            elif api_name == "clarifai":
                # Set up clarifai vision app model
                if 'CLARIFAI_API_KEY' in os.environ:
                    api_key = os.environ['CLARIFAI_API_KEY']
                else:
                    print("\nSet the CLARIFAI_API_KEY environment variable of CLARIFAI.\n")
                    sys.exit()
                app = ClarifaiApp(api_key=api_key)
                self.client = app.public_models.general_model
            elif api_name == "aws":
                # Set up client for aws api
                self.client = boto3.client('rekognition')
            print("[API initialization successful!]")
    
            self.predict=self._predict_online
            self.certify=self._certify_online

            self.tmp_images_dir = 'tmp_images'
            if not os.path.isdir(self.tmp_images_dir):
                os.makedirs(self.tmp_images_dir)
            self.image_counter = 0 # used to identify the noisy and denoised images used in certification 
        else:
            print("[Operating in the offline setting.]...")
            self.predict=self._predict_offline
            self.certify=self._certify_offline


    def _predict_online(self, clean_img, N, noise_sd):
        """
        A function to predict (using randomized smoothing) via the online API. Essentially does 
        majority voting under Gausian perturbation of the inputs
            :param clean_img: the image to predict
            :param N: the number of noise samples over which the voting happens
            :param noise_sd: the std-dev of the Guassian noise perturbation of the input
            :return: (majority_class, top_class_distribution, prediction_logs)
                top_class: the majority vote class over N samples
                top_class_distribution: the distribution of top classes over the N noisy samples
                prediction_logs: a list of query logs for each of the N samples used in prediction 
                                (to save locally for optionally later offline offline use)
        """

        top_class_distribution = defaultdict(int)
        # A list of the history of responses, to save for (optional) later processing 
        # (save moeny by not querying the APIs again an again :) )
        prediction_logs = []
        for i in range(N):
            image_path =  os.path.join(self.tmp_images_dir, "sample_%d.png"%int(self.image_counter))
            noise = np.random.randn(*clean_img.shape) * noise_sd
            img = np.clip(clean_img + noise, 0, 1)
            
            if self.denoiser:
                img = self._denoise_image(img)

            scipy.misc.imsave(image_path, img)
            self.image_counter += 1

            response = self._query_api(image_path)
            
            prediction_logs.append(response)
            top_class = self._get_top_class(response)
            top_class_distribution[top_class] += 1

        majority_class = max(top_class_distribution.items(), key=operator.itemgetter(1))[0] 
        return majority_class, top_class_distribution, prediction_logs

    def _certify_online(self, img, noise_sd, N0, N, alpha):
        """
        Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
            :param img: the image to certify
            :param noise_sd: the std-dev of the Guassian noise perturbation of the input
            :param N0: the number of noise samples over which the voting happens
            :param N: the number of noise samples over which the voting happens
            :param alpha: the failure probability
            :return: (majority_class, radius, logs)
                majority_class: the majority vote class over 20 samples
                radius: the certified L2 radius around this datapoint
                logs: a list of query logs for each of the 120 samples used in certification 
                    (to save locally for optionally later offline use)         
        """
        cAHat, _, prediction_logs_N0 = self._predict_online(img, N0, noise_sd)

        _, prediction, prediction_logs_N = self._predict_online(img, N, noise_sd)
        nA = prediction[cAHat]

        prediction_logs = prediction_logs_N0 + prediction_logs_N

        pABar = proportion_confint(nA, N, 2 * alpha, method="beta")[0]
        print('pA: {:.3f}'.format(pABar))
        if pABar < 0.5:
            return -1 , 0.0, prediction_logs
        else:
            radius = noise_sd * norm.ppf(pABar)
            return cAHat, radius, prediction_logs


    def _predict_offline(self, prediction_logs):
        """
        A function to predict (using randomized smoothing) via the offline API query logs 
        of Gaussian noisy copies of a given image. Essentially does majority voting under 
        Gausian perturbation of the inputs
            :param prediction_logs: a list of the complete N query results returned by the API for 
                                one image (each for a different noisy copy of the image)
            :param api="azure": which API we want to use for prediction
                                ("azure", "google", "aws", "clarifai")
            :return: (majority_class, top_class_distribution)
                top_class: the majority vote class over N samples
                top_class_distribution: the distribution of top classes over the N noisy samples
        """
        top_class_distribution = defaultdict(int)

        for response in prediction_logs:
            top_class_distribution[self._get_top_class(response)] += 1

        majority_class = max(top_class_distribution.items(), key=operator.itemgetter(1))[0] 
        return majority_class, top_class_distribution
    
    def _certify_offline(self, prediction_logs, noise_sd, N0, N, alpha):
        """
        Certify the smoothed classifier given the API query logs.
            :param prediction_logs: a list of the complete query results returned by the API for 
                                one image (each for a different noisy copy of the image)
            :param noise_sd: the std-dev of the Guassian noise perturbation of the input
            :param N0: the number of noise samples over which the voting happens
            :param N: the number of noise samples over which the voting happens
            :param alpha: the failure probability
            :param api="azure": which API we want to use for prediction
                                ("azure", "google", "aws", "clarifai")
            :return: (majority_class, radius)
                majority_class: the majority vote class over 20 samples
                radius: the certified L2 radius around this datapoint
        """
        cAHat, _ = self._predict_offline(prediction_logs[:N0])

        _, prediction = self._predict_offline(prediction_logs[N0:]) 
        nA = prediction[cAHat]

        pABar = proportion_confint(nA, N, 2*alpha, method="beta")[0]
        print('pA: {:.3f}'.format(pABar))
        if pABar < 0.5:
            return -1 , 0.0
        else:
            radius = noise_sd * norm.ppf(pABar)
            return cAHat, radius


    def _get_top_class(self, api_response):
        """
        Given an API response, returns the top class 
            :param api_response: 
            :return str: the name of the top class

            api_response details:
            azure: https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/concept-tagging-images
            google: https://cloud.google.com/vision/docs/labels
            aws: https://www.clarifai.com/models/general-image-recognition-model-aaa03c23b3724a16a56b629203edc62c
            clarifai: https://www.clarifai.com/models/general-image-recognition-model-aaa03c23b3724a16a56b629203edc62c 
        """
        # Check if the response is empty
        if not api_response:
            return "None"

        if self.api_name == "google":
            return [*api_response][0]

        elif self.api_name == "clarifai":
            return api_response["outputs"][0]["data"]["concepts"][0]["name"]

        elif self.api_name == "aws":
            return api_response[0]['Name']

        elif self.api_name == 'azure':
            return api_response[0]['name']

    def _query_api(self, image_path):
        """
        Query the API.
            :param image_path: path to the image to be classified
            :return API response (different for different API)

            API Response details:
            azure: https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/concept-tagging-images
            google: https://cloud.google.com/vision/docs/labels
            aws: https://www.clarifai.com/models/general-image-recognition-model-aaa03c23b3724a16a56b629203edc62c
            clarifai: https://www.clarifai.com/models/general-image-recognition-model-aaa03c23b3724a16a56b629203edc62c 
        """
        if self.api_name == 'google':          
            with io.open(image_path, 'rb') as f:
                content = f.read()

            image = vision.types.Image(content=content)
            response = self.client.label_detection(image=image).label_annotations

            # We do some post-processing of the API response here because the raw 
            # response can not be saved in a pickle file.
            labels = OrderedDict()
            for res in response:
                labels[res.description] = res.score
            return labels

        elif self.api_name == 'clarifai':          
            return self.client.predict_by_filename(image_path)

        elif self.api_name == 'aws':          
            with open(image_path, 'rb') as image:
                response = self.client.detect_labels(Image={'Bytes': image.read()})
            return response['Labels']

        elif self.api_name == 'azure':
            if 'AZURE_COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
                subscription_key = os.environ['AZURE_COMPUTER_VISION_SUBSCRIPTION_KEY']
            else:
                print("\nSet the AZURE_COMPUTER_VISION_SUBSCRIPTION_KEY environment variable of Azure.\n**Restart your shell or IDE for changes to take effect.**")
                sys.exit()

            if 'AZURE_COMPUTER_VISION_ENDPOINT' in os.environ:
                endpoint = os.environ['AZURE_COMPUTER_VISION_ENDPOINT']

            analyze_url = endpoint + "vision/v2.1/analyze"

            # Read the image into a byte array
            image_data = open(image_path, "rb").read()
            headers = {'Ocp-Apim-Subscription-Key': subscription_key,
                    'Content-Type': 'application/octet-stream'}
            params = {'visualFeatures': "Tags"}
            response = requests.post(
                analyze_url, headers=headers, params=params, data=image_data)
            response.raise_for_status()

            # The 'analysis' object contains various fields that describe the image.
            analysis = response.json()
            return analysis["tags"]
            
    def _denoise_image(self, img):
        """
        A function to denoise a noisy image
            :param denoiser: the denoiser
            :param img: a numpy array (H,W,C) of the image to be denoised
            :return denoised_image: a (H,W,C) numpy array denoised image            
        """

        img = np.transpose(img, (2,0,1))
        img = np.expand_dims(img, 0)

        img = torch.Tensor(img).cuda()

        with torch.no_grad():
            out = torch.clamp(self.denoiser(img), 0, 1)

        img_color = np.transpose(out[0].cpu().numpy(), (1,2,0))

        return img_color


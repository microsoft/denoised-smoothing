from architectures import get_architecture

import argparse
import numpy as np
import os
import scipy.misc
import scipy.ndimage
import torch

def denoise_image(img, denoiser, save_dir):
    """
        :param img: the numpy array representing the image
        :param denoiser: Denoiser Network
        :param save_dir: the dir to save denoised images
    """
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, 0)

    img = torch.Tensor(img).cuda()

    with torch.no_grad():
        out = torch.clamp(denoiser(img), 0, 1)

    img_color = np.transpose(out[0].cpu().numpy(), (1,2,0))

    scipy.misc.imsave(os.path.join(save_dir, "denoised_%s_%d.png"%
                            (args.img_file.split(".")[0], args.noise_sd*100)), img_color)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize noisy images and denoised images")
    parser.add_argument("--img_file", type=str, help="path to the ImageNet test image")
    parser.add_argument("--noise_sd", default=0.12, type=float,
                        help="standard deviation of Gaussian noise distribution")
    parser.add_argument("--denoiser", type=str, help="Path to a denoiser")
    parser.add_argument("--save", default="./visualization", type=str, help="path to save the denoised images")
    args = parser.parse_args()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    checkpoint = torch.load(args.denoiser)
    denoiser = get_architecture(checkpoint['arch'], 'imagenet')
    denoiser.load_state_dict(checkpoint)
    denoiser.cuda()

    filename = args.img_file

    img = (scipy.ndimage.imread(args.img_file, mode="RGB")) / 255.
    img += np.random.randn(*img.shape) * args.noise_sd

    scipy.misc.imsave(os.path.join(args.save, "noisy_%s_%d.png"%
                            (args.img_file.split(".")[0], args.noise_sd*100)), np.clip(img, 0, 1))

    denoise_image(img, denoiser, args.save)
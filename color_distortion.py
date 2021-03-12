import cv2
import numpy as np
from torchvision import transforms

np.random.seed(0)

class ColorDistortion(object):

    def __init__(self, s=1.0):
        self.s = s
        self.color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        self.rnd_color_jitter = transforms.RandomApply([self.color_jitter], p=0.8)
        self.rnd_gray = transforms.RandomGrayscale(p=0.2)
        self.color_distort = transforms.Compose([
            self.rnd_color_jitter,
            self.rnd_gray])

    def __call__(self, sample):
        sample = self.color_distort(sample)

        return sample


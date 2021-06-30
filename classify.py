"""
module provides classifiaction of an image
and save a classification result to the given directory
"""


import numpy as np
import torch
from pretrained_model import Net
from resnet import ResNet, ResBlock, _add_conv
from PIL import Image
import torchvision.transforms as transforms
import os
from typing import Union, List
from torch.autograd import Variable
import config as cfg
from utils import get_film_names
import sys
from matplotlib import pyplot as plt
from copy import deepcopy


class Classifier:

    def __init__(self, model_path: str,
                 film_classes_path: str,
                 output_path: Union[str, None] = None,
                 need_cuda: bool = False) -> None:
        """
        input:
            model_path - path to the the model which has already  been trained
            film_classes_path - path to film types
            output_path - where to save an image
            need cuda - defines device type

        """

        # from config
        self.pretrained_model_name = cfg.PRETRAINED_MODEL
        self.num_classes = cfg.NUM_CLASSES

        self.model_path = model_path
        self.image = None
        self.film_names = get_film_names(film_classes_path)
        self.output_path = output_path

        # Tensor is needed later for pass an input to the CUDA device
        if need_cuda and torch.cuda.is_available():
            self.cuda = True
            self.Tensor = torch.cuda.FloatTensor

        else:
            self.cuda = False
            self.Tensor = None

        self._load_model()  # get model
        sys.stdout.write('model has been loaded\n')
        sys.stdout.write(f'cuda: {self.cuda}\n')

    def load_image(self, input_image_path: str,
                   mean: Union[np.ndarray, List[float], None] = None,
                   std: Union[np.ndarray, List[float], None] = None) -> None:
        """
        loads image, makes basic transfroms

        input:
           input_image_path: str
           mean - per channel mean value - the np.ndarray type is preferable
            std - per channel std value - the np.ndarray type is preferable
        """

        image = Image.open(input_image_path)
        self.orig_image = np.array(deepcopy(image))  # ro furteher plot
        sys.stdout.write('image has been gathered\n')

        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])

        self.image = self.transform(image)
        self.image = torch.unsqueeze(self.image, 0)  # create minibatch

        if self.cuda:
            # covert to cuda tensor type
            self.image = Variable(self.image.type(self.Tensor))
            self.image.cuda()

    def classify(self) -> None:
        """
        classifies and saves an image with label from the classification

        """

        film_urls_by_name = [
            name for film_type in self.film_names for name in film_type.keys()]

        with torch.no_grad():

            outputs = self.model(self.image)
            ps = torch.exp(outputs)
            _, predicted = torch.max(ps, 1)

        label = film_urls_by_name[predicted]
        sys.stdout.write(f'the label is: {label}\n')

        self._save_image(label=label, path=self.output_path)
        sys.stdout.write(f'image has been saved to: {self.output_path}\n')

    def _load_model(self, pretrained: bool = True) -> None:
        """
        loads model
        and sets it into inference mode

        input:
            pretrained - which model to load
        """

        if pretrained:
            # change last softamx if not NLLLoss
            self.model = Net(self.pretrained_model_name, self.num_classes)

        else:
            self.model = ResNet(ResBlock, in_channels=3,
                                make_conv=_add_conv, classes=self.num_classes)

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()  # inference mode

        # set to no grad mode
        for param in self.model.parameters():
            param.requires_grad = False

        # pass to cuda
        if self.cuda:
            self.model.cuda()

    def _save_image(self, label: str, path: str = None, showQ: bool = False):
        """
        transforms image to plt format and save
        adds label and saves

        input:
            label - label of an image
            path - save dir
        """

        plt.imshow(self.orig_image)
        plt.title(label)
        plt.legend()
        plt.axis('off')

        if showQ:
            plt.show()

        if path is not None:
            path = os.path.join(path, f'{label}.png')

        else:
            path = f'{label}.png'

        plt.savefig(path)


def main():

    film_path = ''
    model_path = ''
    output_path = ''
    images = ['',
              '']

    classifier = Classifier(model_path=model_path,
                            film_classes_path=film_path,
                            output_path=output_path)

    for image in images:
        classifier.load_image(input_image_path=image,
                              std=cfg.TEST_STD, mean=cfg.TEST_MEAN)
        classifier.classify()


# test and results
if __name__ == '__main__':
    main()

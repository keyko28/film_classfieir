"""
module to compile and return pretrained net with excluded fc layers
"""


from torch.nn.modules.linear import Linear
from torchvision import models
from typing import Any, Union
from typing import List, Union
import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):

    def __init__(self, model_name: str, num_classes: int, is_NLLLos: bool = True) -> None:
        super(Net, self).__init__()

        self.model_name = model_name

        self.model = Net._load_model(self.model_name)
        input_features = self.model.fc.in_features

        # need to add log softmax layer if nlllos is used
        if is_NLLLos:
            fc = nn.Sequential(
                Linear(input_features, num_classes), nn.LogSoftmax(dim=1))

        else:
            fc = nn.Sequential(Linear(input_features, num_classes))

        self.model.fc = fc

    def forward(self, x: torch.Tensor):

        out = self.model(x)
        return out

    @staticmethod
    def _load_model(name: str) -> Any:
        """
        loads model
        input:
            name : model name to load. 

        name Should be one of a resnet family
        others has not been tested yet
        """

        if name not in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            raise KeyError('only the resnet family is supported')

        model_type = getattr(models, name, None)

        model = model_type(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        return model


# tests
if __name__ == '__main__':

    from PIL import Image
    import numpy as np
    import torchvision.transforms as transforms
    from torch.autograd import Variable

    imsize = 256
    loader = transforms.Compose([transforms.Resize(256),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomRotation(25),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()])

    path = ''  # full path of an image
    image = Image.open(path)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = torch.unsqueeze(image, 0)

    model = Net('resnet152', 10)
    model.eval()

    with torch.no_grad():
        predict = model(image)

    print(predict)

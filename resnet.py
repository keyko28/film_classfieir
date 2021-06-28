"""
module to build a resnet
only downsample techique has been developed
for more than resnet18 a bottleneck must be created
or use tesla v100
"""


from typing import List, Union
import torch.nn as nn
import torch.nn.functional as F
import torch


def _add_conv(in_channels: int,
              out_channels: int,
              kernel_size: Union[int, tuple],
              stride: Union[int, tuple],
              padding: int) -> nn.Conv2d:
    """
    returns conv2d layer
    input:
        in_channels - input filters num
        out_channels - output filters num 
        kernel_size
        stride
        padding
    output:
        nn.Conv2d - a layer of a network
    """

    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=False)


class ResBlock(nn.Module):
    """
    provides methods for creation of
    resblocks with downscale factors 
    """

    expansion = 1  # class attribute

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple],
                 stride: Union[int, tuple],
                 padding: int,
                 downsample: Union[nn.Sequential, None],
                 make_conv: callable) -> None:
        """
        input:
            in_channels - input filters num 
            out_channels - output filters num 
            kernel_size
            stride
            padding
            downsample - expected None or Sequanital NN part that downscales params
            make_conv - expected an external function
        """

        super(ResBlock, self).__init__()

        self._add_conv = make_conv

        self.conv1 = self._add_conv(
            in_channels, out_channels, kernel_size, stride, padding)  # input, output = filters
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()

        self.conv2 = self._add_conv(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding)  # out_put_ out_put

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        short_cut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            short_cut = self.downsample(short_cut)

        out += short_cut
        out = self.prelu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 resblock: ResBlock,
                 in_channels: int = 3,
                 out_channels: int = 64,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 res_blocks: List[int] = [2, 2, 2, 2],
                 classes: int = 12,
                 make_conv: callable = None,
                 is_NLLLos: bool = True) -> None:
        """
        input:
            resblock an instance of a resblock class
            in_channels - input filters num 
            out_channels - output filters num 
            kernel_size
            stride
            padding
            res_blocks - List of [2, 2, 2, 2] used to define number of resblock to create
            classes - amount of classes at the tail of a net
            make_conv - expected an external function

        """

        super(ResNet, self).__init__()

        self.in_filters = 64
        self.default_kernel_size = kernel_size
        self.default_padding = padding
        self.default_stride = stride
        self.res_blocks = res_blocks
        self.classes = classes
        self._add_conv = make_conv

        # head of a net
        self.conv1 = self._add_conv(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=7,
                                    stride=2,
                                    padding=3)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # add resblocks
        self.layer1 = self._add_resblock(
            resblock=resblock, out_filters=64, blocks=self.res_blocks[0], stride=1)
        self.layer2 = self._add_resblock(
            resblock=resblock, out_filters=128, blocks=self.res_blocks[1], stride=2)
        self.layer3 = self._add_resblock(
            resblock=resblock, out_filters=256, blocks=self.res_blocks[2], stride=2)
        self.layer4 = self._add_resblock(
            resblock=resblock, out_filters=512, blocks=self.res_blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=2)
        self.dropout = nn.Dropout2d(p=0.5, inplace=True)

        # tail of a net
        if is_NLLLos:
            self.fc = nn.Sequential(nn.Linear(self.in_filters, self.classes),
                                    nn.LogSoftmax(dim=1))
        else:
            self.fc = nn.Sequential(nn.Linear(self.in_filters, self.classes))

    def _add_resblock(self,
                      resblock: ResBlock,
                      out_filters: int,
                      blocks: int,
                      stride: Union[int, tuple] = 1) -> None:
        """
        makes resblock of given amount
        input:
            resblock - an instance of a resblock class
            out_filters - output num filter 
            blocks -exact amount of resblocks to create 
            stride -
        """

        downsample = None

        # downsample params for each first block in the sequence
        if stride != 1 or self.in_filters != out_filters * resblock.expansion:

            downsample = nn.Sequential(nn.Conv2d(self.in_filters, out_filters * resblock.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_filters * resblock.expansion))

        layers: list = []
        layers.append(resblock(in_channels=self.in_filters,
                               out_channels=out_filters,
                               kernel_size=self.default_kernel_size,
                               stride=stride,
                               padding=self.default_padding,
                               downsample=downsample,
                               make_conv=_add_conv))

        self.in_filters = out_filters * resblock.expansion  # increment in_filters

        for _ in range(1, blocks):  # add otehrs blocks
            layers.append(resblock(in_channels=self.in_filters,
                                   out_channels=out_filters,
                                   kernel_size=self.default_kernel_size,
                                   stride=self.default_stride,
                                   padding=self.default_padding,
                                   downsample=None,
                                   make_conv=_add_conv))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)

        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x


# tests
if __name__ == '__main__':

    from PIL import Image
    import torchvision.transforms as transforms
    from torch.autograd import Variable

    imsize = 256
    loader = transforms.Compose([transforms.Resize(imsize),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomRotation(25),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()])

    path = ''  # full path of an image
    image = Image.open(path)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = torch.unsqueeze(image, 0)

    model = ResNet(ResBlock, in_channels=3, make_conv=_add_conv)
    model.eval()

    with torch.no_grad():
        predict = model(image)

    print(predict)

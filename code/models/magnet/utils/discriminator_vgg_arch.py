import os
import torch
from collections import OrderedDict
import torch.nn as nn
from torchvision.models import vgg as vgg
from torch.nn import functional as F
import torchvision
from torch.nn.utils import spectral_norm
from functools import partial

VGG_PRETRAIN_PATH = './model/vgg_ckpt/vgg19-dcbb9e9d.pth'
NAMES = {
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'pool5'
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
        'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'pool5'
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    ]
}


def insert_bn(names):
    """Insert bn layer after each conv.

    Args:
        names (list): The list of layer names.

    Returns:
        list: The list of layer names with bn layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            position = name.replace('conv', '')
            names_bn.append('bn' + position)
    return names_bn


class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()
        # [64, 128, 128, 32]
        self.conv0_0 = nn.Conv3d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv3d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm3d(nf, affine=True)
        # [64, 64, 64, 16]
        self.conv1_0 = nn.Conv3d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm3d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv3d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm3d(nf * 2, affine=True)
        # [128, 32, 32, 8]
        self.conv2_0 = nn.Conv3d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm3d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv3d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm3d(nf * 4, affine=True)
        # [256, 16, 16, 4]
        self.conv3_0 = nn.Conv3d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm3d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv3d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm3d(nf * 8, affine=True)
        # [512, 8, 8, 2]
        self.conv4_0 = nn.Conv3d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm3d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv3d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm3d(nf * 8, affine=True)
        # [512, 4, 4, 4]
        # self.linear1 = nn.Linear(512 * 4 * 2, 100)
        self.linear1 = nn.Linear(9216, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class Discriminator_VGG_128_S(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128_S, self).__init__()
        # [64, 128, 128, 32]
        self.conv0_0 = nn.Conv3d(in_nc, nf, (3, 3, 3), (1, 1, 1), 1, bias=True)
        self.conv0_1 = nn.Conv3d(nf, nf, (4, 4, 4), (2, 2, 2), 1, bias=False)
        self.bn0_1 = nn.BatchNorm3d(nf, affine=True)
        # [64, 64, 64, 16]
        self.conv1_0 = nn.Conv3d(nf, nf * 2, (3, 3, 3), (1, 1, 1), 1, bias=False)
        self.bn1_0 = nn.BatchNorm3d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv3d(nf * 2, nf * 2, (4, 4, 4), (2, 2, 2), 1, bias=False)
        self.bn1_1 = nn.BatchNorm3d(nf * 2, affine=True)
        # [128, 32, 32, 8]
        self.conv2_0 = nn.Conv3d(nf * 2, nf * 4, (3, 3, 3), (1, 1, 1), 1, bias=False)
        self.bn2_0 = nn.BatchNorm3d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv3d(nf * 4, nf * 4, (4, 4, 4), (2, 2, 2), 1, bias=False)
        self.bn2_1 = nn.BatchNorm3d(nf * 4, affine=True)

        # [256, 1, 4, 4]
        self.linear1 = nn.Linear(256 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_nc, nf=64, kw=4, n_layers=3, norm_layer=nn.BatchNorm3d):
        super(NLayerDiscriminator, self).__init__()
        # no need to use bias as BatchNorm3d has affine parameters
        if type(norm_layer) == partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = kw
        padw = 1
        # sequence = [nn.Conv3d(in_nc, nf, kernel_size=(1, kw, kw), stride=(1, 2, 2), padding=(0, 1, 1)),
        #             nn.LeakyReLU(0.2, True)]
        sequence = [nn.Conv3d(in_nc, nf, kernel_size=(kw, kw, kw), stride=(2, 2, 2), padding=(1, 1, 1)),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        # gradually increase the number of filters
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(nf * nf_mult_prev, nf * nf_mult, kernel_size=kw,
                          stride=2, padding=padw, bias=use_bias),
                norm_layer(nf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(nf * nf_mult_prev, nf * nf_mult,
                      kernel_size=(kw, kw, kw), stride=(2, 2, 2), padding=(1, 1, 1), bias=use_bias),
            norm_layer(nf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(nf * nf_mult, 1,
                               kernel_size=(1, kw, kw), stride=(1, 2, 2), padding=(0, 1, 1))]
        # output one channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        in_nc (int): Channel number of inputs. Default: 3.
        nf (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, in_nc, nf, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv3d(in_nc, nf, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv3d(nf, nf * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv3d(nf * 2, nf * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv3d(nf * 4, nf * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv3d(nf * 8, nf * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv3d(nf * 4, nf * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv3d(nf * 2, nf, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv3d(nf, nf, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv3d(nf, nf, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv3d(nf, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = self.lrelu(self.conv0(x))
        x1 = self.lrelu(self.conv1(x0))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='trilinear', align_corners=False)
        x4 = self.lrelu(self.conv4(x3))

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='trilinear', align_corners=False)
        x5 = self.lrelu(self.conv5(x4))

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='trilinear', align_corners=False)
        x6 = self.lrelu(self.conv6(x5))

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = self.lrelu(self.conv7(x6))
        out = self.lrelu(self.conv8(out))
        out = self.conv9(out)
        return out


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, in_nc, nf=64, norm_layer=nn.BatchNorm3d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            in_nc (int)  -- the number of channels in input images
            nf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == partial:  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(in_nc, nf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(nf, nf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(nf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(nf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False,
                 remove_pooling=False,
                 pooling_stride=2,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.names = NAMES[vgg_type.replace('_bn', '')]
        if 'bn' in vgg_type:
            self.names = insert_bn(self.names)

        # only borrow layers that will be used to avoid unused params
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(pretrained=False)
            state_dict = torch.load(VGG_PRETRAIN_PATH, map_location=lambda storage, loc: storage)
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(pretrained=True)

        features = vgg_net.features[:max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                # if remove_pooling is true, pooling operation will be removed
                if remove_pooling:
                    continue
                else:
                    # in some cases, we may want to change the default stride
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.range_norm:
            x = (x + 1) / 2
        # Assume input range is [0, 1]
        # Convert the depth to batch to adapt to vgg19 to extract features
        if x.ndim == 5:
            b, c, d, h, w = x.size()
            x = x.contiguous().view(b * d, 1, h, w).repeat(1, 3, 1, 1)
        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output


if __name__ == "__main__":
    in_put = torch.ones(1, 1, 16, 32, 32)
    out = NLayerDiscriminator(1, nf=8, n_layers=3)(in_put)
    # net = net.cuda()
    print(out.shape)

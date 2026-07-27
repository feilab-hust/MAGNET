import torch
import torch.nn as nn
import math

gloabal_negative_slope=0.2
is_bias=False

class PixelShuffle_tf(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle_tf, self).__init__()
        self.scale=upscale_factor
    def _shuffle_up_tf(self,inputs, scale):
        N, C, H, W = inputs.size()
        inputs = inputs.view(N, scale ** 2, -1, H, W).transpose(2, 1).contiguous()
        inputs = inputs.view(N, C, H, W)
        return nn.functional.pixel_shuffle(inputs, scale)
    def forward(self, input):
        x=self._shuffle_up_tf(input,self.scale)
        return x
def Macron2Stack(x, angRes,view_first=True):
    b,c,h,w=x.shape
    assert b==1,'angle attention needs to reshape tensor to c,view_num,base_h,base_w'
    out = []
    for i in range(angRes):
        for j in range(angRes):
            out.append(x[:, :, i::angRes, j::angRes])
    out = torch.cat(out,0)
    if not view_first:
        out = out.permute(1,0,2,3)
    return out

def SAI2Stack(x, angRes,view_first=True):
    b,c,h,w=x.shape
    bas_h=h//angRes
    bas_w = w // angRes
    # assert b==1,'angle attention needs to reshape tensor to c,view_num,base_h,base_w'
    out = []
    for i in range(angRes):
        for j in range(angRes):
            out.append(x[:, :, i*bas_h:(i+1)*bas_h, j*bas_w:(j+1)*bas_w])
    out = torch.cat(out,0)
    if not view_first:
        out = out.permute(1,0,2,3)
    return out
def do_nothing(x):
     return x


class Conv3D_block(nn.Module):
    """
    "3D convolution block"
    Note: This block contains conv3d --> InstanceNorm (default:not exist) --> activation
    """
    def __init__(self,
                 c_in=32,
                 c_out=32,
                 kernel_size=3,
                 activation = nn.ReLU(),
                 is_bn = False):
        super().__init__()

        n_dim = 3
        self.c_in = c_in
        assert kernel_size in [1,3,5,7],'kernel size must to be odd (1~7)'
        self.kernel_size = kernel_size
        self.padding = (kernel_size-1)//2
        self.stride =1
        self.conv_block=[]
        if is_bn:
            self.conv_block= nn.Sequential(
                                nn.Conv3d(in_channels=c_in, out_channels=c_out,
                                          kernel_size=(kernel_size,)*n_dim,
                                          stride=(self.stride,)*n_dim,
                                          padding=(self.padding,)*n_dim,
                                          bias=is_bias),
                                nn.InstanceNorm3d(c_out, affine=True),
                                activation,
                                )
        else:
            self.conv_block= nn.Sequential(
                                nn.Conv3d(in_channels=c_in, out_channels=c_out,
                                          kernel_size=(kernel_size,)*n_dim,
                                          stride=(self.stride,)*n_dim,
                                          padding=(self.padding,)*n_dim,
                                          bias=is_bias),
                                activation,
                                )

    def forward(self, x):
        return self.conv_block(x)


class Linear(nn.Module):
    """
    Dummy module that do nothing to the inputs.
    """
    def __init__(self, input_ch=1, *args, **kwargs):
        del args, kwargs
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = input_ch

    def forward(self, x):
        return x

    def load_params(self, ckpt: dict):
        pass

    def get_states(self):
        return {}


class UNetn3D(nn.Module):
    """
    3D UNet structure feature extractor.

    Note:
        The default downsampling ratio is "2".
        base size is used to compute the intermediate pathsize
    Args:
        in_channels: The number of channels of input tensor
        out_channels: The  number of channels of output tensor
        base_size: list  (depth,height,width)
        n_depth: The depth of Unet-block
        n_conv_per_depth: The number of convolution blocks in each layer of Unet
        activation: The activation function of convolution blocks
        n_filter_base: the number of base-filter in Unet. (e.g. 8-->16-->32)
        kernel_size: The size of convolution kernels
    Shape:
        - Input: : (N, in_channels, D, H, W)
        - Output: : (N, out_channels, D, H, W)

    """
    def __init__(self,
                 # required parameters
                 in_channels=1,
                 n_depth=2,
                 n_filter_base=16,
                 n_conv_per_depth=2,
                 out_channels=8,
                 base_size=(160, 160, 160),

                 ## not required
                 kernel_size=3,
                 activation=nn.ReLU(),
                 # activation = nn.LeakyReLU(0.2),
                 is_bn=False,
                 *args, **kwargs):
        super().__init__()
        self.n_depth = n_depth
        self.n_conv_per_depth = n_conv_per_depth
        self.activation = activation
        self.n_filter_base = n_filter_base
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.is_bn = is_bn
        self.base_size = base_size
        self.out_channels = out_channels
        self.pool_ksize = 2
        self.pool_pad = 0

        # calculate the patchsize
        if self.base_size is not None:
            _d = base_size[0]
            _h = base_size[1]
            _w = base_size[2]
            self.en_size_list = [[_d, _h, _w]]
            for i in range(1, self.n_depth):
                _d = math.floor((_d + 2 * self.pool_pad - (self.pool_ksize - 1) - 1) / 2 + 1)
                _h = math.floor((_h + 2 * self.pool_pad - (self.pool_ksize - 1) - 1) / 2 + 1)
                _w = math.floor((_w + 2 * self.pool_pad - (self.pool_ksize - 1) - 1) / 2 + 1)
                self.en_size_list.append([_d, _h, _w])
        self.en_block, self.middle_blocks, self.de_block = self._build_layers()
        pass

    def _build_layers(self):

        en_block = []

        # input layer 0
        en_block.extend(
            [Conv3D_block(c_in=self.in_channels, c_out=self.n_filter_base, kernel_size=self.kernel_size,
                          activation=self.activation, is_bn=self.is_bn)])
        for i in range(1, self.n_conv_per_depth):
            en_block.extend(
                [Conv3D_block(c_in=self.n_filter_base, c_out=self.n_filter_base, kernel_size=self.kernel_size,
                              activation=self.activation, is_bn=self.is_bn)])
        en_block.extend(
            [nn.MaxPool3d(kernel_size=self.pool_ksize, stride=2, padding=self.pool_pad)]
        )
        # encoder layer 1~ n-1
        for n in range(1, self.n_depth):
            for i in range(self.n_conv_per_depth):
                if i == 0:
                    c_in = self.n_filter_base * 2 ** (n - 1)
                    c_out = self.n_filter_base * 2 ** n
                else:
                    c_in = self.n_filter_base * 2 ** n
                    c_out = c_in
                en_block.extend([
                    Conv3D_block(c_in=c_in,
                                 c_out=c_out,
                                 kernel_size=self.kernel_size,
                                 activation=self.activation,
                                 is_bn=self.is_bn),
                ]
                )
            en_block.extend(
                [nn.MaxPool3d(kernel_size=self.pool_ksize, stride=2, padding=self.pool_pad)]
            )
        ## middle blocks
        middle_blocks = []
        for i in range(self.n_conv_per_depth):
            if self.n_conv_per_depth != 1:
                if i == 0:
                    c_in = self.n_filter_base * 2 ** (self.n_depth - 1)
                    c_out = self.n_filter_base * 2 ** self.n_depth
                elif i == self.n_conv_per_depth - 1:
                    c_in = self.n_filter_base * 2 ** self.n_depth
                    c_out = self.n_filter_base * 2 ** (self.n_depth - 1)
                else:
                    c_in = self.n_filter_base * 2 ** self.n_depth
                    c_out = c_in
            else:
                c_in = self.n_filter_base * 2 ** (self.n_depth - 1)
                c_out = self.n_filter_base * 2 ** max(0, self.n_depth - 1)
            middle_blocks.extend(
                [
                    Conv3D_block(
                        c_in=c_in,
                        c_out=c_out,
                        kernel_size=self.kernel_size,
                        activation=self.activation,
                        is_bn=self.is_bn)
                ]
            )
        de_block = []
        for n in reversed(range(self.n_depth)):
            max_in_c = self.n_filter_base * 2 ** (n + 1)
            temp_c = self.n_filter_base * 2 ** n
            min_out_c = self.n_filter_base * 2 ** (n - 1) if n > 0 else self.n_filter_base
            for i in range(self.n_conv_per_depth - 1):
                if i == 0:
                    c_in = max_in_c
                    c_out = temp_c
                else:
                    c_in = temp_c
                    c_out = temp_c
                de_block.extend([
                    Conv3D_block(
                        c_in=c_in,
                        c_out=c_out,
                        kernel_size=self.kernel_size,
                        activation=self.activation,
                        is_bn=self.is_bn)])

            de_block.extend([
                Conv3D_block(
                    c_in=temp_c,
                    c_out=min_out_c,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    is_bn=self.is_bn)])
        en_block = nn.Sequential(*en_block)
        middle_blocks = nn.Sequential(*middle_blocks)
        de_block = nn.Sequential(*de_block)
        return en_block, middle_blocks, de_block

    def forward(self, x):
        '''
        input x: dims: (b,c,d,h,w)
        '''
        # encoder
        encoder_layers = []
        for en_idx, _en_layers in enumerate(self.en_block):
            x = _en_layers(x)
            # if en_idx == self.n_conv_per_depth-1 or \
            # (en_idx!=0 and ((en_idx+2) % (self.n_conv_per_depth+1) ==0)):
            if (en_idx + 2) % (self.n_conv_per_depth + 1) == 0:
                # print(f'en_idx={en_idx}')
                encoder_layers.append(x)

        # middle conv
        x = self.middle_blocks(x)

        # decoder

        # for de_idx in reversed(range(self.n_depth)):
        _skip_idx = self.n_depth - 1
        for de_idx, _de_layers in enumerate(self.de_block):
            if de_idx % self.n_conv_per_depth == 0:
                x = nn.functional.interpolate(x, size=self.en_size_list[_skip_idx])
                long_skip = encoder_layers[_skip_idx]
                _skip_idx = _skip_idx - 1
                x = torch.cat([long_skip, x], dim=1)
                # print(f'initial_concat={de_idx}')
            x = _de_layers(x)
        return x

    def load_params(self, ckpt: dict):
        self.load_state_dict(ckpt)

    def get_states(self):
        return self.state_dict()



class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv3d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


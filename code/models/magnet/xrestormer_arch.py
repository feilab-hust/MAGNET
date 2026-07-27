import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numbers
from einops import rearrange
import math
from collections import OrderedDict
import os
import numpy as np
from datetime import datetime

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class PromptPredictor(nn.Module):
    def __init__(self, z_channels=16, task_dim=5, embed_dim=176):
        super().__init__()
        self.task_embed = nn.Linear(task_dim, z_channels)

        self.fuse = nn.Sequential(
            nn.Conv2d(z_channels * 2, 64, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),             # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1)                 # output: 176 channels
        )

    def forward(self, z_sample_prompt, task_id):
        """
        z_sample_prompt: [B, 16, 128, 128]
        task_id: [B, 5]
        return: [B, 176, 32, 32]
        """
        B, C, H, W = z_sample_prompt.shape
        task_feat = self.task_embed(task_id).unsqueeze(-1).unsqueeze(-1).expand_as(z_sample_prompt)
        x = torch.cat([z_sample_prompt, task_feat], dim=1)  # → [B, 32, 128, 128]
        return self.fuse(x)

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
#############################20241122#############################################
# 定义Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # 如果输入和输出通道不一致，增加1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# 定义Prompt Encoder
class PromptEncoder(nn.Module):
    def __init__(self, input_channels, num_blocks=3, initial_channels=44):
        super(PromptEncoder, self).__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, initial_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.ModuleList()
        channels = initial_channels
        for i in range(num_blocks):
            self.blocks.append(ResidualBlock(channels, channels * 2, stride=2))
            channels *= 2

    def forward(self, x):
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        return x
#############################20241120#############################################
class PromptLatentBlocks(nn.Module):
    def __init__(self, blocks):
        """
        A container to sequentially apply the blocks and pass z_sample, z_target to each block.
        """
        super(PromptLatentBlocks, self).__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, z_sample, z_target):
        """
        Apply each block in sequence, passing z_sample and z_target to each one.
        """
        for block in self.blocks:
            x = block(x, z_sample, z_target)  # 传递 z_sample 和 z_target
        return x
## Prompt Cross-Attention (PCA)
class PromptCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(PromptCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1x1 convolutions for Q, K, V
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, z_in, z_sample, z_target):
        """
        Args:
            z_in-->q: Input representation, shape (B, C, H, W)
            z_sample-->k: Prompt input embedding, shape (B, C, H, W)
            z_target-->v: Prompt target embedding, shape (B, C, H, W)
        """
        b, c, h, w = z_in.shape

        # Compute Q, K, V
        q = self.q_proj(z_in)  # (B, C, H, W)
        k = self.k_proj(z_sample)  # (B, C, H, W)
        v = self.v_proj(z_target)  # (B, C, H, W)

        # Reshape for multi-head attention
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Normalize Q, K
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # Compute output
        out = (attn @ v)  # (B, num_heads, C/num_heads, H*W)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # Project output back to input dimension
        out = self.project_out(out)
        return out




##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
## Overlapping Cross-Attention (OCA)
class OCAB(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias):
        super(OCAB, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        #split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_spatial_heads), (qs, ks, vs))

        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = (spatial_attn @ vs)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_spatial_heads, h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)

        # merge spatial and channel
        out = self.project_out(out)

        return out

############################20241120#########################################
class PCABlock(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_channel_heads, num_spatial_heads, spatial_dim_head, ffn_expansion_factor, bias, LayerNorm_type):
        """
        PCABlock module implementing LayerNorm -> MDTA -> LayerNorm -> PCA -> LayerNorm -> FFN sequence.
        """
        super(PCABlock, self).__init__()

        # Multi-DConv Transposed Self-Attention (MDTA)
        self.spatial_attn = OCAB(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias)
        self.channel_attn = ChannelAttention(dim, num_channel_heads, bias)

        # Prompt Cross-Attention (PCA)
        self.pca = PromptCrossAttention(dim, num_channel_heads, bias)

        # Feed Forward Network (FFN)
        self.pca_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)

        # LayerNorms
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.norm5 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x, z_sample, z_target):
        # deprecated
        # x = x + self.channel_attn(self.norm1(x))
        # x = x + self.pca(self.norm2(x), z_sample, z_target)
        # x = x + self.pca_ffn(self.norm3(x))
        # x = x + self.spatial_attn(self.norm4(x))
        # x = x + self.spatial_ffn(self.norm5(x))

        x = x + self.spatial_attn(self.norm1(x))
        x = x + self.pca(self.norm2(x), z_sample, z_target)
        x = x + self.pca_ffn(self.norm3(x))
        # TSAB
        x = x + self.channel_attn(self.norm4(x))
        x = x + self.channel_ffn(self.norm5(x))

        return x
##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_channel_heads, num_spatial_heads, spatial_dim_head, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()


        self.spatial_attn = OCAB(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias)
        self.channel_attn = ChannelAttention(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.spatial_ffn(self.norm4(x))
        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class XRestormer_light(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 48,
        num_blocks = [4,6,6,8],
        num_refinement_blocks = 4,
        channel_heads = [1,2,4,8],
        spatial_heads = [2,2,3,4],
        overlap_ratio=[0.5, 0.5, 0.5, 0.5],
        window_size = 8,
        spatial_dim_head = 16,
        bias = False,
        ffn_expansion_factor = 2.66,
        LayerNorm_type = 'WithBias',    ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        scale = 1,
        align_feature=16
    ):

        super(XRestormer_light, self).__init__()
        print("Initializing XRestormer")
        self.scale = scale

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[1],  num_channel_heads=channel_heads[1], num_spatial_heads=spatial_heads[1], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), window_size = window_size, overlap_ratio=overlap_ratio[3],  num_channel_heads=channel_heads[3], num_spatial_heads=spatial_heads[3], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])


        # self.prompt_align = prompt_aligner(in_ch=1,out_ch=align_feature)
        self.prompt_encoder = PromptEncoder(input_channels=align_feature, num_blocks=2, initial_channels=dim)
        self.prompt_latent = PromptLatentBlocks(
            [PCABlock(
                dim=int(dim * 2 ** 2),
                window_size=window_size,
                overlap_ratio=overlap_ratio[3],
                num_channel_heads=channel_heads[3],
                num_spatial_heads=spatial_heads[3],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[3])]
        )

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[1],  num_channel_heads=channel_heads[1], num_spatial_heads=spatial_heads[1], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_img,prompt_tensor=None,prompt_enable=False,PromptPD=None):
        if PromptPD and prompt_enable:
            local_task = PromptPD.task_id
            task_id_tensor = F.one_hot(torch.tensor(local_task).to(prompt_tensor[0].device), num_classes=12).float()
            task_id_tensor = task_id_tensor.unsqueeze(0).expand(prompt_tensor[0].shape[0], -1)
            #z_targets = prompt_predictor(prompt_tensor[0], task_id_tensor)
            z_targets = PromptPD(prompt_tensor[0], task_id_tensor)
        ##########################################################################

        if self.scale > 1:
            inp_img = F.interpolate(inp_img, scale_factor=self.scale, mode='bilinear', align_corners=False)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # 2 48 48 64

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)  # 2 96 32 64

        inp_enc_level3 = self.down2_3(out_enc_level2)   # 2 192 12 16
        latent = self.latent(inp_enc_level3)

        ############## Prompt encoding  #####################
        if prompt_enable :
            z_samples = prompt_tensor[0]  # [batch_size, 1, H, W]  normalized LR
            z_samples = self.prompt_encoder(z_samples)
            if PromptPD is None:
                z_targets = prompt_tensor[1]  # [batch_size, 1, H, W]  normalized HR ##########################
                z_targets = self.prompt_encoder(z_targets)  ##########################
            latent_up = F.interpolate(latent,size=(z_targets.shape[-2], z_targets.shape[-1]),mode="bilinear",align_corners=False)
            latent_up = self.prompt_latent(latent_up, z_samples, z_targets)
            latent = F.interpolate(latent_up, size=(latent.shape[-2], latent.shape[-1]), mode="bilinear", align_corners=False)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

    def get_prompt_latents_gt(self, inp_img, prompt_tensor):
        if self.scale > 1:
            inp_img = F.interpolate(inp_img, scale_factor=self.scale, mode='bilinear', align_corners=False)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        latent_pre_prompt = self.latent(inp_enc_level3)

        z_sample = prompt_tensor[0]  # [batch_size, 1, H, W]  normalized LR
        z_target = prompt_tensor[1]
        # 编码 z_samples 和 z_targets
        z_sample_feat = self.prompt_encoder(z_sample)
        z_target_feat = self.prompt_encoder(z_target)

        # 得到带 GT 的 prompt latent
        latent_gt = self.prompt_latent(latent_pre_prompt, z_sample_feat, z_target_feat)

        return latent_gt

    def get_prompt_latents_predict(self, inp_img, prompt_tensor,task_embed):
        if self.scale > 1:
            inp_img = F.interpolate(inp_img, scale_factor=self.scale, mode='bilinear', align_corners=False)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        latent_pre_prompt = self.latent(inp_enc_level3)

        z_sample = prompt_tensor  # [batch_size, 1, H, W]  normalized LR
        z_target_feat = task_embed
        # 编码 z_samples 和 z_targets
        z_sample_feat = self.prompt_encoder(z_sample)


        # 得到带 GT 的 prompt latent
        latent_predict = self.prompt_latent(latent_pre_prompt, z_sample_feat, z_target_feat)

        return latent_predict



class XRestormer(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 48,
        num_blocks = [4,6,6,8],
        num_refinement_blocks = 4,
        channel_heads = [1,2,4,8],
        spatial_heads = [2,2,3,4],
        overlap_ratio=[0.5, 0.5, 0.5, 0.5],
        window_size = 8,#8
        spatial_dim_head = 16,
        bias = False,
        ffn_expansion_factor = 2.66,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        scale = 1,
    ):

        super(XRestormer, self).__init__()
        print("Initializing XRestormer")
        self.scale = scale

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)   # convert input c to dim (48)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[1],  num_channel_heads=channel_heads[1], num_spatial_heads=spatial_heads[1], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), window_size = window_size, overlap_ratio=overlap_ratio[2],  num_channel_heads=channel_heads[2], num_spatial_heads=spatial_heads[2], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), window_size = window_size, overlap_ratio=overlap_ratio[3],  num_channel_heads=channel_heads[3], num_spatial_heads=spatial_heads[3], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.prompt_encoder = PromptEncoder(input_channels=1, num_blocks=3, initial_channels=44)
        self.prompt_latent = PromptLatentBlocks(
            [PCABlock(
                dim=int(dim * 2 ** 3),
                window_size=window_size,
                overlap_ratio=overlap_ratio[3],
                num_channel_heads=channel_heads[3],
                num_spatial_heads=spatial_heads[3],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[3])]
        )

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), window_size = window_size, overlap_ratio=overlap_ratio[2],  num_channel_heads=channel_heads[2], num_spatial_heads=spatial_heads[2], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[1],  num_channel_heads=channel_heads[1], num_spatial_heads=spatial_heads[1], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img,prompt_tensor=['None']):

        if self.scale > 1:
            inp_img = F.interpolate(inp_img, scale_factor=self.scale, mode='bilinear', align_corners=False)

        ##############    Encoder  #####################
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        ############## Prompt encoding  #####################
        if not 'None' in prompt_tensor[0] :
            z_samples = prompt_tensor[0]   # [batch_size, 1, H, W]  normalized LR
            z_targets = prompt_tensor[1]   # [batch_size, 1, H, W]  normalized HR
            z_samples = self.prompt_encoder(z_samples)
            z_targets = self.prompt_encoder(z_targets)
            latent = self.prompt_latent(latent, z_samples, z_targets)

        ##############    Decoder  #####################
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


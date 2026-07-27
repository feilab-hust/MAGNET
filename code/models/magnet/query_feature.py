import torch
import unfoldNd
import math
import random
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from scipy.ndimage import zoom

__all__=[
         'query_2d',
         'query_3d',
         'to_pixel_samples',
         'to_pixel_samples_3d',
         'resize_fn',
         'resize_3d_fn',
         'gen_postSR_2D_pair',
         'padding_rescale_2d_imgs'
         ]

def gen_postSR_2D_pair(base_lr_size,img_lr,img_gt,s_factor, train_mode=True):

    # input image : c, h, w
    if train_mode:
        # PostSR network training
        # Output: LR with input_size and GT with s*input_size
        w_lr = base_lr_size
        w_hr = round(w_lr * s_factor)

        # coordinates of randomly cropping
        y0 = random.randint(0, img_lr.shape[-2] - w_hr)
        x0 = random.randint(0, img_lr.shape[-1] - w_hr)

        crop_lr = img_lr[:, y0: y0 + w_hr, x0: x0 + w_hr]
        crop_hr = img_gt[:, y0: y0 + w_hr, x0: x0 + w_hr]

        # down-sample input LR
        if crop_lr.device.type == 'cuda':
            crop_lr = F.interpolate(crop_lr.unsqueeze(0), size=(w_lr,w_lr),mode='bicubic')[0]
        else:
            crop_lr = resize_fn(crop_lr, w_lr)

    else:
        # PostSR network validation
        # Output: LR prompt with the same spatial size as the current model
        # input block and HR prompt with the corresponding scaled field of view.
        #
        # Do not infer h_lr/w_lr from img_gt.shape / s_factor here. A prompt
        # target can be a convenient reference patch (for example 256x256)
        # while the current inference block is 128 with sr=1.8. Inferring from
        # the prompt target would produce floor(256 / 1.8)=142, which later
        # becomes 36 after Restormer downsampling and breaks window attention.
        # The block size is the real LR size used by the current inference.
        h_lr = base_lr_size
        w_lr = base_lr_size

        # crop
        crop_lr = img_lr[:, :round(h_lr * s_factor), :round(w_lr * s_factor)]
        crop_hr = img_gt[:, :round(h_lr * s_factor), :round(w_lr * s_factor)]

        # reSize LR
        if crop_lr.device.type == 'cuda':
            crop_lr = F.interpolate(crop_lr.unsqueeze(0), size=(h_lr, w_lr), mode='bicubic')[0]
        else:
            crop_lr = resize_fn(crop_lr, (h_lr, w_lr))

    return crop_lr, crop_hr


def gen_postSR_3D_pair(base_lr_size,base_lr_depth,img_lr,img_gt,s_factor, train_mode=True):

    if train_mode:
        # PostSR network training
        # Output: LR with input_size and GT with s*input_size
        w_lr = base_lr_size
        w_hr = round(w_lr * s_factor)
        d_lr = base_lr_depth
        d_hr = round(d_lr * s_factor)

        z0 = random.randint(0, img_lr.shape[-3] - d_hr)
        x0 = random.randint(0, img_lr.shape[-2] - w_hr)   # random crop
        y0 = random.randint(0, img_lr.shape[-1] - w_hr)

        crop_lr1 = img_lr[:,
                  z0: z0 + d_hr,
                  x0: x0 + w_hr,
                  y0: y0 + w_hr
                  ]
        crop_hr = img_gt[:,
                  z0: z0 + d_hr,
                  x0: x0 + w_hr,
                  y0: y0 + w_hr
                  ]

        crop_lr = resize_3d_fn(crop_lr1,(d_lr,w_lr,w_lr) )

    else:
        # PostSR network validation
        # Output: cropped LR/GT with s_factor

        d_lr = math.floor(img_gt.shape[-3] / s_factor + 1e-9)
        h_lr = math.floor(img_gt.shape[-2] / s_factor + 1e-9)
        w_lr = math.floor(img_gt.shape[-1] / s_factor + 1e-9)
        crop_lr = img_lr[:,
                   :round(d_lr * s_factor),
                   :round(h_lr * s_factor),
                   :round(w_lr * s_factor)
                   ]  # assume round int

        crop_lr = resize_3d_fn(crop_lr,(d_lr,w_lr,w_lr) )

        crop_hr = img_gt[:,
                 :round(d_lr * s_factor),
                 :round(h_lr * s_factor),
                 :round(w_lr * s_factor)
                 ]


    return crop_lr, crop_hr
def padding_rescale_2d_imgs(img_lr,img_gt,ref_size,s_factor):
    # For Pre-SR, lr and gt would be cropped/padding to ref size and resample LR with s_factor

    # padding or crop the LR-GT pairs to ref_size
    if ref_size > img_gt.shape[-2]:
        padding_sz = ref_size - img_gt.shape[-2]
        img_lr = torch.nn.functional.pad(img_lr, (0, padding_sz, 0, padding_sz), 'reflect')
        img_gt = torch.nn.functional.pad(img_gt, (0, padding_sz, 0, padding_sz), 'reflect')
    else:
        # randomly cropping
        y0 = random.randint(0, img_gt.shape[-2] - ref_size)
        x0 = random.randint(0, img_gt.shape[-1] - ref_size)

        img_lr = img_lr[:, y0:y0+ref_size, x0:x0+ref_size]
        img_gt = img_gt[:, y0:y0+ref_size, x0:x0+ref_size]

    # reSize LR
    h_lr = math.floor(img_gt.shape[-2] / s_factor + 1e-9)
    w_lr = math.floor(img_gt.shape[-1] / s_factor + 1e-9)

    # downsample with "s" and resample to gt size
    img_down = resize_fn(img_lr, (h_lr, w_lr))
    img_down = resize_fn(img_down, (img_gt.shape[-2], img_gt.shape[-1]))

    return img_down, img_gt


def resize_3d_fn(img, size, is_LF=False):
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
    if len(img.shape) == 4:
        img = img.unsqueeze(0)  # (1, C, D, H, W)
    if is_LF:
        img = img.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
    img = F.interpolate(img, size=size, mode='trilinear', align_corners=False)
    return img.squeeze(0)


def resize_fn(img, size):
    '''return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))'''

    #assert img.shape == (1, 128, 128), "输入矩阵形状必须为 (1, 128, 128)"
    if isinstance(size, int):
        size = (size, size)
    zoom_factor = (1, size[-2]/img.shape[-2], size[-1]/img.shape[-1])
    scaled_matrix = zoom(img, zoom_factor, order=3)
    return torch.tensor(scaled_matrix)



def to_pixel_samples(img):
    """ Convert the image to coord-grayscale pairs.
        img: Tensor, (1, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(1, -1).permute(1, 0) #
    return coord, rgb

def to_pixel_samples_3d(img):
    """ Convert the image to coord-grayscale pairs.
        img: Tensor, (1, H, W)
    """
    # c,d,h,w: 3D shape as input
    # c,h,w: 2D shape as input
    coord = make_coord(img.shape[1:])
    assert  img.dim() == 3 or img.dim() == 4,'unsupported image shape'
    rgb = img.view(1, -1).permute(1, 0)
    return coord, rgb

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def query_2d(feat,
             coord=None,
             cell=None,
             input_trans= None,
             MLP_net =None,
             token_weight = (None,None),
             embedder_function = None
             ):
    # feat = x

    feature_unfold=True
    local_ensemble = True
    cell_decode = True

    if feature_unfold:
        feat = F.unfold(feat, 3, padding=1).view(
            feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

    if local_ensemble:
        vx_lst = [-1, 1]  # 用于指定局部模型集成时的水平和垂直偏移量的列表
        vy_lst = [-1, 1]
        eps_shift = 1e-6
    else:
        vx_lst, vy_lst, eps_shift = [0], [0], 0

    # field radius (global: [-1, 1])
    rx = 2 / feat.shape[-2] / 2  # 这里的 2 表示特征图的宽度被划分为的像素数量。
    ry = 2 / feat.shape[-1] / 2  # 这里的 2 表示特征图的高度被划分为的像素数量。

    feat_coord = make_coord(feat.shape[-2:], flatten=False).permute(2, 0, 1) \
        .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[
                                                -2:])  # 生成的 feat_coord 张量的形状是 (batch_size, 2, height, width) LR coordinates meshgrid
    feat_coord = feat_coord.to(feat.device)
    preds = []
    areas = []
    for vx in vx_lst:
        for vy in vy_lst:
            coord_ = coord.clone()
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
            q_feat = F.grid_sample(
                feat, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_coord = F.grid_sample(
                feat_coord, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            rel_coord = coord - q_coord
            rel_coord[:, :, 0] *= feat.shape[-2]
            rel_coord[:, :, 1] *= feat.shape[-1]

            # add PE
            inp = torch.cat([q_feat, rel_coord, *embedder_function(coord)], dim=-1)

            if cell_decode:
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1)

            bs, q = coord.shape[:2]

            pred_trans = input_trans(inp.view(bs * q, -1))

            # add prompt
            if token_weight[0] is not None:
                scale = token_weight[0].unsqueeze(1).expand(bs, q, -1).reshape(bs * q, -1)
                bias = token_weight[1].unsqueeze(1).expand(bs, q, -1).reshape(bs * q, -1)
                pred_trans = pred_trans * (1 + scale) + bias


            pred = MLP_net(pred_trans).view(bs, q, -1)
            preds.append(pred)

            area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
            areas.append(area + 1e-9)

    tot_area = torch.stack(areas).sum(dim=0)
    if local_ensemble:
        t = areas[0];
        areas[0] = areas[3];
        areas[3] = t

        t = areas[1];
        areas[1] = areas[2];
        areas[2] = t
    ret = 0
    for pred, area in zip(preds, areas):
        ret = ret + pred * (area / tot_area).unsqueeze(-1)

    # if imnet is None
    # ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1),
    #             mode='bicubic', align_corners=False)[:, :, 0, :] \
    #             .permute(0, 2, 1)
    return ret

def query_3d(feat,
             coord=None,
             cell=None,
             input_trans= None,
             MLP_net=None,
             embedder_function = None,
             token_weight=(None, None)
             ):

    feature_unfold=False
    local_ensemble = True
    cell_decode = True

    # random batched RGB 32x32 image-shaped input tensor of batch size 64
    kernel_size = 3
    dilation = 1
    padding = 1
    stride = 1

    # define unfold
    unfold_Nd_op = unfoldNd.UnfoldNd(
        kernel_size, dilation=dilation, padding=padding, stride=stride
    )
    # query 3D function

    if feature_unfold:
        feat = unfold_Nd_op(feat).view(feat.shape[0],
                                       feat.shape[1] * 3**3,
                                       feat.shape[2],
                                       feat.shape[3],
                                       feat.shape[4]
                                       )
    if local_ensemble:
        vx_lst = [-1, 1]  # 用于指定局部模型集成时的水平和垂直偏移量的列表
        vy_lst = [-1, 1]
        vz_lst = [-1, 1]
        eps_shift = 1e-6
    else:
        vx_lst, vy_lst, vz_lst,eps_shift = [0], [0],[0], 0

    rz = 2 / feat.shape[-3] / 2  # rz -->d
    rx = 2 / feat.shape[-2] / 2  # rx -->h
    ry = 2 / feat.shape[-1] / 2  # ry -->w

    # repeat coordinates in LR feature

    # feature coordi: b,(d,h,w)order ,x,x,x
    feat_coord = make_coord(feat.shape[2:], flatten=False).permute(3, 0, 1,2).unsqueeze(0).expand(feat.shape[0], 3, *feat.shape[2:])
    feat_coord = feat_coord.to(feat.device)
    pass

    preds = []
    areas = []

    for vz in vz_lst:
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vz * rz + eps_shift
                coord_[:, :, 1] += vx * rx + eps_shift
                coord_[:, :, 2] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, 0,:] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-3]
                rel_coord[:, :, 1] *= feat.shape[-2]
                rel_coord[:, :, 2] *= feat.shape[-1]

                # add PE
                inp = torch.cat([q_feat, rel_coord, *embedder_function(coord)], dim=-1)

                if cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-3]
                    rel_cell[:, :, 1] *= feat.shape[-2]
                    rel_cell[:, :, 2] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]

                pred_trans = input_trans(inp.view(bs * q, -1))

                # add prompt
                if token_weight[0] is not None:
                    scale = token_weight[0].unsqueeze(1).expand(bs, q, -1).reshape(bs * q, -1)
                    bias = token_weight[1].unsqueeze(1).expand(bs, q, -1).reshape(bs * q, -1)
                    pred_trans = pred_trans * (1 + scale) + bias

                pred = MLP_net(pred_trans).view(bs, q, -1)
                preds.append(pred)
                #
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1]* rel_coord[:, :, 2])
                areas.append(area + 1e-9)

    tot_area = torch.stack(areas).sum(dim=0)
    if local_ensemble:
        # 7->0 0->7
        t = areas[0]
        areas[0] = areas[7]
        areas[7] = t

        # 1->6 6->1
        t = areas[1]
        areas[1] = areas[6]
        areas[6] = t

        # 2->5 5->2
        t = areas[2]
        areas[2] = areas[5]
        areas[5] = t

        # 3->4 4->3
        t = areas[3]
        areas[3] = areas[4]
        areas[4] = t
    ret = 0
    for pred, area in zip(preds, areas):
        ret = ret + pred * (area / tot_area).unsqueeze(-1)
    return ret

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .pytorch_ssim import SSIM
from .discriminator_vgg_arch import VGGFeatureExtractor



__all__ = [
    'ssim_k3_loss',
    'ssim_k7_loss',
    'ssim_k11_loss',
    'mae_loss',
    'mse_loss',
    'fre_loss',
    'EPI_loss',
    'edge_loss',
    'PerceptualLoss',
     'random_prj_loss',
    'Charbonnier_loss',
    'tv_loss',
    'hess_loss'

]

ssim_k3_loss = SSIM(window_size=3)
ssim_k7_loss = SSIM(window_size=7)
ssim_k11_loss = SSIM(window_size=11)


def mse_loss(pred,target,device='cuda:0',**kwargs):
    return torch.mean((pred - target)**2)

def fre_loss(pred):
    lambda_freq = 0.001
    # 假设output是2D信号 [batch, channels, height, width]
    fft = torch.fft.fft2(pred)
    magnitude = torch.abs(fft)

    # 创建频率权重矩阵（惩罚高频分量）
    h, w = pred.shape[-2:]
    y_freq = torch.fft.fftfreq(h).abs().view(h, 1).to(pred.device)
    x_freq = torch.fft.fftfreq(w).abs().view(1, w).to(pred.device)
    freq_weights = (y_freq + x_freq) / 2  # 平均频率

    # 高频惩罚
    penalty = lambda_freq * (magnitude * freq_weights).mean()
    return penalty

def Charbonnier_loss(pred,target,eps=1e-3,device='cuda:0',**kwargs):
    diff = torch.add(pred, -target)
    error = torch.sqrt(diff * diff + eps)
    loss = torch.mean(error)
    return loss

def mae_loss(pred,target,device='cuda:0',**kwargs):
    return torch.mean(torch.abs(pred - target))

def EPI_loss(pred,target,device='cuda:0',**kwargs):
    def _SAI2ViewMap(input_tensor, angRes):
        input_tensor = input_tensor[:, 0, ...][..., None]
        _, h, w, _ = input_tensor.shape
        n_num = angRes
        new_h = h // n_num
        new_w = w // n_num
        angRes = n_num
        out = []
        for i in range(angRes):
            out_u = []
            for j in range(angRes):
                temp_view = input_tensor[:, i * new_h:(i + 1) * new_h, j * new_w:(j + 1) * new_w, :]
                out_u.append(temp_view)
            u_list = torch.cat(out_u, 3)
            u_stack = u_list[..., None]
            out.append(u_stack)
        out = torch.cat(out, 4)  # b,h,w,u,v
        return out

    def _gradient(pred):
        D_dx = pred[:, 1:, :, :, :] - pred[:, :-1, :, :, :]
        D_dy = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        D_dax = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        D_day = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        return D_dx, D_dy, D_dax, D_day
    if 'angRes' in kwargs:
        angRes=kwargs['angRes']
    else:
        angRes=15
    if len(pred.shape) != 5:
        pred = _SAI2ViewMap(pred, angRes=angRes)
        target = _SAI2ViewMap(target, angRes=angRes)
    pred_dx, pred_dy, pred_dax, pred_day = _gradient(pred)
    label_dx, label_dy, label_dax, label_day = _gradient(target)
    return mse_loss(pred_dx, label_dx) + mse_loss(pred_dy, label_dy) + mse_loss(pred_dax, label_dax) + mse_loss(pred_day, label_day)

def edge_loss(pred,target,device='cuda:0',**kwargs):

    '''
    2D edge computation
    dims: b,c,h,w
    '''
    if len(pred.shape)==5:
        # compute the first channel of input vol
        pred = pred[:,0,...]
        target = target[:, 0, ...]

    kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
    # padding=  [[0, 0], [1, 1], [1, 1], [0, 0]]
    kernel_x=torch.from_numpy(np.asarray(kernels[0],np.float32)).to(device)
    kernel_y = torch.from_numpy(np.asarray(kernels[1], np.float32)).to(device)

    kernelsX_rep = kernel_x.reshape(-1,1,3,3).repeat(pred.shape[1],1,1,1)
    kernelsY_rep = kernel_y.reshape(-1,1,3,3).repeat(pred.shape[1],1,1,1)
    # # pred_paddi
    pred_edge_x = torch.nn.functional.conv2d(input=pred, weight=kernelsX_rep,padding=1,groups=pred.shape[1])
    pred_edge_y = torch.nn.functional.conv2d(input=pred, weight=kernelsY_rep,padding=1,groups=pred.shape[1])

    target_edge_x= torch.nn.functional.conv2d(input=target, weight=kernelsX_rep,padding=1,groups=pred.shape[1])
    target_edge_y = torch.nn.functional.conv2d(input=target, weight=kernelsY_rep,padding=1,groups=pred.shape[1])

    return (mse_loss(pred_edge_x,target_edge_x)+mse_loss(pred_edge_y,target_edge_y))/2




def random_prj_loss(pred,target,device='cuda:0',**kwargs):

    ## xy
    if len(pred.shape)==5:
        # compute the first channel of input vol
        pred = pred[:,0,...]
        target = target[:, 0, ...]
    batch,depth,height,width = pred.shape
    prj_range = 10
    d_idx = np.random.randint(prj_range//2, depth - prj_range//2)
    h_idx = np.random.randint(prj_range//2, height - prj_range//2)
    w_idx = np.random.randint(prj_range//2, width - prj_range//2)

    pred_prj = [torch.max(pred[:,d_idx- prj_range//2: d_idx+ prj_range//2],dim=1)[0],
                torch.max(pred[:, :, h_idx - prj_range // 2: h_idx + prj_range // 2,:],dim=2)[0],
                torch.max(pred[ :, :,:,w_idx - prj_range // 2: w_idx + prj_range // 2],dim=3)[0],
                ]

    target_prj = [torch.max(target[:,d_idx- prj_range//2: d_idx+ prj_range//2],dim=1)[0],
                torch.max(target[:, :, h_idx - prj_range // 2: h_idx + prj_range // 2,:],dim=2)[0],
                torch.max(target[:, :,:,w_idx - prj_range // 2: w_idx + prj_range // 2],dim=3)[0],
                ]
    loss = 0

    for _pred, _target in zip(pred_prj, target_prj):
            loss= loss+mse_loss(_pred, _target)
    return torch.mean(loss)



class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-6):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        x_max = x.max()
        target_max = target.max()
        if x.ndim == 4:
            x = x.unsqueeze(2)
            target = target.unsqueeze(2)
        if self.losstype == 'l2':
            # return torch.mean(torch.sum((x - target) ** 2, (1, 2, 3, 4)))
            return torch.mean(torch.mean((x - target) ** 2))
        elif self.losstype == 'l1':
            diff = x - target
            # loss = torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3, 4)))
            loss = torch.mean(torch.mean(torch.sqrt(diff * diff + self.eps)))

            return loss
        else:
            print("reconstruction loss type error!")
            return 0


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


'''class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()


    def forward(self, x, y=None):
        temp1 = torch.cat((x[:, :, 1:, :, :], x[:, :, -1, :, :].unsqueeze(2)), 2)
        temp2 = torch.cat((x[:, :, :, 1:, :], x[:, :, :, -1, :].unsqueeze(3)), 3)
        temp3 = torch.cat((x[:, :, :, :, 1:], x[:, :, :, :, -1].unsqueeze(4)), 4)
        temp = (x - temp1) ** 2 / x.size()[2] + (x - temp2) ** 2 / x.size()[3] + (x - temp3) ** 2 / x.size()[4]
        loss = torch.mean(temp)
        return loss'''

class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        b, c, h, w = x.size()
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return self.weight * (tv_h + tv_w) / (b * c * h * w)

tv_loss = TVLoss()

class HessLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(HessLoss, self).__init__()
        self.weight = weight

    def forward(self, y_pred):
        d = y_pred.shape[1]
        h = y_pred.shape[2]
        w = y_pred.shape[3]
        hess_loss = 0

        x = y_pred[:, :, 1:, :] - y_pred[:, :, :h - 1, :]
        y = y_pred[:, :, :, 1:] - y_pred[:, :, :, :w - 1]
        if d > 1:
            z = y_pred[:, 1:, :, :] - y_pred[:, :d - 1, :, :]
            for tv in [x, y, z]:
                hess = tv[:, :, 1:, :] - tv[:, :, :-1, :]
                hess_loss = hess_loss + torch.mean(torch.square(hess))
                hess = tv[:, :, :, 1:] - tv[:, :, :, :-1]
                hess_loss = hess_loss + torch.mean(torch.square(hess))
                hess = tv[:, 1:, :, :] - tv[:, :-1, :, :]
                hess_loss = hess_loss + torch.mean(torch.square(hess))
        else:
            for tv in [x, y]:
                hess = tv[:, :, 1:, :] - tv[:, :, :-1, :]
                hess_loss = hess_loss + torch.mean(torch.square(hess))
                hess = tv[:, :, :, 1:] - tv[:, :, :, :-1]
                hess_loss = hess_loss + torch.mean(torch.square(hess))

        return self.weight * hess_loss

hess_loss = HessLoss()

class SparseLoss(nn.Module):
    def __init__(self):
        super(SparseLoss, self).__init__()

    def forward(self, x):
        x = torch.abs(x)
        loss = torch.mean(x)
        return loss


class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def forward(self, x, target):
        loss_mean = []
        b, c, d, h, w = x.size()
        x = x.contiguous().view(-1, d, h, w)
        target = target.contiguous().view(-1, d, h, w)
        x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
        x_fft = torch.stack((x_fft.real, x_fft.imag), -1)
        target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))
        target_fft = torch.stack((target_fft.real, target_fft.imag), -1)

        _, d, h, w, f = x_fft.size()

        x_fft = x_fft.view(b, c, d, h, w, f)
        target_fft = target_fft.view(b, c, d, h, w, f)
        diff = x_fft - target_fft
        mask_75 = torch.zeros_like(diff)
        mask_75[:, :, d // 8:7 * d // 8, h // 8:7 * h // 8, w // 8:7 * w // 8, :] = 1
        diff = mask_75 * diff
        loss = torch.mean(torch.mean(diff ** 2, (1, 2, 3, 4, 5)))
        # inner_product = (x_fft * target_fft).sum(dim=-1)
        # norm1 = (x_fft.pow(2).sum(dim=-1)+1e-20).pow(0.5)
        # norm2 = (target_fft.pow(2).sum(dim=-1)+1e-20).pow(0.5)
        # cos = inner_product / (norm1*norm2 + 1e-20)
        # loss_mean.append(-1.0*cos.mean())
        # loss_mean = torch.tensor(loss_mean)
        # loss = torch.mean(loss_mean)
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        return percep_loss


class GeoLoss(nn.Module):
    def __init__(self, model, criterion=nn.L1Loss()):
        super(GeoLoss, self).__init__()
        self.m = model
        self.criterion = criterion

    def forward(self, x, flip_dir="h"):
        out1 = self.m(x)
        out2 = self.geometry_ensemble(x, flip_dir)
        loss = self.criterion(out1, out2)
        return loss

    def geometry_ensemble(self, x, flip_dir="h"):
        if flip_dir == "h":
            flip_axes = [3]
        elif flip_dir == "v":
            flip_axes = [2]
        else:
            raise NotImplementedError

        n, c, d, h, w = x.shape

        imgs = torch.zeros((8, n, c, d, h, w)).to(x.device)
        out_imgs = torch.zeros((8, n, c, d, h, w)).to(x.device)
        for i in range(4):
            imgs[i, ...] = torch.rot90(x, i, [3, 4])
        for i in range(4):
            imgs[4 + i, ...] = torch.flip(imgs[i], flip_axes)

        for i in range(8):
            img = imgs[i, ...]
            temp = self.m(img)
            if i < 4:
                out_imgs[i, ...] = torch.rot90(temp, -i, [3, 4])
            else:
                temp1 = torch.flip(temp, flip_axes)
                out_imgs[i, ...] = torch.rot90(temp1, -(i % 4), [3, 4])

        for i in range(1, 8):
            out_imgs[0] += out_imgs[i]
        return out_imgs[0] / 8


def L1_Charbonnier_loss(X, Y):
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + eps)
    loss = torch.sum(error) / torch.numel(error)
    return loss


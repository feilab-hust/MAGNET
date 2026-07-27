import numpy as np
from skimage import metrics
from einops import rearrange
import os
import imageio
import tifffile
import re
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from omegaconf import OmegaConf
import shutil
import math
import random
import ast


def parse_int_sequence(value, name="value"):
    """Parse an int or comma-separated/list-like value into integer items."""
    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = ast.literal_eval(text)
        items = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    else:
        items = [value]
    try:
        return [int(item) for item in items]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain integers, got {value!r}") from error


def task_head_mapping(task_idx, task_list, model_task):
    """Return aligned dataset-task and UNiFMIR-head lists."""
    data_tasks = (
        parse_int_sequence(task_list, "task_list")
        if int(task_idx) == -1 else [int(task_idx)]
    )
    if int(task_idx) == -1 and not data_tasks:
        data_tasks = list(range(9))
    heads = parse_int_sequence(model_task, "model_task/task_id")
    if len(heads) != len(data_tasks):
        raise ValueError(
            "UNiFMIR requires one model_task/task_id per dataset task: "
            f"task_list={data_tasks} ({len(data_tasks)}), "
            f"model_task={heads} ({len(heads)})"
        )
    return data_tasks, heads

def crop_right_top(img, crop_shape=(64, 128, 128)):
    _, D, H, W = img.shape
    cd, ch, cw = crop_shape

    # 起始位置
    start_d = 0                # 从最前面（z=0）开始
    start_h = 0                # 从顶部（y=0）开始
    start_w = W - cw           # 从右边倒数128列开始

    return img[:, start_d:start_d+cd, start_h:start_h+ch, start_w:start_w+cw]
def random_crop_same_block(img1, img2, crop_shape=(64, 128, 128)):
    _, D, H, W = img1.shape
    cd, ch, cw = crop_shape

    # 确保裁剪尺寸不超过原图尺寸
    assert D >= cd and H >= ch and W >= cw, "Crop size must be smaller than image size."

    # 随机起始位置（深度、高度、宽度）
    sd = random.randint(0, D - cd)
    sh = random.randint(0, H - ch)
    sw = random.randint(0, W - cw)

    # 执行裁剪
    cropped1 = img1[:, sd:sd+cd, sh:sh+ch, sw:sw+cw]
    cropped2 = img2[:, sd:sd+cd, sh:sh+ch, sw:sw+cw]

    return cropped1, cropped2
def choose_task(task_idx,task_list,idx_epoch):
    # choose task
    if task_idx == -1:
        nomi_tasks = task_list
        if len(nomi_tasks) != 0:
            nomi_tasks = eval(nomi_tasks)  # customized task pool
        else:
            nomi_tasks = np.linspace(0, 8, num=9, dtype=np.uint8)  # all-task pool
        if isinstance(nomi_tasks, list) or isinstance(nomi_tasks, tuple):
            nomi_idx = idx_epoch % len(nomi_tasks)
            local_task = nomi_tasks[nomi_idx]
        else:
            local_task = nomi_tasks
    else:
        local_task = task_idx

    return local_task

def cleanup_dataloader(loader):
    """安全关闭DataLoader的多进程worker"""
    if loader is not None:
        try:
            # PyTorch 1.7+的关闭方式
            if hasattr(loader, '_iterator'):
                loader._iterator._shutdown_workers()
            loader._iterator = None
        except:
            pass
    del loader
    # torch.cuda.empty_cache()

def save_settings(save_path,args):
    f = os.path.join(save_path, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(save_path ,'config.txt')
        with open(f, 'w', encoding='utf-8') as file:
            with open(args.config, 'r', encoding='utf-8') as config_file:
                file.write(config_file.read())
    # save data.yaml
    # print(os.getcwd())
    yaml_file = os.path.join(os.getcwd(),args.MT_data_config)
    shutil.copy2(yaml_file, os.path.join(save_path, os.path.basename(yaml_file)))

def find_valid_folders(root_path, mode='train'):
    # mode='train' or 'test'
    valid_folders = []
    for root, dirs, files in os.walk(root_path):
        if mode in dirs:
            valid_folders.append(root)
    return valid_folders
def generate_boundary_idx(global_size=(161,480,480),roi_size=(80,160,160),is_fixed_sampling=False):
    [D,H,W]=global_size
    br_size=roi_size
    c_d_idx = np.random.randint(br_size[0], D - br_size[0])
    c_h_idx = np.random.randint(br_size[1], H - br_size[1])
    c_w_idx = np.random.randint(br_size[2], W - br_size[2])
    if not is_fixed_sampling:
        return \
        c_d_idx - br_size[0], \
        c_d_idx + br_size[0], \
        c_h_idx - br_size[1], \
        c_h_idx + br_size[1], \
        c_w_idx - br_size[2], \
        c_w_idx + br_size[2]
    else:
        return \
            D//2 - br_size[0], \
            D//2 + br_size[0], \
            H//2 - br_size[1], \
            H//2 + br_size[1], \
            W//2 - br_size[2], \
            W//2 + br_size[2]

def str2value(x):
    if isinstance(x, str):
        return eval(x)
    else:
        return x
def cal_metrics(label, out):

    U, V, C, H, W = label.size()

    label_y = label.squeeze(2).data.cpu().numpy().clip(0, 1)
    out_y = out.squeeze(2).data.cpu().numpy().clip(0, 1)

    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')
    for u in range(U):
        for v in range(V):
            PSNR[u, v] = metrics.peak_signal_noise_ratio(label_y[u, v, :, :], out_y[u, v, :, :])
            SSIM[u, v] = metrics.structural_similarity(label_y[u, v, :, :], out_y[u, v, :, :], gaussian_weights=True)

    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean

def cal_recon_metrics(label, out):
    d, H, W = label.size()

    label_y = label.data.cpu().numpy().clip(0, 1)
    out_y = out.data.cpu().numpy().clip(0, 1)

    PSNR = np.zeros(shape=(d, 1), dtype='float32')
    SSIM = np.zeros(shape=(d, 1), dtype='float32')
    for d_idx in range(d):
        PSNR[d_idx, 0] = metrics.peak_signal_noise_ratio(label_y[d_idx, :, :], out_y[d_idx, :, :])
        SSIM[d_idx, 0] = metrics.structural_similarity(label_y[d_idx, :, :], out_y[d_idx, :, :], gaussian_weights=True)

    PSNR_mean = PSNR.sum() / np.sum(PSNR > 0)
    SSIM_mean = SSIM.sum() / np.sum(SSIM > 0)

    return PSNR_mean, SSIM_mean

def norm_fixed(im):
    assert im.dtype in [np.uint8, np.uint16]
    x = im.astype(np.float32)
    max_ = 255. if im.dtype == np.uint8 else 65535.
    # x = x / (max_ / 2.) - 1.
    x = x / (max_)
    return x

def norm_shift(x):
    x = x.astype(np.float32)
    max_ = np.max(x) * 1.1
    # max_ = 255.

    x = x / (max_ / 2.)
    x = x - 1
    return x

def Macron2LF4D(x,angRes):
    x = x.astype(np.float32)

    out=[]
    for u in range(angRes):
        out_v=[]
        for v in range(angRes):
            temp_view=x[u::angRes,v::angRes]
            out_v.append(temp_view)
        v_list=np.stack(out_v,axis=0)
        v_stack=v_list[np.newaxis,...]
        out.append(v_stack)
    out=np.stack(out,axis=0)
    out=np.transpose(out,axes=[0,2,1,3,4])

    return out

def Macron2LF4D_tensor(x,angRes=15,scale_factor=1):

    out=[]
    for u in range(angRes):
        out_v=[]
        for v in range(angRes):
            temp_view=x[...,u::angRes,v::angRes]
            out_v.append(temp_view)
        v_list=torch.cat(out_v,dim=0)
        if scale_factor!=1:
            v_list=torch.nn.functional.interpolate(v_list, scale_factor=scale_factor)
        out.append(v_list)
    out=torch.cat(out,dim=1)
    # torch.nn.functional.interpolate
    # out=out.transpose([0,2,1,3,4])

    return out[None,...]

def Rearrange3D(image):
    image = np.squeeze(image) # remove channels dimension
    #print('reshape : ' + str(image.shape))
    depth, height, width = image.shape
    image_re = np.zeros([height, width, depth])
    for d in range(depth):
        image_re[:,:,d] = image[d,:,:]
    return image_re

def mkdir_or_exsist(path,verbose=False):

    def _create_folder(_path):
        if not os.path.exists(_path):
            if verbose:
                print("[*] creates %s ..." % _path)
            os.makedirs(_path,exist_ok=True)
        else:
            if verbose:
                print("[*] Exist %s ..." % _path)

    if isinstance(path,list):
        for _p in path:
            _create_folder(_p)
    else:
        _create_folder(path)

def normalize_percentile(im, low=0, high=100, clip=False, is_random=False, dtype=np.float32):
    if is_random:
        _p_low = np.random.uniform(0.1, 0.5)
        p_low = np.percentile(im, _p_low)

        _p_high = np.random.uniform(99.5, 99.9)
        p_high = np.percentile(im, _p_high)
    else:
        p_low = np.percentile(im, low)
        p_high = np.percentile(im, high)
    eps = 1e-6
    '''if p_high - p_low == 0:
        print(im.max())'''
    if dtype is not None:
        x = ((im - p_low) / (p_high - p_low + eps)).astype(np.float32)
    else:
        x = ((im - p_low) / (p_high - p_low + eps))
    if clip:
        x[x > 1.0] = 1.0
        x[x < .0] = .0

    if dtype is not None:
        x = x.astype(dtype)

    return x


def normalize_mean_std(im, target_mean=0.0, target_std=0.5, eps=1e-6, dtype=np.float32):

    im = im.astype(np.float32)

    mean = np.mean(im)
    std = np.std(im)

    # 防止 std 为 0
    if std < eps:
        std = eps

    x = (im - mean) / std
    x = x * target_std + target_mean

    if dtype is not None:
        x = x.astype(dtype)

    return x

def normalize_percentile_care(im, low=(1,3), high=(99.5,99.9), clip=False, is_random=True, dtype=np.float32):
    """
    CARE-style percentile normalization.

    Parameters
    ----------
    im : np.ndarray
        Input image.
    low : float or tuple
        If tuple, defines (low_min, low_max) for random sampling.
        If float, defines fixed low percentile.
    high : float or tuple
        If tuple, defines (high_min, high_max) for random sampling.
        If float, defines fixed high percentile.
    clip : bool
        Whether to clip output to [0,1].
    is_random : bool
        Whether to sample percentiles randomly within low and high.
    dtype : type
        Output dtype.

    Returns
    -------
    x : np.ndarray
        Normalized image.
    """

    # Determine low percentile
    if isinstance(low, tuple) and is_random:
        _p_low = np.random.uniform(*low)
    else:
        _p_low = low

    # Determine high percentile
    if isinstance(high, tuple) and is_random:
        _p_high = np.random.uniform(*high)
    else:
        _p_high = high

    # Calculate percentile values
    p_low = np.percentile(im, _p_low)
    p_high = np.percentile(im, _p_high)

    if p_high <= p_low:
        # Option 1: force minimal difference
        p_high = p_low + 1e-6
    # Avoid divide by zero
    eps = 1e-20
    x = (im - p_low) / (p_high - p_low + eps)

    # Optional clip
    if clip:
        x = np.clip(x, 0.0, 1.0)

    # Convert dtype
    if dtype is not None:
        x = x.astype(dtype)

    return x
def lf_extract_fn(lf2d, n_num=11, mode='toChannel', padding=False):
    """
    Extract different views from a single LF projection

    Params:
        -lf2d - 2-D light field projection
        -mode - 'toDepth' -- extract views to depth dimension (output format [depth=multi-slices, h, w, c=1])
                'toChannel' -- extract views to channel dimension (output format [h, w, c=multi-slices])
        -padding -   True : keep extracted views the same size as lf2d by padding zeros between valid pixels
                     False : shrink size of extracted views to (lf2d.shape / Nnum);
    Returns:
        ndarray [height, width, channels=n_num^2] if mode is 'toChannel'
                or [depth=n_num^2, height, width, channels=1] if mode is 'toDepth'
    """
    n = n_num
    h, w, c = lf2d.shape
    if padding:
        if mode == 'toDepth':
            lf_extra = np.zeros([n * n, h, w, c])  # [depth, h, w, c]

            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, i: h: n, j: w: n, :] = lf2d[i: h: n, j: w: n, :]
                    d += 1
        elif mode == 'toChannel':
            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([h, w, n * n])

            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[i: h: n, j: w: n, d] = lf2d[i: h: n, j: w: n]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)
    else:
        new_h = int(np.ceil(h / n))
        new_w = int(np.ceil(w / n))

        if mode == 'toChannel':

            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([new_h, new_w, n * n])

            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[:, :, d] = lf2d[i: h: n, j: w: n]
                    d += 1

        elif mode == 'toDepth':
            lf_extra = np.zeros([n * n, new_h, new_w, c])  # [depth, h, w, c]

            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, :, :, :] = lf2d[i: h: n, j: w: n, :]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)

    return lf_extra

def get_2d_lf(filename, path, normalize_fn, **kwargs):
    def _LFP2ViewMap(img, angRes):
        img = np.squeeze(img)
        h, w = img.shape
        base_h = h // angRes
        base_w = w // angRes
        VP_ = np.zeros(img.shape, np.float32)
        for v in range(angRes):
            for u in range(angRes):
                VP_[v * base_h:(v + 1) * base_h, u * base_w:(u + 1) * base_w] = img[v::angRes, u::angRes]
        return VP_

    def _ViewMap2LFP(img, angRes):
        img = np.squeeze(img)
        h, w = img.shape
        base_h = h // angRes
        base_w = w // angRes
        LFP_ = np.zeros(img.shape, np.float32)
        for v in range(angRes):
            for u in range(angRes):
                LFP_[v::angRes, u::angRes] = img[v * base_h:(v + 1) * base_h, u * base_w:(u + 1) * base_w]
        return LFP_

    def _identity(img, angRes):
        return img

    # image = imageio.imread(path + filename).astype(np.uint16)
    image = imageio.imread(os.path.join(path,filename)).astype(np.uint16)
    if 'read_type' in kwargs:
        read_type = kwargs['read_type']
    else:
        read_type = None

    if read_type is not None:
        assert 'ViewMap' in read_type or 'LFP' in read_type, 'wrong img type'
        if '1' in read_type:
            trans_func = _identity if 'LFP' in read_type else _ViewMap2LFP
        elif '2' in read_type:
            trans_func = _identity if 'ViewMap' in read_type else _LFP2ViewMap
        else:
            raise Exception('wrong img type')
        image = trans_func(image, angRes=kwargs['angRes'])

    image = image[:, :, np.newaxis] if image.ndim == 2 else image
    return normalize_fn(image)

def Normalize_data(x,is_clip=False,cast_bitdepth=16):

    if is_clip:
        # x[x>1.0]=1.0
        x[x<0]=0
    x = (x - x.min()) / (x.max()- x.min()+1e-20)
    if cast_bitdepth==16:
        max_=65535
        x = np.array(x*max_,np.uint16)
    elif cast_bitdepth==8:
        max_=255
        x = np.array(x * max_, np.uint8)
    else:
        x = x
    return x

def normalize_bench(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    def _normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
        if dtype is not None:
            x = x.astype(dtype, copy=False)
            mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
            ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
            eps = dtype(eps)

        try:
            import numexpr
            x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
        except ImportError:
            x = (x - mi) / (ma - mi + eps)
        # print('normalize_mi_ma_debug: ', mi, ma-mi)

        if clip:
            x = np.clip(x, 0, 1)
        return x

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    # print('minmax: ', mi, ma)
    return _normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)
def load_file_list(path=None, regx='\.npz', printable=True,is_fullpath=False):
    r"""Return a file list in a folder by given a path and regular expression.

    Parameters
    ----------
    path : str or None
        A folder path, if `None`, use the current directory.
    regx : str
        The regx of file name.
    printable : boolean
        Whether to print the files infomation.
    """
    if path is None:
        path = os.getcwd()
    file_list = sorted(os.listdir(path))
    return_list = []
    for _, f in enumerate(file_list):
        if re.search(regx, f):
            if is_fullpath:
                return_list.append(os.path.join(path, f))
            else:
                return_list.append(f)
    return return_list

def easydict_to_dict(obj):
    if not isinstance(obj, edict):
        return obj
    else:
        return {k: easydict_to_dict(v) for k, v in obj.items()}

def load_config(config_path):
    with open(config_path, encoding="utf-8-sig") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = easydict_to_dict(config)
        config = OmegaConf.create(config)
    return config




def is_64_times_power_of_2(size):
    """判断 size 是否是 64 × 2^n 形式"""
    if size % 64 != 0:
        return False
    quotient = size // 64
    return (quotient & (quotient - 1)) == 0  # 检查是否是 2 的幂

def get_nearest_64_times_power_of_2(size):
    """获取大于等于 size 的最小 64 × 2^n"""
    if is_64_times_power_of_2(size):
        return size
    n = math.ceil(math.log2(size / 64))  # 计算最小 n，使得 64 × 2^n >= size
    return 64 * (2 ** n)
def pad_to_64_times_2n(image):
    """
    如果图像尺寸不是 64 × 2^n，则填充到最近的符合尺寸，使用 reflect 填充模式
    :param image: (C, H, W) 的 torch.Tensor
    :return:  填充后的图像
    """
    h, w = image.shape[-2],image.shape[-1]

    # 计算目标尺寸
    new_h = get_nearest_64_times_power_of_2(h)
    new_w = get_nearest_64_times_power_of_2(w)

    # 计算需要填充的大小 (左, 右, 上, 下)
    pad_h = new_h - h
    pad_w = new_w - w
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

    # 进行填充 (使用 reflect 模式)
    padded_image = F.pad(image, padding, mode='reflect')

    return padded_image,padding

import torch.nn as nn
class PromptPredictor(nn.Module):
    def __init__(self, z_channels=16, task_dim=5, embed_dim=176):
        super().__init__()

        self.task_id=0
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

import os
import torch
import random
import math
import tifffile
import numpy as np
from models.magnet.query_feature import (resize_fn,resize_3d_fn,
                                 to_pixel_samples_3d,to_pixel_samples,
                                 gen_postSR_2D_pair,padding_rescale_2d_imgs,)
from torch.utils.data.dataset import Dataset
from torch.utils.data import dataloader
from .utility import normalize_percentile, load_file_list, load_config, str2value, find_valid_folders, \
    random_crop_same_block, crop_right_top, normalize_percentile_care
from omegaconf import ListConfig
import torch.nn.functional as F
from models.magnet.physical_model import *
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader
__all__ = [ 'get_multi_data']
global_seed = 42
P_LOW = 0
P_HIGH = 100
CLIP = True

def _load_text_embedding(dataset_root, required=False):
    """Load FluoResFM's per-dataset embedding as a CPU float tensor."""
    embedding_path = os.path.join(dataset_root, 'txt_embedding.pt')
    if not os.path.isfile(embedding_path):
        if required:
            raise FileNotFoundError('Text embedding not found: %s' % embedding_path)
        return None
    try:
        embedding = torch.load(embedding_path, map_location='cpu', weights_only=True)
    except TypeError:  # PyTorch < 2.0
        embedding = torch.load(embedding_path, map_location='cpu')
    if not isinstance(embedding, torch.Tensor):
        raise TypeError('Expected a tensor in %s, got %s' % (
            embedding_path, type(embedding).__name__))
    if tuple(embedding.shape) != (1, 160, 768):
        raise ValueError('Expected text embedding shape [1, 160, 768], got %s: %s' % (
            tuple(embedding.shape), embedding_path))
    return embedding[0].to(dtype=torch.float32)


def _worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        seed = global_seed + rank * 1000 + worker_id
    else:
        seed = global_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


sub_folder_description=['input','target','psf']
# sub_folder_description=['target_scale','target_scale']

#Template_2D_list = [0, 1, 2, 3, 4, 5, 6]
#Template_3D_list = [7,8]

Template_2D_list = [0, 1, 2, 3, 4, 5, 6]
Template_3D_list = [7,8]

def get_multi_data(task_idx,data_config=None,testonly=False,num_work=0,seed=global_seed,
                   local_rank=0,world_size=1,require_text_embedding=False):
    # Loading multi-data with a form of spare sampled points
    assert os.path.exists(data_config) and data_config.endswith('yaml'), 'data_config should be a .yaml file'
    data_configs = load_config(data_config)
    t = task_idx
    assert (t in Template_2D_list or t in Template_3D_list)
    if testonly:
        loader_train = None
        train_sampler = None
    task_str_list = data_configs.Task_labels
    local_task_str = task_str_list[t]

    prompt_enable = data_configs.prompt_enable
    prompt_test_idx = data_configs.prompt_valid_idx


    if t in Template_2D_list:
        local_dict =  getattr(data_configs.data_info, local_task_str)
        use_npy = bool(local_dict['mem']) if 'mem' in local_dict else False
        dataset_class = LIIF_2D_Restoration_NPY if use_npy else LIIF_2D_Restoration
        patch_size =   str2value( local_dict['patch_size'] )
        eval_patch_size = (
            str2value(local_dict['eval_patch_size'])
            if 'eval_patch_size' in local_dict else None
        )
        data_path = local_dict['data_path']

        [scale_min, scale_max] = local_dict['sr_ratio'] \
            if isinstance(local_dict['sr_ratio'], ListConfig) else [local_dict['sr_ratio'], local_dict['sr_ratio']]

        sample_q = str2value( local_dict['sample_q'] )
        batch_size = local_dict['batch_size']
        repeat_ = local_dict['repeat'] if 'repeat' in local_dict else 1
        data_dim = local_dict['Dims']
        validation_scale = local_dict['sr_ration_val']
        # Post-SR is a property of the validation scale, not of the task name
        # or the removed patch_size_X option.
        post_SR = float(validation_scale) != 1.0
        read_folder_mode = local_dict['data_loading_mode'].upper()  # upper
        discard_folder = local_dict['discard_train_folder']
        pre_normalize = local_dict['pre_normalize']
        pre_scale = local_dict['pre_scale']
        val_num = local_dict['val_num'] if 'val_num' in local_dict else None
        tra_num = local_dict['tra_num'] if 'tra_num' in local_dict else -1

        # load prompt images
        if not testonly:
            my_trainset = dataset_class(
                rootdatapath=data_path,
                inp_size=patch_size,
                scale_min=scale_min,
                scale_max=scale_max,
                augment=True,
                sample_q=sample_q,
                train=True,
                tra_num=tra_num,
                repeat=repeat_,
                read_folder_mode=read_folder_mode,
                discard_folder=discard_folder,
                pre_normalize=pre_normalize,
                pre_scale=pre_scale,
                loading_prompt=prompt_enable,
                prompt_valid_idx=None,
                require_text_embedding=require_text_embedding,

            )
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=my_trainset,
                shuffle=True,
                seed=seed+local_rank,
                rank=local_rank,
                num_replicas=world_size,
                drop_last=True  # 与DataLoader保持一致
            )
            loader_train = dataloader.DataLoader(
                dataset=my_trainset,
                batch_size=batch_size,
                sampler=train_sampler,
                # generator=seed_generator,
                shuffle=False,   # False when Sampler was set
                pin_memory=True,
                num_workers=num_work,
                persistent_workers=False,
                multiprocessing_context='spawn' if num_work > 0 else None,
                drop_last=True,
                worker_init_fn=_worker_init_fn
            )


        loader_test = dataloader.DataLoader(
            dataset_class(
                rootdatapath=data_path,
                inp_size=None,
                scale_min=validation_scale,
                scale_max=validation_scale,
                augment=False,
                sample_q=None,
                train=False,
                val_num=val_num,
                read_folder_mode=read_folder_mode,
                discard_folder=discard_folder,
                pre_normalize=pre_normalize,
                pre_scale=pre_scale,
                loading_prompt=prompt_enable,
                prompt_valid_idx=prompt_test_idx,
                require_text_embedding=require_text_embedding,
            ),
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            # Validation is executed inside a torch.multiprocessing-spawned
            # DDP process. Spawning another worker here on Windows can load a
            # second Intel OpenMP runtime and abort with OMP Error #15.
            num_workers=0)


    elif t in Template_3D_list:

        local_dict = eval('data_configs.data_info.%s'%local_task_str)
        use_npy = bool(local_dict['mem']) if 'mem' in local_dict else False
        dataset_class = LIIF_3D_Restoration_NPY if use_npy else LIIF_3D_Restoration
        data_path = local_dict['data_path']
        depth = local_dict['depth']
        patch_size =   str2value( local_dict['patch_size'] )
        eval_patch_size = (
            str2value(local_dict['eval_patch_size'])
            if 'eval_patch_size' in local_dict else None
        )
        [scale_min, scale_max] = local_dict['sr_ratio'] \
            if isinstance(local_dict['sr_ratio'], ListConfig) else [local_dict['sr_ratio'], local_dict['sr_ratio']]

        sample_q = str2value( local_dict['sample_q'] )
        batch_size = local_dict['batch_size']
        repeat_ = local_dict['repeat'] if 'repeat' in local_dict else 1
        data_dim = local_dict['Dims']
        validation_scale = local_dict['sr_ration_val']
        post_SR = float(validation_scale) != 1.0
        val_num = local_dict['val_num'] if 'val_num' in local_dict else None
        read_folder_mode = local_dict['data_loading_mode'].upper()  # upper
        discard_folder = local_dict['discard_train_folder']
        pre_normalize = local_dict['pre_normalize']
        pre_scale = local_dict['pre_scale']

        if not testonly:

            my_trainset = dataset_class(
                    rootdatapath=data_path,
                    inp_size=patch_size,
                    depth=depth,
                    scale_min=scale_min,
                    scale_max=scale_max,
                    augment=True,
                    sample_q=sample_q,
                    train=True,
                    repeat=repeat_,
                    read_folder_mode=read_folder_mode,
                    discard_folder=discard_folder,
                    pre_normalize=pre_normalize,
                    pre_scale=pre_scale,
                    loading_prompt=prompt_enable,
                    prompt_valid_idx = None,
                    require_text_embedding=require_text_embedding,
                )
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=my_trainset,
                shuffle=True,
                seed=seed+local_rank,
                rank=local_rank,
                num_replicas=world_size,
                drop_last=True
            )
            loader_train = dataloader.DataLoader(
                dataset=my_trainset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=num_work,
                sampler=train_sampler,
                persistent_workers=False,
                multiprocessing_context='spawn' if num_work > 0 else None,
                drop_last=True,
                worker_init_fn=_worker_init_fn
            )

        loader_test = dataloader.DataLoader(
            dataset_class(
                rootdatapath=data_path,
                inp_size=None,
                depth=None,
                scale_min=validation_scale,
                scale_max=validation_scale,
                val_num = val_num,
                augment=False,
                sample_q=None,
                train=False,
                read_folder_mode=read_folder_mode,
                discard_folder=discard_folder,
                pre_normalize=pre_normalize,
                pre_scale=pre_scale,
                loading_prompt=prompt_enable,
                prompt_valid_idx=prompt_test_idx,
                require_text_embedding=require_text_embedding,
            ),
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            # See the 2D loader above. Keep validation in the DDP process;
            # training can still use the configured worker count.
            num_workers=0)



    return (loader_train,
            train_sampler,
            loader_test,
            data_configs.data_norm,
            data_configs.Task_labels[t],
            data_dim,
            prompt_enable,
            post_SR,
            eval_patch_size,
            )


class LIIF_2D_Restoration(Dataset):
    file_regex = '.*.tif'

    def __init__(self,
                rootdatapath='',
                inp_size=None,
                depth=None,
                scale_min=1,
                scale_max=None,
                augment=True,
                sample_q=None,
                repeat=1,
                tra_num = -1,
                val_num = 100,
                train=True,
                read_folder_mode = 'SINGLE',
                discard_folder = None,
                pre_normalize = True,
                pre_scale = True,
                loading_prompt=False,
                prompt_valid_idx = None,
                require_text_embedding=False,
                ):

        self.repeat = repeat

        self.datamin, self.datamax = P_LOW, P_HIGH
        self.inp_size= inp_size
        self.depth = depth
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q
        self.augment = augment
        self.train = train
        self.pre_normalize = pre_normalize
        self.pre_scale=pre_scale
        self.loading_prompt = loading_prompt
        self.require_text_embedding = require_text_embedding
        self.txt_emb = {}

        if read_folder_mode == 'SINGLE':
            folder_list = [rootdatapath]
        elif read_folder_mode == 'MULTI':
            # folder_list = os.listdir(rootdatapath)
            folder_list = find_valid_folders(rootdatapath, mode='train' if self.train else 'test')
        elif read_folder_mode == 'PICK':
            folder_list_all = find_valid_folders(rootdatapath, mode='train' if self.train else 'test')
            # folder_list_all = os.listdir(rootdatapath)
            folder_list = [np.random.choice(list(set(folder_list_all)), replace=False)]   # randomly pick one dir
        else:
            raise ValueError('read_folder_mode must be SINGLE or MULTI or PICK')
        #self.label_emb_dict = torch.load(os.path.join(rootdatapath, 'label.pt'))
        self.data_path = rootdatapath
        # loading filenames
        file_list_lr=[]
        file_list_gt=[]
        file_prompt_lr = []
        file_prompt_gt = []
        label_dir = []
        if read_folder_mode == 'MULTI':
            for _folder in sorted(folder_list):
                assert os.path.exists(_folder)
                # get labels
                local_label = os.path.basename(_folder)
                embedding = _load_text_embedding(_folder, self.require_text_embedding)
                if embedding is not None:
                    self.txt_emb[local_label] = embedding
                if train:
                    if discard_folder is not None and local_label in discard_folder:
                        continue
                    local_path_1 = os.path.join(_folder,'train',sub_folder_description[0])
                    local_path_2 = os.path.join(_folder,'train',sub_folder_description[1])
                else :
                    local_path_1 = os.path.join(_folder,'test',sub_folder_description[0])
                    local_path_2 = os.path.join(_folder,'test',sub_folder_description[1])

                filenames = load_file_list(local_path_1,regx=self.file_regex,is_fullpath=True)
                filenames_gt = load_file_list(local_path_2,regx=self.file_regex,is_fullpath=True)
                assert len(filenames) == len(filenames_gt), 'The num of LR and GT (%s) is not the same (PATH:%s)' % (
                'train' if self.train else 'test',
                local_path_1)

                if not self.train:
                    filenames = filenames[:val_num]
                    filenames_gt = filenames_gt[:val_num]
                elif self.train and tra_num > 0:
                    indices = random.sample(range(len(filenames)), min(tra_num,len(filenames)))
                    filenames = [filenames[i] for i in indices]
                    filenames_gt = [filenames_gt[i] for i in indices]
                else:
                    pass

                file_list_lr.extend(filenames)
                file_list_gt.extend(filenames_gt)
                label_dir.extend([local_label] * len(filenames))

                if self.loading_prompt:
                    # get prompts (default the first pair in prompt folder)
                    filename_prompt_lr = load_file_list(
                        os.path.join(_folder, 'prompt', sub_folder_description[0]), regx=self.file_regex,
                        is_fullpath=True)
                    filename_prompt_gt = load_file_list(
                        os.path.join(_folder, 'prompt', sub_folder_description[1]), regx=self.file_regex,
                        is_fullpath=True)
                    if self.train:
                        # Determine how many prompts and training samples
                        num_prompts = len(filename_prompt_lr)
                        num_samples = len(filenames)

                        if num_prompts >= num_samples:
                            # Enough prompts, just truncate or match
                            selected_prompt_lr = filename_prompt_lr[:num_samples]
                            selected_prompt_gt = filename_prompt_gt[:num_samples]
                        else:
                            # Not enough prompts, randomly expand
                            expanded_indices = random.choices(range(num_prompts), k=num_samples)  # sample with replacement
                            selected_prompt_lr = [filename_prompt_lr[i] for i in expanded_indices]
                            selected_prompt_gt = [filename_prompt_gt[i] for i in expanded_indices]

                        file_prompt_lr.extend(selected_prompt_lr)
                        file_prompt_gt.extend(selected_prompt_gt)
                    else:
                        if prompt_valid_idx is not None:
                            file_prompt_lr.extend([filename_prompt_lr[prompt_valid_idx]] * len(filenames))
                            file_prompt_gt.extend([filename_prompt_gt[prompt_valid_idx]] * len(filenames))
                        else:
                            file_prompt_lr.extend([filename_prompt_lr[0]] * len(filenames))
                            file_prompt_gt.extend([filename_prompt_gt[0]] * len(filenames))
                else:
                    file_prompt_lr = None
                    file_prompt_gt = None
        else:
            assert os.path.exists(os.path.join(rootdatapath))
            # get labels
            local_label = os.path.basename(rootdatapath)
            embedding = _load_text_embedding(rootdatapath, self.require_text_embedding)
            if embedding is not None:
                self.txt_emb[local_label] = embedding
            if train:
                local_path_1 = os.path.join(rootdatapath, 'train', sub_folder_description[0])
                local_path_2 = os.path.join(rootdatapath, 'train', sub_folder_description[1])
            else:
                local_path_1 = os.path.join(rootdatapath, 'test', sub_folder_description[0])
                local_path_2 = os.path.join(rootdatapath, 'test', sub_folder_description[1])

            filenames = load_file_list(local_path_1, regx=self.file_regex, is_fullpath=True)
            filenames_gt = load_file_list(local_path_2, regx=self.file_regex, is_fullpath=True)
            assert len(filenames) == len(filenames_gt), 'The num of LR and GT (%s) is not the same (PATH:%s)' % (
                'train' if self.train else 'test',
                local_path_1)

            if not self.train:
                filenames = filenames[:val_num]
                filenames_gt = filenames_gt[:val_num]
            elif self.train and tra_num>0:
                indices = random.sample(range(len(filenames)), tra_num)
                filenames = [filenames[i] for i in indices]
                filenames_gt = [filenames_gt[i] for i in indices]
            else:
                pass
            file_list_lr.extend(filenames)
            file_list_gt.extend(filenames_gt)
            label_dir.extend([local_label] * len(filenames))

            if self.loading_prompt:
                # get prompts (default the first pair in prompt folder)
                filename_prompt_lr = load_file_list(
                    os.path.join(rootdatapath, 'prompt', sub_folder_description[0]), regx=self.file_regex,
                    is_fullpath=True)
                filename_prompt_gt = load_file_list(
                    os.path.join(rootdatapath, 'prompt', sub_folder_description[1]), regx=self.file_regex,
                    is_fullpath=True)
                if self.train:
                    # Determine how many prompts and training samples
                    num_prompts = len(filename_prompt_lr)
                    num_samples = len(filenames)

                    if num_prompts >= num_samples:
                        # Enough prompts, just truncate or match
                        selected_prompt_lr = filename_prompt_lr[:num_samples]
                        selected_prompt_gt = filename_prompt_gt[:num_samples]
                    else:
                        # Not enough prompts, randomly expand
                        expanded_indices = random.choices(range(num_prompts), k=num_samples)  # sample with replacement
                        selected_prompt_lr = [filename_prompt_lr[i] for i in expanded_indices]
                        selected_prompt_gt = [filename_prompt_gt[i] for i in expanded_indices]

                    file_prompt_lr.extend(selected_prompt_lr)
                    file_prompt_gt.extend(selected_prompt_gt)
                else:
                    if prompt_valid_idx is not None:
                        file_prompt_lr.extend([filename_prompt_lr[prompt_valid_idx]] * len(filenames))
                        file_prompt_gt.extend([filename_prompt_gt[prompt_valid_idx]] * len(filenames))
                    else:
                        file_prompt_lr.extend([filename_prompt_lr[0]] * len(filenames))
                        file_prompt_gt.extend([filename_prompt_gt[0]] * len(filenames))
            else:
                file_prompt_lr = None
                file_prompt_gt = None


        self.file_prompt_lr = file_prompt_lr
        self.file_prompt_gt = file_prompt_gt
        self.filenames=file_list_lr
        self.filenames_gt=file_list_gt
        self.label_dir=label_dir
        pass


    def __len__(self):
        return len(self.filenames) * self.repeat

    def load_2d_imgs(self,path):
        img = self._read_array(path)
        img = np.squeeze(img)

        if self.pre_normalize:
            img = img
        else:
            img = normalize_percentile(img, self.datamin, self.datamax, clip=CLIP)
            # img = normalize_percentile_care(img)
        img = torch.tensor(img).unsqueeze(0)    # c,h,w
        return img

    def _read_array(self, path):
        return np.asarray(tifffile.imread(path), np.float32)


    def __getitem__(self, idx):

        x = self.load_2d_imgs(self.filenames[idx % len(self.filenames)])  # remainder for repeatedly training
        x_gt = self.load_2d_imgs(self.filenames_gt[idx % len(self.filenames)])

        if x.shape[-2] != x_gt.shape[-2] or x.shape[-1] != x_gt.shape[-1]:
            x = resize_fn(x, (x_gt.shape[-2], x_gt.shape[-1]))

        scaled_inp = x.clone()
        scaled_tar = x_gt.clone()
        scaled_inp, scaled_tar = gen_postSR_2D_pair(base_lr_size=self.inp_size,img_lr=scaled_inp,img_gt=scaled_tar,s_factor=1,train_mode=self.train)

        local_label = self.label_dir[idx % len(self.filenames)]
        s = random.uniform(self.scale_min, self.scale_max)

        crop_lr, crop_hr = gen_postSR_2D_pair(base_lr_size=self.inp_size,img_lr=x,img_gt=x_gt,s_factor=s,train_mode=self.train)

        if self.loading_prompt:
            prompt_gt = self.load_2d_imgs(self.file_prompt_gt[idx % len(self.filenames)])
            prompt_lr = self.load_2d_imgs(self.file_prompt_lr[idx % len(self.filenames)])
            # rescale before
            if not self.pre_scale:
                prompt_lr = resize_fn(prompt_lr, (prompt_gt.shape[-2], prompt_gt.shape[-1]))

        else:
            prompt_gt = 'None'
            prompt_lr = 'None'

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            scaled_inp  = augment(scaled_inp)
            scaled_tar  = augment(scaled_tar)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        outdict = {
            'inp': crop_lr,
            'scaled_inp': scaled_inp,
            'scaled_tar': scaled_tar,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'gts': x_gt,
            'sample_shape': crop_hr.shape,
            'sample_label': local_label,
            'prompt': [prompt_lr,prompt_gt],
            's_factor': s,
        }
        if local_label in self.txt_emb:
            outdict['txt_emb'] = self.txt_emb[local_label]
        return outdict


class LIIF_2D_Restoration_NPY(LIIF_2D_Restoration):
    """2D dataset backed by pre-normalized NumPy arrays."""

    file_regex = '.*.npy'

    def __init__(self, *args, **kwargs):
        if not kwargs.get('pre_normalize', True):
            raise ValueError('NPY datasets must use pre_normalize=True')
        super().__init__(*args, **kwargs)

    def _read_array(self, path):
        # mmap lets Windows cache file pages instead of decoding TIFF repeatedly.
        return np.load(path, mmap_mode='r')


class LIIF_3D_Restoration(Dataset):
    file_regex = '.*.tif'

    def __init__(self,
                 rootdatapath='',
                 inp_size=None,
                 depth=None,
                 scale_min=1,
                 scale_max=None,
                 val_num = 100,
                 augment=True,
                 sample_q=None,
                 repeat=1,
                 train=True,
                 read_folder_mode='SINGLE',
                 discard_folder=None,
                 pre_normalize=True,
                 pre_scale=True,
                 loading_prompt=False,
                 prompt_valid_idx = None,
                 require_text_embedding=False,
                 ):

        self.repeat = repeat

        self.datamin, self.datamax = P_LOW, P_HIGH
        self.inp_size= inp_size
        self.depth = depth
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q
        self.augment = augment
        self.train = train
        self.pre_normalize = pre_normalize
        self.pre_scale=pre_scale
        self.loading_prompt = loading_prompt
        self.require_text_embedding = require_text_embedding
        self.txt_emb = {}

        if read_folder_mode == 'SINGLE':
            folder_list = [rootdatapath]
        elif read_folder_mode == 'MULTI':
            # folder_list = os.listdir(rootdatapath)
            folder_list = find_valid_folders(rootdatapath, mode='train' if self.train else 'test')
        elif read_folder_mode == 'PICK':
            folder_list_all = find_valid_folders(rootdatapath, mode='train' if self.train else 'test')
            # folder_list_all = os.listdir(rootdatapath)
            folder_list = [np.random.choice(list(set(folder_list_all)), replace=False)]   # randomly pick one dir
        else:
            raise ValueError('read_folder_mode must be SINGLE or MULTI or PICK')


        # loading filenames
        file_list_lr=[]
        file_list_gt=[]
        file_prompt_lr = []
        file_prompt_gt = []
        label_dir = []
        if read_folder_mode == 'MULTI':
            for _folder in sorted(folder_list):
                assert os.path.exists(_folder)
                # get labels
                local_label = os.path.basename(_folder)
                embedding = _load_text_embedding(_folder, self.require_text_embedding)
                if embedding is not None:
                    self.txt_emb[local_label] = embedding
                if train:
                    if discard_folder is not None and local_label in discard_folder:
                        continue
                    local_path_1 = os.path.join(_folder,'train',sub_folder_description[0])
                    local_path_2 = os.path.join(_folder,'train',sub_folder_description[1])
                else :
                    local_path_1 = os.path.join(_folder,'test',sub_folder_description[0])
                    local_path_2 = os.path.join(_folder,'test',sub_folder_description[1])

                filenames = load_file_list(local_path_1,regx=self.file_regex,is_fullpath=True)
                filenames_gt = load_file_list(local_path_2,regx=self.file_regex,is_fullpath=True)
                assert len(filenames) == len(filenames_gt), 'The num of LR and GT (%s) is not the same (PATH:%s)' % (
                'train' if self.train else 'test',
                local_path_1)

                if not self.train:
                    filenames = filenames[:val_num]
                    filenames_gt = filenames_gt[:val_num]

                file_list_lr.extend(filenames)
                file_list_gt.extend(filenames_gt)
                label_dir.extend([local_label] * len(filenames))

                if self.loading_prompt:
                    # get prompts (default the first pair in prompt folder)
                    filename_prompt_lr = load_file_list(
                        os.path.join(_folder, 'prompt', sub_folder_description[0]), regx=self.file_regex,
                        is_fullpath=True)
                    filename_prompt_gt = load_file_list(
                        os.path.join(_folder, 'prompt', sub_folder_description[1]), regx=self.file_regex,
                        is_fullpath=True)
                    if self.train:
                        # Determine how many prompts and training samples
                        num_prompts = len(filename_prompt_lr)
                        num_samples = len(filenames)

                        if num_prompts >= num_samples:
                            # Enough prompts, just truncate or match
                            selected_prompt_lr = filename_prompt_lr[:num_samples]
                            selected_prompt_gt = filename_prompt_gt[:num_samples]
                        else:
                            # Not enough prompts, randomly expand
                            expanded_indices = random.choices(range(num_prompts), k=num_samples)  # sample with replacement
                            selected_prompt_lr = [filename_prompt_lr[i] for i in expanded_indices]
                            selected_prompt_gt = [filename_prompt_gt[i] for i in expanded_indices]

                        file_prompt_lr.extend(selected_prompt_lr)
                        file_prompt_gt.extend(selected_prompt_gt)
                    else:
                        if prompt_valid_idx is not None:
                            file_prompt_lr.extend([filename_prompt_lr[prompt_valid_idx]] * len(filenames))
                            file_prompt_gt.extend([filename_prompt_gt[prompt_valid_idx]] * len(filenames))
                        else:
                            file_prompt_lr.extend([filename_prompt_lr[0]] * len(filenames))
                            file_prompt_gt.extend([filename_prompt_gt[0]] * len(filenames))
                else:
                    file_prompt_lr = None
                    file_prompt_gt = None
        else:
            assert os.path.exists(os.path.join(rootdatapath))
            # get labels
            local_label = os.path.basename(rootdatapath)
            embedding = _load_text_embedding(rootdatapath, self.require_text_embedding)
            if embedding is not None:
                self.txt_emb[local_label] = embedding
            if train:
                local_path_1 = os.path.join(rootdatapath, 'train', sub_folder_description[0])
                local_path_2 = os.path.join(rootdatapath, 'train', sub_folder_description[1])
            else:
                local_path_1 = os.path.join(rootdatapath, 'test', sub_folder_description[0])
                local_path_2 = os.path.join(rootdatapath, 'test', sub_folder_description[1])

            filenames = load_file_list(local_path_1, regx=self.file_regex, is_fullpath=True)
            filenames_gt = load_file_list(local_path_2, regx=self.file_regex, is_fullpath=True)
            assert len(filenames) == len(filenames_gt), 'The num of LR and GT (%s) is not the same (PATH:%s)' % (
                'train' if self.train else 'test',
                local_path_1)

            if not self.train:
                filenames = filenames[:val_num]
                filenames_gt = filenames_gt[:val_num]

            file_list_lr.extend(filenames)
            file_list_gt.extend(filenames_gt)
            label_dir.extend([local_label] * len(filenames))

            if self.loading_prompt:
                # get prompts (default the first pair in prompt folder)
                filename_prompt_lr = load_file_list(
                    os.path.join(rootdatapath, 'prompt', sub_folder_description[0]), regx=self.file_regex,
                    is_fullpath=True)
                filename_prompt_gt = load_file_list(
                    os.path.join(rootdatapath, 'prompt', sub_folder_description[1]), regx=self.file_regex,
                    is_fullpath=True)
                if self.train:
                    # Determine how many prompts and training samples
                    num_prompts = len(filename_prompt_lr)
                    num_samples = len(filenames)

                    if num_prompts >= num_samples:
                        # Enough prompts, just truncate or match
                        selected_prompt_lr = filename_prompt_lr[:num_samples]
                        selected_prompt_gt = filename_prompt_gt[:num_samples]
                    else:
                        # Not enough prompts, randomly expand
                        expanded_indices = random.choices(range(num_prompts), k=num_samples)  # sample with replacement
                        selected_prompt_lr = [filename_prompt_lr[i] for i in expanded_indices]
                        selected_prompt_gt = [filename_prompt_gt[i] for i in expanded_indices]

                    file_prompt_lr.extend(selected_prompt_lr)
                    file_prompt_gt.extend(selected_prompt_gt)
                else:
                    if prompt_valid_idx is not None:
                        file_prompt_lr.extend([filename_prompt_lr[prompt_valid_idx]] * len(filenames))
                        file_prompt_gt.extend([filename_prompt_gt[prompt_valid_idx]] * len(filenames))
                    else:
                        file_prompt_lr.extend([filename_prompt_lr[0]] * len(filenames))
                        file_prompt_gt.extend([filename_prompt_gt[0]] * len(filenames))
            else:
                file_prompt_lr = None
                file_prompt_gt = None

        self.filenames=file_list_lr
        self.filenames_gt=file_list_gt
        self.label_dir=label_dir
        self.file_prompt_lr = file_prompt_lr
        self.file_prompt_gt = file_prompt_gt
        pass


    def __len__(self):
        return len(self.filenames) * self.repeat

    def load_3d_imgs(self,path):
        img = self._read_array(path)
        if self.pre_normalize:
            img = img
        else:
            img = normalize_percentile(img, self.datamin, self.datamax, clip=CLIP)
        img = torch.tensor(img).unsqueeze(0)    # c,h,w
        return img

    def _read_array(self, path):
        return np.asarray(tifffile.imread(path), np.float32)


    def padding_rescale_3d_imgs(self,img_lr,img_gt,ref_size,s_factor):
        # padding or crop the LR-GT pairs to ref_size
        if ref_size > img_gt.shape[-2]:
            padding_sz = ref_size - img_gt.shape[-2]
            img_lr = torch.nn.functional.pad(img_lr, (0, padding_sz, 0, padding_sz, 0, 0), 'reflect')
            img_gt = torch.nn.functional.pad(img_gt, (0, padding_sz, 0, padding_sz, 0, 0), 'reflect')
        else:
            y0 = random.randint(0, img_gt.shape[-2] - ref_size)
            x0 = random.randint(0, img_gt.shape[-1] - ref_size)

            img_lr = img_lr[:, :, y0:y0+ref_size, x0:x0+ref_size]
            img_gt = img_gt[:, :, y0:y0+ref_size, x0:x0+ref_size]

        # reSize LR
        d_lr = math.floor(img_gt.shape[-3] / s_factor + 1e-9)
        h_lr = math.floor(img_gt.shape[-2] / s_factor + 1e-9)
        w_lr = math.floor(img_gt.shape[-1] / s_factor + 1e-9)

        # downsample with "s" and resample to gt size
        img_down = resize_3d_fn(img_lr, (d_lr, h_lr, w_lr))
        img_down = resize_3d_fn(img_down, (img_gt.shape[-3], img_gt.shape[-2], img_gt.shape[-1]))

        return img_down, img_gt

    def gen_postSR_3D_pair(self,base_lr_size,base_lr_depth,img_lr,img_gt,s_factor, train_mode=True):

        if train_mode:
            w_lr = base_lr_size
            w_hr = round(w_lr * s_factor)
            d_lr = base_lr_depth
            d_hr = round(d_lr * s_factor)

            z0 = random.randint(0, img_lr.shape[-3] - d_hr)
            x0 = random.randint(0, img_lr.shape[-2] - w_hr)
            y0 = random.randint(0, img_lr.shape[-1] - w_hr)

            crop_lr1 = img_lr[:, z0:z0 + d_hr, x0:x0 + w_hr, y0:y0 + w_hr]
            crop_hr = img_gt[:, z0:z0 + d_hr, x0:x0 + w_hr, y0:y0 + w_hr]
            crop_lr = resize_3d_fn(crop_lr1, (d_lr, w_lr, w_lr))

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



    def _load_crop_pair(self, idx):
        x = self.load_3d_imgs(self.filenames[idx % len(self.filenames)])  # remainder for repeatedly training
        x_gt = self.load_3d_imgs(self.filenames_gt[idx % len(self.filenames)])

        if not self.pre_scale:
            x = resize_3d_fn(
                x,
                (x_gt.shape[-3],x_gt.shape[-2], x_gt.shape[-1])
            )
        s = random.uniform(self.scale_min, self.scale_max)
        # x: c_in, d, h ,w
        # x_gt: c_in, d, h ,w
        if self.inp_size is None:
            crop_lr, crop_hr = self.gen_postSR_3D_pair(
                base_lr_size=None, base_lr_depth=None, img_lr=x, img_gt=x_gt,
                s_factor=s, train_mode=False)
        else:
            # Output: LR with input_size and GT with s*input_size
            crop_lr, crop_hr = self.gen_postSR_3D_pair(base_lr_size=self.inp_size,
                                                       base_lr_depth=self.depth,
                                                       img_lr=x,img_gt=x_gt,
                                                       s_factor=s,
                                                       train_mode=True)
        return crop_lr, crop_hr, s


    def __getitem__(self, idx):

        crop_lr, crop_hr, s = self._load_crop_pair(idx)
        local_label = self.label_dir[idx % len(self.filenames)]



        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5
            zflip = random.random() < 0.5
            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                if zflip:
                    x = x.flip(-3)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples_3d(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        for _i in range(cell.shape[-1]):
            loca_dim = 1 + _i
            # print(loca_dim)
            cell[:, _i] *= 2 / crop_lr.shape[loca_dim]

        ## loading prompt ###
        if self.loading_prompt:
            prompt_gt = self.load_3d_imgs(self.file_prompt_gt[idx % len(self.filenames)])
            prompt_lr = self.load_3d_imgs(self.file_prompt_lr[idx % len(self.filenames)])

            if not self.pre_scale:
                prompt_lr = resize_3d_fn(prompt_lr, (prompt_gt.shape[-3],prompt_gt.shape[-2], prompt_gt.shape[-1]))


        else:
            prompt_gt = 'None'
            prompt_lr = 'None'



        outdict = {
            'inp': crop_lr,
            'gtc': crop_hr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'sample_shape': crop_hr.shape, # c,d,h,w  (real gt size)
            'sample_label': local_label,
            'prompt': [prompt_lr, prompt_gt],
            's_factor': s
        }
        if local_label in self.txt_emb:
            outdict['txt_emb'] = self.txt_emb[local_label]
        return outdict


class LIIF_3D_Restoration_NPY(LIIF_3D_Restoration):
    """3D dataset using pre-normalized NPY files and crop-first mmap IO."""

    file_regex = '.*.npy'

    def __init__(self, *args, **kwargs):
        if not kwargs.get('pre_normalize', True):
            raise ValueError('NPY datasets must use pre_normalize=True')
        super().__init__(*args, **kwargs)

    def _read_array(self, path):
        return np.load(path, mmap_mode='r')

    @staticmethod
    def _volume_view(array, path):
        """Accept [D,H,W] or singleton-wrapped volumes without copying."""
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            raise ValueError(
                'Expected NPY volume [D,H,W] (optionally singleton-wrapped), '
                f'got {array.shape}: {path}'
            )
        return array

    def _load_crop_pair(self, idx):
        # Validation needs the complete volume. The normal base path still
        # avoids TIFF decoding, while training below reads only the mmap crop.
        if self.inp_size is None or not self.pre_scale:
            return super()._load_crop_pair(idx)

        sample_index = idx % len(self.filenames)
        lr_path = self.filenames[sample_index]
        gt_path = self.filenames_gt[sample_index]
        img_lr = self._volume_view(np.load(lr_path, mmap_mode='r'), lr_path)
        img_gt = self._volume_view(np.load(gt_path, mmap_mode='r'), gt_path)
        if img_lr.shape != img_gt.shape:
            raise ValueError(
                f'pre_scale=True requires matching LR/GT shapes, got '
                f'{img_lr.shape} and {img_gt.shape}'
            )

        s = random.uniform(self.scale_min, self.scale_max)
        d_lr, w_lr = int(self.depth), int(self.inp_size)
        d_hr, w_hr = round(d_lr * s), round(w_lr * s)
        if d_hr > img_lr.shape[-3] or w_hr > min(img_lr.shape[-2:]):
            raise ValueError(
                f'Requested crop {(d_hr, w_hr, w_hr)} exceeds volume '
                f'{img_lr.shape}: {lr_path}'
            )
        z0 = random.randint(0, img_lr.shape[-3] - d_hr)
        y0 = random.randint(0, img_lr.shape[-2] - w_hr)
        x0 = random.randint(0, img_lr.shape[-1] - w_hr)
        region = np.s_[z0:z0 + d_hr, y0:y0 + w_hr, x0:x0 + w_hr]

        # Copy only selected pages out of the read-only memmap.
        crop_lr = torch.from_numpy(
            np.array(img_lr[region], dtype=np.float32, copy=True)
        ).unsqueeze(0)
        crop_hr = torch.from_numpy(
            np.array(img_gt[region], dtype=np.float32, copy=True)
        ).unsqueeze(0)
        if crop_lr.shape[-3:] != (d_lr, w_lr, w_lr):
            crop_lr = resize_3d_fn(crop_lr, (d_lr, w_lr, w_lr))
        return crop_lr, crop_hr, s

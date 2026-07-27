import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from .Triplane_Ops_norm import Projectio_model, Triplanes2Vol, TDABlock
from .query_feature import (
    query_2d, query_3d, resize_fn, resize_3d_fn, gen_postSR_2D_pair,
)
from .Embedder import Embedder
from .xrestormer_arch import XRestormer_light
from .utils.customized import CAB


class prompt_aligner(nn.Module):
    def __init__(self, in_ch=1, out_ch=16):
        super(prompt_aligner, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=out_ch),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def _forward_postSR(self, prompt=None, base_size=64, s_factor=None,
                        is_training=False):
        """Align an SR prompt using the sample's current scale factor."""
        batch_num = prompt[0].shape[0]
        p_lr_list, p_hr_list = [], []
        scale_values = s_factor.detach().cpu().numpy()
        for batch_index in range(batch_num):
            local_lr, local_hr = gen_postSR_2D_pair(
                base_lr_size=base_size,
                img_lr=prompt[0][batch_index],
                img_gt=prompt[1][batch_index],
                s_factor=scale_values[batch_index],
                train_mode=is_training,
            )
            local_hr = self.downsample(local_hr.unsqueeze(0))
            local_hr = F.interpolate(
                local_hr, size=local_lr.shape[-2:], mode='bicubic',
                align_corners=False,
            )
            p_lr_list.append(local_lr.unsqueeze(0))
            p_hr_list.append(local_hr)

        prompt_lr = self.downsample(torch.cat(p_lr_list, dim=0))
        prompt_hr = torch.cat(p_hr_list, dim=0)
        return self.output_layer(prompt_lr), self.output_layer(prompt_hr)

    def _forward_PreScale(self, prompt=None,
                             ):
        # forward for image pairs without shape variation: like pre-SR(interp in dataloader), denoise, deblur....
        p_lr,p_hr = prompt
        p_lr = self.output_layer(self.downsample(p_lr))
        p_hr = self.output_layer(self.downsample(p_hr))
        return (p_lr,p_hr)

    def forward(self, prompt, s_factor=None, base_size=None, post_SR=False,
                is_training=False):
        if post_SR:
            return self._forward_postSR(
                prompt=prompt,
                base_size=base_size,
                s_factor=s_factor,
                is_training=is_training,
            )
        return self._forward_PreScale(prompt=prompt)


class MultiModel_X_light(nn.Module):
    def __init__(self,
                 tsk=1,

                 # features num in TripPrj
                 num_feat=32,

                 # img shape: only valid in "conv prj"
                 n_slices=101,
                 lateral_size=48,

                 prj_mode=None,

                 # Task tokens
                 dim_token=0,

                 # Embedder type: 'FourierEncoding', None
                 embedder_type_2D=None,
                 embedder_type_3D=None,

                 # MLP settings
                 hidden_feat=256,
                 hidden_lens=3,

                 # Transformer settings
                 window_size=8,

                 **kwargs):

        super(MultiModel_X_light, self).__init__()
        self.img_range = 1
        self.mean = 0
        self.window_size = window_size
        self.task = tsk
        self.n_slices = n_slices

        # TODO add the modules under different model types
        self.PFE_type = 'CNN'
        self.encoder_type = 'xRestormer'
        self.decoder_type = 'LIIF'

        self.num_feat = num_feat
        self.output_dim = 1

        # LIIF ops settings
        self.embedder_2d = Embedder.get_by_name(embedder_type_2D,
                                                **{'multires': 8, 'multiresZ': None, 'input_dim': 2, 'log_sample': True}
                                                )
        self.embedder_3d = Embedder.get_by_name(embedder_type_3D,
                                                **{'multires': 8, 'multiresZ': None, 'input_dim': 3, 'log_sample': True}
                                                )

        self.Linear_2d_input = nn.Sequential(
            nn.Linear(num_feat * 9 + 2 + 2 + self.embedder_2d.out_dim, hidden_feat),
            nn.ReLU()
        )
        self.Linear_3d_input = nn.Sequential(
            nn.Linear(num_feat * 3 + 3 + 3 + self.embedder_3d.out_dim, hidden_feat),
            nn.ReLU()
        )
        self.imnet = MLP_LIIF(in_dim=hidden_feat, out_dim=self.output_dim, hidden_list=hidden_lens * [hidden_feat, ])

        # task prompt definition
        self.dim_token = dim_token
        self.task_ids = torch.tensor([0, 1], dtype=torch.long)
        self.task_token = nn.Embedding(2, num_feat)  # 0: 2D, 1: 3D
        self.task_gamma = nn.Embedding(2, hidden_feat)
        self.task_beta = nn.Embedding(2, hidden_feat)

        # 3D projection module
        self.Tri_prj_models = Projectio_model(
            prj_mode=prj_mode,
            num_feat=num_feat,
            image_shape=(n_slices, lateral_size, lateral_size)
        )

        #promt aligner
        aligne_feature_num = 16
        self.prompt_aligner = prompt_aligner(in_ch=1, out_ch=aligne_feature_num)

        self.body = XRestormer_light(
            inp_channels=num_feat,
            out_channels=num_feat,
            window_size=self.window_size,
            dim=44,
            num_blocks=[2, 4, 4, 4],
            num_refinement_blocks=4,
            channel_heads=[1, 2, 4, 4],  # default 1 2 4 8; for lighter model, [0],[1],[3] were actived
            spatial_heads=[1, 2, 4, 4],
            overlap_ratio=[0.5, 0.5, 0.5, 0.5],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',  ## Other option 'BiasFree'
            dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
            scale=1,
            align_feature=aligne_feature_num
        )

        # 3D-aware blok
        self.TDABlock = TDABlock(num_feat)

        # post-conv in vol-feature
        self.AttenBlock_1 = CAB(num_feat * 3, compress_ratio=3, squeeze_factor=1)
        self.AttenBlock_2 = CAB(num_feat * 3, compress_ratio=3, squeeze_factor=1)
        self.apply(self._init_weights)

    def freeze_layers(self):
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False

    def _init_weights(self, m):
        # linear and layer norm initialization
        # print('Model parameters are initialized. (MLP and LayerNorm)')
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        if len(x.shape) == 4:
            h = x.shape[-2]
            w = x.shape[-1]
            mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
            mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if len(x.shape) == 5:
            d = x.shape[-3]
            h = x.shape[-2]
            w = x.shape[-1]
            mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
            mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
            mod_pad_d = (self.window_size - d % self.window_size) % self.window_size
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d),
                      'reflect')  # padding: left right -- top bottom -- front back (reverse padding)
        return x

    def compute_3d_max_projections(self, tensor_5d):
        """
        输入:
            tensor_5d: torch.Tensor, shape (B, C, D, H, W)
        输出:
            List of 3 projections:
                - proj_xy_all: shape (B, C, H, W)
                - proj_yz_all: shape (B, C, D, H)
                - proj_xz_all: shape (B, C, D, W)
        """
        # (B, C, D, H, W)
        proj_xy_all = torch.amax(tensor_5d, dim=2).squeeze(2)  # Max over D → (B, C, H, W)
        proj_yz_all = torch.amax(tensor_5d, dim=4).squeeze(2)   # Max over W → (B, C, D, H)
        proj_xz_all = torch.amax(tensor_5d, dim=3).squeeze(2)   # Max over H → (B, C, D, W)

        return [proj_xy_all, proj_yz_all, proj_xz_all]

    def gen_features(self, x,
                     prompt_tensor=None,
                     prompt_enable=False,
                     post_SR=False,
                     s_factor = None,
                     promptPD = None
                     ):


        # formating
        x = self.check_image_size(x)
        self.raw_dim = x.dim()

        prj_dims_num = 3  # default prj times along 3-axis
        paddings_front = paddings_back = 0  # default paddingZ
        if self.raw_dim == 4:
            x = x.unsqueeze_(2)  # b c h w --> b c 1 h w
            prj_dims_num = 1
        _, _, loca_d, loca_h, loca_w = x.shape  # projection times: (d) h w

        # projection
        x = self.Tri_prj_models(x, prj_num=prj_dims_num)

        # processing prompt for 2D/3D
        if prompt_enable:
            if self.raw_dim == 4:
                # resample Prompt
                # For PostSR: resample the input Prompt with diverse s_factor
                # For PreSR or other tasks without size mismatching:
                prompt_list = self.prompt_aligner(
                    prompt=prompt_tensor,
                    s_factor=s_factor,
                    base_size=loca_h,
                    post_SR=post_SR,
                    is_training=self.training,
                )
                prompt_list = [prompt_list]

            else:
                prompt_input = self.compute_3d_max_projections(prompt_tensor[0])
                prompt_target = self.compute_3d_max_projections(prompt_tensor[1])
                combined = list(zip(prompt_input, prompt_target))
                prompt_list = []  # save middle results
                for single_prompt in combined:
                    prompt_out = self.prompt_aligner(
                        prompt=single_prompt,
                        s_factor=s_factor,
                        base_size=loca_h,
                        post_SR=post_SR,
                        is_training=self.training,
                    )
                    prompt_list.append(prompt_out)
        else:
            prompt_list = [prompt_tensor]


        # generate task embedding dynamically
        if self.dim_token == 1 or self.dim_token == 3:
            task_id = self.task_ids[int(self.raw_dim == 5)].to(x[0].device)
            task_embed = self.task_token(task_id).unsqueeze(0)  # [1, C]
            task_embed = task_embed.unsqueeze(-1).unsqueeze(-1)  # [1, C, 1, 1]
        else:
            task_embed = 0

        # Tri-planes projection
        prj_planes = []

        if prompt_list is not None and len(prompt_list) == 1 and len(x) > 1:
            prompt_list = prompt_list * len(x)
        for dim_idx, (_x,local_prompt) in enumerate(zip(x,prompt_list)):
            _x = _x + task_embed
            _x = self.body(_x, local_prompt,prompt_enable,promptPD)  # DFE by transformer    (c_prj-->embedding-->embedding-->num_feature)
            _x = _x.unsqueeze(dim_idx + 2)
            prj_planes.append(_x)  # b,c --> offset 2

        # extracted-feature aggregation
        if self.raw_dim == 4:
            x = prj_planes[0].squeeze(2)
        else:
            # generate feature volumes
            prj_planes = self.TDABlock(prj_planes)
            x = Triplanes2Vol(prj_planes, shape=[loca_d - paddings_front - paddings_back, loca_h, loca_w])
            # CAB block for refinement
            res = x
            x = self.AttenBlock_1(x)
            x = self.AttenBlock_2(x)
            x = x + res

        # output
        self.features = x



    def queries_values(self, coord, cell):

        # generate task embedding dynamically
        if self.dim_token == 2 or self.dim_token == 3:
            task_id1 = self.task_ids[int(self.raw_dim == 5)].to(coord.device)
            gamma = self.task_gamma(task_id1).unsqueeze(0)
            beta = self.task_beta(task_id1).unsqueeze(0)
        else:
            gamma = beta = None

        if self.raw_dim == 4:
            x = query_2d(self.features,
                         coord=coord,
                         cell=cell,
                         input_trans=self.Linear_2d_input,
                         token_weight=(gamma, beta),
                         MLP_net=self.imnet,
                         embedder_function=self.embedder_2d.embed
                         )
        else:
            x = query_3d(self.features,
                         coord=coord,
                         cell=cell,
                         input_trans=self.Linear_3d_input,
                         token_weight=(gamma, beta),
                         MLP_net=self.imnet,
                         embedder_function=self.embedder_3d.embed
                         )
        # x = x / self.img_range + self.mean
        return x

    def forward(self, x,
                coord=None,
                cell=None,
                promptPD=None,
                prompt_tensor=None,
                prompt_enable=False,
                post_SR=False,
                s_factor = 1
                ):

        self.gen_features(x=x,
                          promptPD=promptPD,
                          prompt_tensor=prompt_tensor,
                          prompt_enable=prompt_enable,
                          post_SR=post_SR,
                          s_factor=s_factor,
                          )
        return self.queries_values(coord, cell)

    def gen_prompt(self, x,
                     prompt_tensor=None,
                     prompt_enable=False,
                     post_SR=False,
                     s_factor = None,
                     ):


        # formating
        x = self.check_image_size(x)
        self.raw_dim = x.dim()

        prj_dims_num = 3  # default prj times along 3-axis
        paddings_front = paddings_back = 0  # default paddingZ
        if self.raw_dim == 4:
            x = x.unsqueeze_(2)  # b c h w --> b c 1 h w
            prj_dims_num = 1
        _, _, loca_d, loca_h, loca_w = x.shape  # projection times: (d) h w

        # projection
        x = self.Tri_prj_models(x, prj_num=prj_dims_num)

        # processing prompt for 2D/3D
        if prompt_enable:
            if self.raw_dim == 4:
                # resample Prompt
                # For PostSR: resample the input Prompt with diverse s_factor
                # For PreSR or other tasks without size mismatching:
                prompt_list = self.prompt_aligner(
                    prompt=prompt_tensor,
                    s_factor=s_factor,
                    base_size=loca_h,
                    post_SR=post_SR,
                    is_training=self.training,
                )
                prompt_list = [prompt_list]

            else:
                prompt_input = self.compute_3d_max_projections(prompt_tensor[0])
                prompt_target = self.compute_3d_max_projections(prompt_tensor[1])
                combined = list(zip(prompt_input, prompt_target))
                prompt_list = []  # save middle results
                for single_prompt in combined:
                    prompt_out = self.prompt_aligner(
                        prompt=single_prompt,
                        s_factor=s_factor,
                        base_size=loca_h,
                        post_SR=post_SR,
                        is_training=self.training,
                    )
                    prompt_list.append(prompt_out)


        else:
            prompt_list = [prompt_tensor]



        # generate task embedding dynamically
        if self.dim_token == 1 or self.dim_token == 3:
            task_id = self.task_ids[int(self.raw_dim == 5)].to(x[0].device)
            task_embed = self.task_token(task_id).unsqueeze(0)  # [1, C]
            task_embed = task_embed.unsqueeze(-1).unsqueeze(-1)  # [1, C, 1, 1]
        else:
            task_embed = 0

        # Tri-planes projection
        prj_planes = []
        if prompt_list is not None and len(prompt_list) == 1 and len(x) > 1:
            prompt_list = prompt_list * len(x)
        return x , prompt_list


class MLP_LIIF(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

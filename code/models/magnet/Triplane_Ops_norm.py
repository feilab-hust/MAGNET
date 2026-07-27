import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# logger.info("Start print log")
# logger.debug("Do something")
# logger.warning("Something maybe fail.")
# logger.info("Finish")

__all__=['Projectio_model',
         'Triplanes2Vol',
         'TDABlock'
         ]
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def Triplanes2Vol(feature_list: object, shape: object = None, mode: object = 'cat') -> object:
    '''
    input: feature_list: list of tensors (feature plane): (b,c,d,h,w)
    shape: The shape of composed volume.Form: (d,h,w)
    mode: The feature fusion mode.
           Default is 'cat': Features of different planes were concatenated ---> c_fuse=3C
           Else: 'add': Features of different planes were added together--> c_fuse = C
    output: (d,h,w,c_fuse)
    '''

    f_hw,f_dw,f_dh = feature_list
    d,h,w = shape
    # assert f_hw.shape[1]==f_dw.shape[1]
    # assert f_hw.shape[0]==f_dh.shape[1]
    # max_dim = max(shape)
    # f_hw = F.pad(f_hw,pad=(), mode='constant',value=0)

    if mode == 'cat':
        f_hw_rep = f_hw.repeat(1, 1, d, 1, 1)
        f_dw_rep = f_dw.repeat(1, 1, 1, h, 1)
        f_dh_rep = f_dh.repeat(1, 1, 1, 1, w)
        return torch.cat([f_hw_rep,f_dw_rep,f_dh_rep],dim=1)
    elif mode == 'add':
        return f_hw+f_dw+f_dh


class ReduceFN:
    @staticmethod
    def var(xs, dim=0):
        return xs.var(dim=dim)

    @staticmethod
    def avg(xs, dim=0):
        return xs.mean(dim=dim)

    @staticmethod
    def max(xs, dim=0):
        return torch.amax(xs, dim=dim)

    @staticmethod
    def conv(xs, dim=None):
        # grayscale stack --> depth as channel
        return xs.squeeze(1)

def Tensor_Transpose(input_tensor:torch.Tensor,dim:int):
    '''
    input: input_tensor: tensor of shape (b,d,h,w)
    dim: flag for transpose tensor
    output: order of dimensions
    '''
    if dim == 2:
        return torch.permute(input_tensor,[0,1,2,3])
    elif dim == 3:
        return torch.permute(input_tensor,[0,2,1,3])
    elif dim == 4:
        return torch.permute(input_tensor,[0,3,1,2])

class TriplaneConv(nn.Module):
    def __init__(self, channels, out_channels, kernel_size, padding, is_rollout=True) -> None:
        super().__init__()
        in_channels = channels * 3 if is_rollout else channels
        self.is_rollout = is_rollout
        self.conv_xy = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_xz = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_yz = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    def forward(self, featmaps):

        # feature maps:
        # tpl_xy: b,c,h,w
        # tpl_xz: b,c,d,w
        # tpl_yz: b,c,d,h


        tpl_xy, tpl_xz, tpl_yz = featmaps
        H, W = tpl_xy.shape[-2:]
        D = tpl_xz.shape[-1]

        if self.is_rollout:
            tpl_xy_h = torch.cat([
                                  tpl_xy,
                                  torch.mean(tpl_xz, dim=-2, keepdim=True).expand_as(tpl_xy),             # b c d w --> b c 1 w --> b c {h} w
                                  torch.mean(tpl_yz, dim=-2, keepdim=True).transpose(-1,-2).expand_as(tpl_xy)  # b c d h --> b c 1 h --> b c h 1 --> b c h {w}
                                         ], dim=1
                               )
            tpl_xz_h = torch.cat([tpl_xz,
                               torch.mean(tpl_xy, dim=-2, keepdim=True).expand_as(tpl_xz),        # b c h w --> b c 1 w --> b c {d} w
                               torch.mean(tpl_yz, dim=-1, keepdim=True).expand_as(tpl_xz)]        # b c d h --> b c d 1 --> b c d {w}
                                 ,
                                dim=1)

            tpl_yz_h = torch.cat([tpl_yz,
                               torch.mean(tpl_xy, dim=-1, keepdim=True).transpose(-1, -2).expand_as(tpl_yz),  # b c h w --> b c h 1 --> b c 1 h --> b c {d} h
                               torch.mean(tpl_xz, dim=-1, keepdim=True).expand_as(tpl_yz)], dim=1)     # b c d w --> b c d 1 --> b c d {w}
        else:
            tpl_xy_h = tpl_xy
            tpl_xz_h = tpl_xz
            tpl_yz_h = tpl_yz


        tpl_xy_h = self.conv_xy(tpl_xy_h)
        tpl_xz_h = self.conv_xz(tpl_xz_h)
        tpl_yz_h = self.conv_yz(tpl_yz_h)

        return (tpl_xy_h, tpl_xz_h, tpl_yz_h)
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class TriplaneNorm(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.norm_xy = GroupNorm32(32, channels)
        self.norm_xz = GroupNorm32(32, channels)
        self.norm_yz = GroupNorm32(32, channels)

    def forward(self, featmaps):

        tpl_xy, tpl_xz, tpl_yz = featmaps
        # H, W = tpl_xy.shape[-2:]
        # D = tpl_xz.shape[-1]

        tpl_xy_h = self.norm_xy(tpl_xy)  # [B, C, H, W]
        tpl_xz_h = self.norm_xz(tpl_xz)  # [B, C, H, D]
        tpl_yz_h = self.norm_yz(tpl_yz)  # [B, C, W, D]

        # assert tpl_xy_h.shape[-2] == H and tpl_xy_h.shape[-1] == W
        # assert tpl_xz_h.shape[-2] == H and tpl_xz_h.shape[-1] == D
        # assert tpl_yz_h.shape[-2] == W and tpl_yz_h.shape[-1] == D

        return (tpl_xy_h, tpl_xz_h, tpl_yz_h)


class TriplaneSiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.silu = SiLU()

    def forward(self, featmaps):
        # tpl: [B, C, H + D, W + D]
        tpl_xy, tpl_xz, tpl_yz = featmaps
        return (self.silu(tpl_xy), self.silu(tpl_xz), self.silu(tpl_yz))



class TDABlock(nn.Module):
    '''
    Adapted from SIN3DM https://sin3dm.github.io/
    '''
    def __init__(self, channels,kernel_sz = 3):
        super(TDABlock, self).__init__()
        self.Seq = nn.Sequential(
                    TriplaneNorm(channels),
                    TriplaneSiLU(),
                    TriplaneConv(channels,channels,kernel_size=kernel_sz,padding=1,is_rollout=True)
                    )

    def forward(self, x):
        x = [_x.squeeze(dim_idx+2) for dim_idx,_x in enumerate(x)]
        out = self.Seq(x)


        return [
                nn.functional.tanh(x[i]+out[i]).unsqueeze(i+2)
                for i in range(len(x))
                ]

class Res_block(nn.Module):
    def __init__(self, channels,kernel_sz = 3):
        super(Res_block, self).__init__()
        self.Seq = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=kernel_sz, stride=1, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, kernel_size=kernel_sz, stride=1, padding=1, bias=False)
                    )
    def forward(self, x):
        out = self.Seq(x)
        return out + x

class Res_blockNorm(nn.Module):
    def __init__(self, channels,kernel_sz = 3):
        super(Res_blockNorm, self).__init__()

        self.Seq = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=kernel_sz, stride=1, padding=1, bias=False),
                    SiLU(),
                    nn.GroupNorm(8, channels),

                    nn.Conv2d(channels, channels, kernel_size=kernel_sz, stride=1, padding=1, bias=False),
                    SiLU(),
                    nn.GroupNorm(8, channels),

                    )
    def forward(self, x):
        out = self.Seq(x)
        return out + x


class Tri_prj_conv(nn.Module):
    def __init__(self,
                 prj_mode,
                 out_feature=32,
                 depth=48,#64 ZTF
                 res_block_num=1
                ):
        super(Tri_prj_conv, self).__init__()
        self.prj_mode = prj_mode
        self.prj_conv_op = nn.Conv2d(depth, out_feature, kernel_size=1, stride=1, padding=0, bias=False)

        # prj_mode_len = len(prj_mode)-1 if 'conv' in prj_mode else len(prj_mode)

        # aboundunt ops for query (if prj_mode only contains 'conv', the op_list would not be used);
        # Another mode; delete the "return" in 'conv' branch -> pre-process the input tensor,and process all-type projection parallel
        prj_mode_len = len(prj_mode)
        self.conv_op1_list = nn.ModuleList(prj_mode_len*[torch.nn.Conv2d(out_feature, out_feature, 3, 1, 1, bias=False)])
        self.conv_op2_list = nn.ModuleList(prj_mode_len*[torch.nn.Conv2d(out_feature, out_feature, 3, 1, 1, bias=False)])

    def forward(self, x, branch_idx):
        if self.prj_mode[branch_idx] == 'conv':
            x = self.prj_conv_op(x)
            return x
        else:
            feature = x
            feature = self.conv_op1_list[branch_idx](feature)
            feature = F.relu(feature)
            feature = self.conv_op2_list[branch_idx](feature)
            return feature+x



class Tri_prj_conv_Norm(nn.Module):
    def __init__(self,
                 prj_mode,
                 out_feature=32,
                 depth=48,
                 res_block_num=1
                ):
        super(Tri_prj_conv_Norm, self).__init__()
        self.prj_mode = prj_mode
        self.prj_conv_op = nn.Conv2d(depth, out_feature, kernel_size=1, stride=1, padding=0, bias=False)

        # prj_mode_len = len(prj_mode)-1 if 'conv' in prj_mode else len(prj_mode)

        # aboundunt ops for query (if prj_mode only contains 'conv', the op_list would not be used);
        # Another mode; delete the "return" in 'conv' branch -> pre-process the input tensor,and process all-type projection parallel
        prj_mode_len = len(prj_mode)
        self.conv_op1_list = nn.ModuleList(prj_mode_len*[torch.nn.Conv2d(out_feature, out_feature, 3, 1, 1, bias=False)])
        self.act_1_list = nn.ModuleList(prj_mode_len*[SiLU()])
        self.norm_1_list = nn.ModuleList(prj_mode_len*[nn.GroupNorm(8, out_feature)])

        self.conv_op2_list = nn.ModuleList(prj_mode_len*[torch.nn.Conv2d(out_feature, out_feature, 3, 1, 1, bias=False)])
        self.act_2_list = nn.ModuleList(prj_mode_len*[SiLU()])
        self.norm_2_list = nn.ModuleList(prj_mode_len*[nn.GroupNorm(8, out_feature)])

    def forward(self, x, branch_idx):
        if self.prj_mode[branch_idx] == 'conv':
            x = self.prj_conv_op(x)
            return x
        else:
            feature = x
            feature = self.conv_op1_list[branch_idx](feature)
            feature = self.act_1_list[branch_idx](feature)
            feature = self.norm_1_list[branch_idx](feature)

            feature = self.conv_op2_list[branch_idx](feature)
            feature = self.act_2_list[branch_idx](feature)
            feature = self.norm_2_list[branch_idx](feature)

            return feature+x


class Projectio_model(nn.Module):
    def __init__(self,
                 image_shape= None,
                 prj_mode=None,
                 num_feat=32,
                 ):
        super(Projectio_model, self).__init__()

        # default paras
        if image_shape is None:
            image_shape = [64, 128, 128]
        if prj_mode is None:
            prj_mode = ['avg', 'max', 'var', 'conv']
        logger.info(
            '[Alert]:Projection mode: %s'%prj_mode)
        if 'conv' in prj_mode:
            logger.info('[Altert]:Conv--> Require pre-definded shape parameters -> The input of net was restricted by the shape paras')

        # define ops
        self.image_shape=image_shape
        # Conv--> require pre-definded shape parameters -> The input of net was restricted by the shape paras
        self.prj_mode = prj_mode

        self.conv_initial_3D = torch.nn.Conv3d(in_channels=1,out_channels=num_feat,kernel_size=3,stride=1,padding=1)
        self.conv_initial_2D = torch.nn.Conv2d(in_channels=1, out_channels=num_feat, kernel_size=3, stride=1, padding=1)

        conv3d_hw = torch.nn.Conv3d(in_channels=len(prj_mode), out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        conv3d_dw = torch.nn.Conv3d(in_channels=len(prj_mode), out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        conv3d_dh = torch.nn.Conv3d(in_channels=len(prj_mode), out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)

        Prj_hw_module= Tri_prj_conv_Norm(prj_mode,depth=self.image_shape[0],out_feature=num_feat)
        Prj_dw_module= Tri_prj_conv_Norm(prj_mode,depth=self.image_shape[1],out_feature=num_feat)
        Prj_dh_module= Tri_prj_conv_Norm(prj_mode,depth=self.image_shape[2],out_feature=num_feat)

        self.Prj_op_list = nn.ModuleList([Prj_hw_module, Prj_dw_module, Prj_dh_module])
        self.Fuse_op_list = nn.ModuleList([conv3d_hw, conv3d_dw, conv3d_dh])

    def forward(self, img_stack,prj_num=3):
        if prj_num==1:

            # one branch inference
            x = self.conv_initial_2D(img_stack.squeeze(2))
            op_idx = [i for i,_f in enumerate(self.prj_mode) if _f == 'avg'][0]
            x = self.Prj_op_list[0](x,op_idx)


            # copy parameters

            ###20250324#########################
            # 获取 Prj_op_list[0] 的参数字典
            if self.training:
                source_state_dict = self.Prj_op_list[0].state_dict()
                # 获取 Prj_op_list[1] 和 Prj_op_list[2] 的当前参数字典
                target_state_dict_1 = self.Prj_op_list[1].state_dict()
                target_state_dict_2 = self.Prj_op_list[2].state_dict()

                # 只加载形状匹配的参数
                matched_state_dict_1 = {k: v for k, v in source_state_dict.items() if
                                        k in target_state_dict_1 and v.shape == target_state_dict_1[k].shape}
                matched_state_dict_2 = {k: v for k, v in source_state_dict.items() if
                                        k in target_state_dict_2 and v.shape == target_state_dict_2[k].shape}
                # 加载匹配的参数
                self.Prj_op_list[1].load_state_dict(matched_state_dict_1, strict=False)
                self.Prj_op_list[2].load_state_dict(matched_state_dict_2, strict=False)
                ################################################################
            return [x]
        else:
            x  = self.conv_initial_3D(img_stack)
            tri_planes = []
            for i in range(prj_num):
                reduce_dim = i+2
                prj_f_list =[]
                for op_idx in range(len(self.prj_mode)):
                    # feature input or original image input
                    if self.prj_mode[op_idx] == 'conv':
                        input_tensor = img_stack
                        prj_func = ReduceFN.conv
                        reduce_prj = prj_func(input_tensor, dim=reduce_dim)
                        reduce_prj = Tensor_Transpose(reduce_prj, dim=reduce_dim)
                    else:
                        input_tensor = x
                        prj_func = eval('ReduceFN.%s'%self.prj_mode[op_idx])
                        reduce_prj = prj_func(input_tensor,dim=reduce_dim)
                    reduce_prj = self.Prj_op_list[i](reduce_prj,op_idx)
                    prj_f_list.append(reduce_prj)
                prj_f_list = torch.stack(prj_f_list,dim=1)  # b, 3 , c_f, dim1,dim2
                prj_f_list =self.Fuse_op_list[i](prj_f_list)  # weight the prj-features
                tri_planes.append(prj_f_list.squeeze(1))
            return tri_planes


# if __name__ == '__main__':
#     # torch.cuda.empty_cache()
#     # input_c = 32
#     img_stack = torch.randn((2,1,16,48,48)).to('cuda:0')
#     batch,in_c,depth,height,width = img_stack.shape
#     prj_list = ['avg', 'var', 'conv']
#     prj_model = Projectio_model(prj_mode=prj_list,
#                                 image_shape=(depth,height,width),
#                                 num_feat=32)
#     prj_model.to('cuda:0')
#     prj_features = prj_model(img_stack)
#     pass

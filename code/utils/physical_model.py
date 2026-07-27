import torch
import torch.nn as nn
import math
from models.magnet.utils.fft_conv import fft_conv
def LF_back_project(projections: torch.Tensor, Hts: torch.Tensor):
    """
    Perform back projection, by deconvolution.
    """
    N, H, W, C = projections.shape
    Nm, Dm, Hm, Wm = Hts.shape
    assert N == Nm, "#Hts and # projections is not matched!"
    projections_perm = projections.permute([3, 0, 1, 2])  # (C,N,H,W)
    Hts_perm = Hts.permute([1, 0, 2, 3])  # (D,N,H,W)
    realspace = fft_conv(projections_perm, Hts_perm, None, padding='same')  # (C,D,H,W)
    realspace = realspace.permute([1, 2, 3, 0]) / N
    return realspace
def forward_project(realspace:torch.Tensor, H_mtx:torch.Tensor, padding:str='same'):
    """
    Perform forward projection.
    realspace: object space volume voxel
    H_mtx: H matrix used for forward projection. (Related to PSF)
    padding: padding added to each dim and each side. See torch.nn.functional.pad().
    """
    D,H,W,C = realspace.shape
    Dm,Hm,Wm = H_mtx.shape
    assert D == Dm, "The Depth of realspace and H_matrix is not matched!"
    realspace_padded = realspace
    try:
        realspace_padded = torch.nn.functional.pad(realspace_padded, pad=padding, value=0)
        padding = 'valid'
    except: pass
    # out = torch.nn.functional.conv2d(realspace_padded, H_mtx[None,...], None, padding='valid')
    realspace_perm = realspace_padded.permute([-1,0,1,2])   # (C,D,H,W)
    H_perm = H_mtx[None,...]                                # (1,Dm,Hm,Wm)
    # out = torch.nn.functional.conv2d(realspace_perm, H_perm, None, padding=padding).permute([1,2,3,0])[0]   # (H,W,C)
    out = fft_conv(realspace_perm, H_perm, None, padding=padding).permute([1, 2, 3, 0])[0]  # (H,W,C)
    return out
def SPLF_forward(input_tensor,psf,prj_size,Nnum,device='cuda:0',eps = 1e-7):
    height,width,depth=prj_size
    projection = torch.zeros([1, 1, height, width], dtype=torch.float32, device=device)
    for view_idx in range(Nnum * Nnum):
        u_idx = view_idx // Nnum
        v_idx = view_idx - u_idx * Nnum
        # print('u_idx: %02d v_idx: %02d'%(u_idx,v_idx))
        local_psf = psf[..., u_idx, v_idx]
        # local_psf = TF.rotate(local_psf, angle=180)
        temp_space = torch.zeros_like(input_tensor)
        temp_space[..., u_idx:height:Nnum, v_idx:width:Nnum] = input_tensor[..., u_idx:height:Nnum,
                                                               v_idx:width:Nnum]
        # temp_out = torch.nn.functional.conv2d(temp_space, local_psf[None,...], None, padding='same')
        temp_out = fft_conv(temp_space, local_psf[:, None, ...], bias=None, stride=1, padding='same',
                            groups=depth)
        temp_out = torch.sum(temp_out, dim=1)
        projection = projection + temp_out
    projection = ((projection - torch.min(projection)) / (torch.max(projection) - torch.min(projection) + eps))
    return projection

if __name__=='__main__':
    import numpy as np
    import tifffile
    import mat73
    import os
    import torchvision.transforms.functional as TF

    is_transpose = True # transpose for conv-op
    psf_tif_folder = r'E:\Software\Foundation_model\SimuProjection\View3_simu_[m3.4-3.4]_zstep_0.17\psf_stack'
    psf_list = []
    for _f in os.listdir(psf_tif_folder):
        if _f.endswith('.tif'):
            v_path = os.path.join(psf_tif_folder, _f)
            v_data = np.asarray(tifffile.imread(v_path),np.float32)
            if is_transpose:
                v_data = v_data[:, ::-1, ::-1]
            psf_list.append(v_data)
    psf = np.stack(psf_list,axis=0)
    psf_tensor = torch.from_numpy(psf)  # LF op
    psf_Ht_tensor  = TF.rotate(psf_tensor, angle=180)  # LF-back op


    raw_view = tifffile.imread(r'E:\Data\All_TrainingData\FoundationData_Prompt\LF_3DRecon\F_VCD\Mito_View3_[LF160_SF160_WF480]\HR\Data001_idx0001.tif')
    raw_view = np.asarray(raw_view,np.float32)
    raw_view = raw_view[..., np.newaxis] # view,h,w,c

    assert raw_view.shape[0]==psf.shape[0]
    input_view = torch.from_numpy(raw_view)


    # cover to 3D
    O_guess = LF_back_project(projections=input_view, Hts=psf_Ht_tensor)
    tifffile.imwrite(r'E:\Software\Foundation_model\SimuProjection\View3_simu_[m3.4-3.4]_zstep_0.17\results\O_1.tif', np.squeeze(O_guess.cpu().data.numpy()))

    # TEST deconvolution
    iter_num = 30
    intial_vol = torch.ones(O_guess.shape, dtype=torch.float32)
    for ii in range(iter_num):
        erro_list = []
        for view_idx in range(psf_Ht_tensor.shape[0]):
            fpj = forward_project(intial_vol, psf_tensor[view_idx])
            erro_back = input_view[view_idx] / fpj
            erro_back[erro_back < 0] = 0
            erro_back[torch.isinf(erro_back)] = 0
            erro_back[torch.isnan(erro_back)] = 0
            erro_list.append(erro_back)

        erro_list = torch.stack(erro_list, dim=0)
        bpjerro = LF_back_project(erro_list, psf_Ht_tensor)
        intial_vol = bpjerro * intial_vol

    tifffile.imwrite(r'E:\Software\Foundation_model\SimuProjection\View3_simu_[m3.4-3.4]_zstep_0.17\results\deconv_%02d.tif'%iter_num, np.squeeze(intial_vol.cpu().data.numpy()))

    pass

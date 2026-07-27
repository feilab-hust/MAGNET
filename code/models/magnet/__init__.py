from .MultiTaskModel_X_Hybrid_light import MultiModel_X_light
from .MultiTaskModel_X_Hybrid_light_3c import MultiModel_X_light_3c
import torch

def batch_LIIF_inference(Refine_net,lr,vali_data,mini_batch = 128 ** 3,**kwargs):
    # Accept either a plain nn.Module or a DDP/DataParallel wrapper. Validation
    # may intentionally unwrap DDP when only rank 0 performs inference.
    inference_net = getattr(Refine_net, 'module', Refine_net)

    inference_net.gen_features(x=lr,
                               prompt_tensor=vali_data['prompt'],
                               prompt_enable=kwargs['prompt_enable'],
                               post_SR=kwargs.get('post_SR', False),
                               s_factor=vali_data['s_factor'],
                               )
    n = vali_data['coord'].shape[1]
    ql = 0
    preds = []

    while ql < n:
        qr = min(ql + mini_batch, n)
        pred = inference_net.queries_values(
            coord=vali_data['coord'][:, ql:qr, :],
            cell=vali_data['cell'][:, ql:qr, :],
        )
        preds.append(pred)
        ql = qr
    recon_out = torch.cat(preds, dim=1)
    return recon_out
def create_model(model_name, network_parameters=None, **kwargs):

    if model_name == 'MultiModel_X_light':
        local_paras ={
                    'PFE_type': network_parameters.PFE_type,
                    'encoder_type': network_parameters.encoder_type,
                    'decoder_type' : network_parameters.decoder_type,


                    'tsk':network_parameters.task_idx,

                    # tri-planes settings
                    'n_slices': 64, #64
                    'lateral_size': 128, #128
                    #'prj_mode': ['avg', 'max', 'var'],
                    'prj_mode': None,
                    # dim_token
                    'dim_token': network_parameters.dim_token,

                    # PE
                    'embedder_type_2D': network_parameters.embedder_type_2D,
                    'embedder_type_3D': network_parameters.embedder_type_3D,

                    # MLP settings
                    'hidden_feat' : network_parameters.hidden_feat,
                    'hidden_lens' : network_parameters.hidden_lens,
                     }
    elif model_name == 'MultiModel_X_light_3c':
        local_paras ={
                    'PFE_type': network_parameters.PFE_type,
                    'encoder_type': network_parameters.encoder_type,
                    'decoder_type' : network_parameters.decoder_type,


                    'tsk':network_parameters.task_idx,

                    # tri-planes settings
                    'n_slices': 64,
                    'lateral_size': 128,
                    #'prj_mode': ['avg', 'max', 'var'],
                    'prj_mode': None,
                    # dim_token
                    'dim_token': network_parameters.dim_token,

                    # PE
                    'embedder_type_2D': network_parameters.embedder_type_2D,
                    'embedder_type_3D': network_parameters.embedder_type_3D,

                    # MLP settings
                    'hidden_feat' : network_parameters.hidden_feat,
                    'hidden_lens' : network_parameters.hidden_lens,
                     }

    else:
        raise ('No network found')

    local_paras.update(kwargs)
    model = eval(model_name)(**local_paras)
    return model,local_paras

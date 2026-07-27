"""Single command entry point: ``python main.py --config <file>``."""
import os
import random

import numpy as np
import torch
from torch.multiprocessing import spawn

import Configs


def set_seed(seed):
    seed = 42 if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_ddp(handler, args):
    gpu_ids = [item.strip() for item in args.gpu_para_list.split(',') if item.strip()]
    if len(gpu_ids) < args.world_size:
        raise ValueError(f"world_size={args.world_size}, but only {len(gpu_ids)} GPUs were configured")
    spawn(handler, args=(args.world_size, args), nprocs=args.world_size, join=True)


def main(argv=None):
    args = Configs.parse_args(argv)
    action = args.action.upper()
    model_family = args.model_family.lower()
    if action == "SSL" and model_family != "magnet":
        hint = (
            " Use action=TST for model_family=unet2stage."
            if model_family == "unet2stage" else ""
        )
        raise ValueError(
            "action=SSL is restricted to model_family=magnet; "
            f"got model_family={args.model_family!r}.{hint}"
        )
    Configs.configure(args)

    # CUDA visibility must be configured before the first CUDA API call.
    # set_seed() calls torch.cuda.is_available()/manual_seed_all(), so placing
    # it earlier would lock every independently launched job onto the GPUs
    # that were visible before its own config was applied.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_para_list
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(args.port)
    set_seed(args.seed)
    print(
        f"[GPU] CUDA_VISIBLE_DEVICES={args.gpu_para_list} | "
        f"local gid={args.gid} | world_size={args.world_size}"
    )

    if action == "TRAIN_DDP":
        from train_DDP import TRAIN_DDP
        _run_ddp(TRAIN_DDP, args)
    elif action == "VST_DDP":
        from train_VST_DDP import VST_DDP
        _run_ddp(VST_DDP, args)
    elif action == "TRAIN_PDOR":
        from train_Prompt_Predictor import TRAIN_PROMPT_PREDICTOR
        _run_ddp(TRAIN_PROMPT_PREDICTOR, args)
    elif action == "EVAL":
        from eval_multi import Inference
        Inference(args)
    elif action == "SSL":
        from self_tto import TRAIN_TTO
        TRAIN_TTO(args)
    else:
        raise ValueError(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    main()

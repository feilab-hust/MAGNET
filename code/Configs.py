import configargparse
from pathlib import Path

from models.defaults import MAGNET_DEFAULTS


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_optional_int(value):
    if value is None or isinstance(value, int):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("", "none", "null"):
        return None
    return int(value)


def config_parser():
    # Windows commonly defaults to GBK, while project configs are UTF-8 and
    # may contain Chinese comments. Always decode config files explicitly.
    parser=configargparse.ArgumentParser(
        config_file_open_func=lambda filename: open(filename, 'r', encoding='utf-8')
    )
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    parser.add_argument('--action', type=str, required=True)

    parser.add_argument('--expname', type=str, default='default_expname',
                        help="Experiment name.")

    parser.add_argument('--datadir', type=str,help="Dataset root path.")

    parser.add_argument("--task_idx", type=int, default=-1,
                        help="")
    parser.add_argument("--task_list",type=str,default="",help="T")

    # hardware settings
    parser.add_argument('--cuda', action='store_true',
                        help="Use CUDA or not.")

    parser.add_argument('--gid', type=int, default=0,
                        help="GPU ID to be used when '--cuda' is set to True.")

    parser.add_argument(
        "--gpu_para_list",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU ids to use, e.g., '0,1,2,3'",
    )

    parser.add_argument(
        "--DDP_backend",
        type=str,
        default="gloo",
        help="The backend to use for distributed training.",
    )
    parser.add_argument("--port", type=int, default=29500, help="Port for distributed training")

    parser.add_argument(
        "--world_size",
        type=int,
        default=4,
        help="Number of GPUs to use (must match the number of GPUs in --gpus)",
    )
    parser.add_argument('--num_workers', type=int, default=0,help="num_workers in Pytorch Dataloader")
    # random
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility.")



    ## training data skips (deprecated)
    parser.add_argument('--sample_skips', type=int, default='1',help="The sample interval of training dataset, e.g. 0,2,4,6.....")


    # Multi-task (2D SR,denoise,iso,3D-Recon)
    parser.add_argument("--model_family", type=str, default='magnet',
                        choices=['magnet'],
                        help="Model implementation selected by the unified trainer.")
    parser.add_argument("--model_parameters", type=str, default='{}',
                        help="Python/JSON dictionary passed to the selected model constructor.")
    parser.add_argument(
        "--model_task", "--task_id", type=str, default="1",
        help="UNiFMIR task head(s), e.g. model_task=2,1 aligned with task_list.",
    )
    parser.add_argument("--model_loss", type=str, default='l1', choices=['l1', 'mse'])
    parser.add_argument("--find_unused_parameters", type=bool, default=True)
    parser.add_argument("--MT_model_name", type=str, default='MultiModel_X_light')
    parser.add_argument("--MT_data_config", type=str, default='',help="The config of MultiModel")

    # checkpoint loading and resume
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--resume', action='store_true', help='whether to resume training')

    parser.add_argument("--loading_UniF_Trans", action='store_true', help='whether to load pretrained transformer')
    parser.add_argument("--loading_MT_ckpt_path", type=str, default='', help='The path to load pretrained MultiTask model')


    ## traing hyper parameters
    parser.add_argument('--gclip', type=float, default=0, help='gradient clipping threshold (0 = no clipping)')
    parser.add_argument('--sample_interval', type=int, default=10, help='The interval of validation')
    parser.add_argument('--test_num', type=_parse_optional_int, default=None,
                        help='Maximum evaluated test samples per label. None = all, 0 disables metrics.')
    parser.add_argument('--save_num', type=_parse_optional_int, default=None,
                        help='Maximum saved prediction TIFFs per label. None = all, 0 disables saving.')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=50, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')


    ## loss function settings
    parser.add_argument("--refine_loss", type=str,
                        default={'mse_loss': 1.0,
                                 'mae_loss': 1.0,
                                },
                        help="The loss function of refinement network",nargs='*')

    parser.add_argument("--texture_loss", help="Whether to compute texture_loss in refined net", action='store_true')
    parser.add_argument(
        "--tto_bl_weight", type=float, default=10.0,
        help="TTO background Laplacian (BL) loss weight.",
    )
    parser.add_argument(
        "--tto_cg_weight", type=float, default=0.6,
        help="TTO coordinate-grid high-frequency (CG) loss weight.",
    )
    parser.add_argument(
        "--tto_eval_initial", type=_parse_bool, default=False,
        help="Evaluate the loaded checkpoint before any TTO epoch.",
    )

    #wsi
    parser.add_argument('--sr', type=float, default=1.0)
    parser.add_argument("--block_size", type=int, nargs=3, default=[1, 128, 128])
    parser.add_argument("--overlap", type=int, nargs=3, default=[0, 16, 16])
    parser.add_argument(
        "--prenorm", type=_parse_bool, default=False,
        help=("EVAL input is already normalized: skip whole-image and per-patch "
              "normalization."),
    )
    parser.add_argument(
        "--use_rate", type=_parse_bool, default=True,
        help=("EVAL background/signal compensation switch. When true, each "
              "prediction block is multiplied by the local SSM-derived rate."),
    )
    parser.add_argument("--inp_path", type=str, default='')
    parser.add_argument("--prompt_path", type=str, default='')
    parser.add_argument(
        "--use_prompt", type=_parse_bool, default=True,
        help="Enable image prompt pairs/PDOR during MAGNET EVAL only.",
    )
    parser.add_argument("--save_path", type=str, default='')

    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--psf_dir", type=str, default='')
    # Retained so older config files still parse. TTO always uses FFT with the
    # original-size PSF; this value no longer selects an implementation.
    parser.add_argument("--deconv", type=str, default='f')
    parser.add_argument("--savestr", type=str, default='')

    parser.add_argument("--pdor", type=str, default='')

    parser.add_argument("--iso_3d", type=bool, default=False)
    parser.add_argument("--VST", type=bool, default=False)
    return parser

def parse_args(argv=None):
    """Parse once in main.py; importing this module has no CLI side effects."""
    args = config_parser().parse_args(argv)
    if args.config:
        config_dir = Path(args.config).expanduser().resolve().parent
        for name in ('MT_data_config', 'loading_MT_ckpt_path', 'psf_dir'):
            value = getattr(args, name, None)
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute() and not path.exists():
                candidate = config_dir / path
                if candidate.exists():
                    setattr(args, name, str(candidate.resolve()))
    return args


def configure(parsed_args):
    """Expose legacy path constants for the non-training action modules."""
    global args, sample_num, sample_results_path, log_dir
    global RefineNet_ckpt_prefix, device, max_task_str_lens
    args = parsed_args
    for name, value in MAGNET_DEFAULTS.items():
        setattr(args, name, value)
    sample_num = 4
    project_root = Path(__file__).resolve().parent.parent
    sample_results_path = str(project_root / 'log' / args.expname)
    log_dir = str(
        Path(sample_results_path) / 'ckpt' / args.model_family.lower()
    )
    RefineNet_ckpt_prefix = 'MultiModel_CKPT'
    device = args.gid
    max_task_str_lens = 12


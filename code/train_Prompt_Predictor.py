import os
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path

import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import models.magnet as model
from utils.checkpoint import extract_model_state
from utils.utility import *
from models.magnet.utils.criterions import *
from utils.dataloader_MultiTask import get_multi_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import Configs
from train_DDP import PROJECT_ROOT, _latest_epoch_checkpoint
from utils.checkpoint import load_checkpoint

def TRAIN_PROMPT_PREDICTOR(local_rank, world_size, args=None):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(args.DDP_backend,
                            timeout=datetime.timedelta(seconds=1800),
                            rank=local_rank,
                            world_size=world_size)
    cudnn.benchmark = True

    print('Distribution initialization with %s    Port:%d' % (args.DDP_backend, args.port))
    print(f"Process {local_rank} is running with GPU {local_rank}  -- Name: {torch.cuda.get_device_name(local_rank)}")
    print('Exp name:%s' % args.expname)

    family = args.model_family.lower()
    if family not in ("magnet", "magnet_old"):
        raise ValueError(
            f"TRAIN_PDOR only supports MAGNET families, got model_family={args.model_family!r}"
        )

    # dir make. PDOR is an extra MAGNET component, so keep it in its own ckpt
    # folder instead of mixing with the frozen MAGNET weights.
    sample_results_path = str(PROJECT_ROOT / 'log' / args.expname)
    log_dir = str(Path(sample_results_path) / 'ckpt' / 'pdor')
    task_idx = args.task_idx
    if local_rank == 0:
        mkdir_or_exsist([sample_results_path, log_dir], verbose=True)
        save_settings(save_path=os.path.join(sample_results_path), args=args)

    # define model (network parameters and initialization (Layer norm and Linear))
    model_module = __import__(f"models.{family}", fromlist=["create_model"])
    Refine_net, _ = model_module.create_model(
        model_name=args.MT_model_name,
        network_parameters=args,
    )
    Refine_net.to(local_rank)

    _, refine_ckpt = load_checkpoint(
        args.loading_MT_ckpt_path,
        type("FrozenMagnet", (), {"model": Refine_net, "family": family})(),
        torch.device("cuda", local_rank),
        mode="pretrained",
    )
    if refine_ckpt is None:
        raise FileNotFoundError(
            f"TRAIN_PDOR requires a frozen MAGNET checkpoint: {args.loading_MT_ckpt_path}"
        )

    for param in Refine_net.parameters():
        param.requires_grad = False
    print('[✓] All Refine_net parameters frozen.')

    prompt_predictor = PromptPredictor(z_channels=16, task_dim=12, embed_dim=176).to(local_rank)
    # define optimizer
    optimizer_fused_train = torch.optim.AdamW(
        [
            *[paras for paras in prompt_predictor.parameters() if paras.requires_grad == True]
        ], lr=args.lr)
    # define scheduler
    MT_Scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fused_train, milestones=[args.n_steps],
                                                        gamma=args.gamma)

    # loading ckpt (pre-trained or trained ckpt)
    epoch_state = 0
    if args.resume:
        resume_path = _latest_epoch_checkpoint(Path(log_dir))
        if resume_path is not None:
            print(f"[PDOR] resume from {resume_path}")
            checkpoint = torch.load(
                resume_path,
                map_location=torch.device("cuda", local_rank),
                weights_only=False,
            )
            prompt_predictor.load_state_dict(
                extract_model_state(checkpoint), strict=True
            )
            opt_state = checkpoint.get("optimizer")
            sched_state = checkpoint.get("lr_scheduler")
            if opt_state:
                optimizer_fused_train.load_state_dict(opt_state)
            if sched_state:
                MT_Scheduler.load_state_dict(sched_state)
            epoch_state = int(checkpoint.get("epoch") or 0)
        else:
            print("[PDOR] resume requested, but no prompt predictor checkpoint found.")
    dist.barrier()

    prompt_predictor = DDP(prompt_predictor,
                           device_ids=[local_rank],
                           output_device=local_rank,
                           find_unused_parameters=False)
    refine_loss_functions = eval(args.refine_loss[0]) if isinstance(args.refine_loss, list) else eval(args.refine_loss)
    refine_loss_ratio_dict = refine_loss_functions
    refine_loss_func_dict = {k: eval(k) for k, v in refine_loss_functions.items()}

    logs_data_per_epoch = []
    all_iter_times = []
    for idx_epoch in range(epoch_state, args.n_epochs):

        local_task = choose_task(task_idx=task_idx,
                                 task_list=args.task_list,
                                 idx_epoch=idx_epoch)

        # data loading
        (train_loader, train_sampler, test_loader,
         data_norm, task_str, data_dim,
         prompt_enable, post_SR, _eval_patch_size) = get_multi_data(task_idx=local_task, data_config=args.MT_data_config,
                                                    num_work=args.num_workers, local_rank=local_rank,
                                                    world_size=world_size
                                                    )
        train_sampler.set_epoch(idx_epoch)

        # pre-normalization for LIIF
        f = data_norm['inp']
        inp_sub = torch.FloatTensor(f['sub']).view([1, -1, *data_dim * [1]]).to(local_rank)  # b,c,d,h,w or b,c,h,w
        inp_div = torch.FloatTensor(f['div']).view([1, -1, *data_dim * [1]]).to(local_rank)

        f1 = data_norm['gt']
        gt_sub = torch.FloatTensor(f1['sub']).view(1, 1, -1).to(local_rank)
        gt_div = torch.FloatTensor(f1['div']).view(1, 1, -1).to(local_rank)

        loss_recon_epoch = []
        losses = {}

        ############ Training Loop Start ############
        prompt_predictor.train(mode=True)  # Important (model train for bn and dropout)
        Refine_net.train(mode=True)
        for idx_iter, train_data in enumerate(train_loader):
            # torch.cuda.empty_cache()
            torch.cuda.synchronize(local_rank)
            step_time = time.perf_counter()
            train_data = {k: v.to(local_rank) if isinstance(v, torch.Tensor) else v for k, v in train_data.items()}

            # clear gradients
            optimizer_fused_train.zero_grad()

            # normalization (-1,1) in LIIF
            lr = (train_data['inp'] - inp_sub) / inp_div
            hr = (train_data['gt'] - gt_sub) / gt_div

            # assign device for prompt data
            if prompt_enable:
                train_data['prompt'][0] = (train_data['prompt'][0].to(local_rank) - inp_sub) / inp_div
                train_data['prompt'][1] = (train_data['prompt'][1].to(local_rank) - gt_sub) / gt_div

            # forward
            with torch.no_grad():
                aligned_lr,aligned_prompt = Refine_net.gen_prompt(
                    x =lr,
                    prompt_tensor=train_data['prompt'],
                    prompt_enable=prompt_enable,
                    post_SR=post_SR,
                    s_factor=train_data['s_factor'],
                )
                prompt_list_gt = []
                for dim_idx, (_x, local_prompt) in enumerate(zip(aligned_lr, aligned_prompt)):
                    gt_prompt_latent = Refine_net.body.get_prompt_latents_gt(_x, local_prompt)
                    prompt_list_gt.append(gt_prompt_latent)


            prompt_list = []
            for dim_idx, (_x, local_prompt) in enumerate(zip(aligned_lr, aligned_prompt)):
                z_sample_prompt = local_prompt[0]
                task_id_tensor = F.one_hot(torch.tensor(local_task).to(z_sample_prompt.device), num_classes=12).float()
                task_id_tensor = task_id_tensor.unsqueeze(0).expand(z_sample_prompt.shape[0], -1)
                z_target_prompt = prompt_predictor(z_sample_prompt, task_id_tensor)
                predict_prompt_latent = Refine_net.body.get_prompt_latents_predict(_x, z_sample_prompt, z_target_prompt)
                prompt_list.append(predict_prompt_latent)

            #2d
            if len(prompt_list)==1:
                pred = torch.stack(prompt_list, dim=0)  # [N, B, C, H, W]
                target = torch.stack(prompt_list_gt, dim=0)
                # computing loss
                loss = 0
                for key in refine_loss_functions:
                    temp_loss_1 = refine_loss_ratio_dict[key] * refine_loss_func_dict[key](pred=pred, target=target,device=local_rank)
                    loss = loss + temp_loss_1
                    losses['%s_%s' % (task_str, key)] = temp_loss_1.data.cpu().numpy()
            #3d
            else:
                loss = 0
                for i in range(3):#3投影，三个方向的prompt
                    for key in refine_loss_functions:
                        temp_loss_1 = refine_loss_ratio_dict[key] * refine_loss_func_dict[key](pred=prompt_list[i], target=prompt_list_gt[i],
                                                                                               device=local_rank)
                        loss = loss + temp_loss_1
                        losses['%s_%s' % (task_str, key)] = temp_loss_1.data.cpu().numpy()

            # accumulate loss
            # loss_tensor = torch.tensor([loss.item()], device=local_rank)
            loss_tensor = loss.detach()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_loss = loss_tensor / dist.get_world_size()

            loss.backward()

            if args.gclip > 0:
                torch.nn.utils.clip_grad_norm_(prompt_predictor.parameters(), args.gclip)

            optimizer_fused_train.step()
            torch.cuda.synchronize(local_rank)
            iter_time = torch.tensor(
                time.perf_counter() - step_time,
                dtype=torch.float64,
                device=local_rank,
            )
            dist.all_reduce(iter_time, op=dist.ReduceOp.MAX)
            iter_seconds = float(iter_time.cpu())

            loss_recon_epoch.append(global_loss.cpu().item())
            loss_str = [name + ':' + str(value) for name, value in losses.items() if 'loss' in name]
            if local_rank == 0:
                all_iter_times.append(iter_seconds)
                print(
                    f"\rLRate: {MT_Scheduler.get_last_lr()[0]:e} | "
                    f"Epoch: [{idx_epoch}/{args.n_epochs}] Iter: [{idx_iter + 1}/{len(train_loader)}] "
                    f"iter_time={iter_seconds:.4f}s | "
                    f"Global Loss: {global_loss:.4f} | {loss_str}",
                    end=''
                )
        ############ Training Loop End ############

        ############ Validation Loop Start ############
        if local_rank == 0:
            # save ckpt
            training_loss_log = np.mean(loss_recon_epoch)
            print('\n%s--Training Loss: %f' % (task_str, training_loss_log))
            RefineNet_ckpt_prefix = 'PromptPredictor_CKPT'
            # save ckpt per epoch
            torch.save({'epoch': idx_epoch + 1,
                        'state_dict': prompt_predictor.module.state_dict(),
                        'optimizer': optimizer_fused_train.state_dict(),
                        "lr_scheduler": MT_Scheduler.state_dict(),
                        },
                       os.path.join(log_dir, '%s.tar' % RefineNet_ckpt_prefix)
                       )

            if idx_epoch % args.sample_interval == 0:
                torch.save({'epoch': idx_epoch + 1,
                            'state_dict': prompt_predictor.module.state_dict(),
                            'optimizer': optimizer_fused_train.state_dict(),
                            "lr_scheduler": MT_Scheduler.state_dict(),
                            },
                           os.path.join(log_dir, '%s_epoch_%03d.tar' % (RefineNet_ckpt_prefix, idx_epoch))
                           )

                prompt_predictor.eval()
                Refine_net.eval()

                val_loss_epoch = []
                val_loop_desc = f"Validation for PromptPredictor at Epoch {idx_epoch}"


                with torch.no_grad():
                    for valid_idx_iter, val_data in enumerate(tqdm(test_loader, desc=val_loop_desc)):
                        for k, v in val_data.items():
                            if isinstance(v, torch.Tensor):
                                val_data[k] = v.to(local_rank)

                        # Normalization
                        lr = (val_data['inp'] - inp_sub) / inp_div
                        hr = val_data['gt']

                        if prompt_enable:
                            val_data['prompt'][0] = (val_data['prompt'][0].to(local_rank) - inp_sub) / inp_div
                            val_data['prompt'][1] = (val_data['prompt'][1].to(local_rank) - gt_sub) / gt_div

                        aligned_lr, aligned_prompt = Refine_net.gen_prompt(
                            x=lr,
                            prompt_tensor=val_data['prompt'],
                            prompt_enable=prompt_enable,
                            post_SR=post_SR,
                            s_factor=val_data['s_factor'],
                        )

                        prompt_list_gt, prompt_list_pred = [], []
                        for dim_idx, (_x, local_prompt) in enumerate(zip(aligned_lr, aligned_prompt)):
                            z_sample_prompt = local_prompt[0]
                            task_id_tensor = F.one_hot(torch.tensor(local_task).to(z_sample_prompt.device),
                                                       num_classes=12).float()
                            task_id_tensor = task_id_tensor.unsqueeze(0).expand(z_sample_prompt.shape[0], -1)

                            z_target_prompt = prompt_predictor(z_sample_prompt, task_id_tensor)
                            gt_prompt_latent = Refine_net.body.get_prompt_latents_gt(_x, local_prompt)
                            predict_prompt_latent = Refine_net.body.get_prompt_latents_predict(_x, z_sample_prompt,
                                                                                               z_target_prompt)

                            prompt_list_gt.append(gt_prompt_latent)
                            prompt_list_pred.append(predict_prompt_latent)
                        if len(prompt_list_pred) == 1:
                            pred = torch.stack(prompt_list_pred, dim=0)  # [N, B, C, H, W]
                            target = torch.stack(prompt_list_gt, dim=0)
                            # computing loss
                            val_loss = 0
                            for key in refine_loss_functions:
                                temp_loss_1 = refine_loss_ratio_dict[key] * refine_loss_func_dict[key](pred=pred,
                                                                                                       target=target,
                                                                                                       device=local_rank)
                                val_loss = val_loss + temp_loss_1
                                #losses['%s_%s' % (task_str, key)] = temp_loss_1.data.cpu().numpy()
                        # 3d
                        else:
                            val_loss = 0
                            for i in range(3):  # 3投影，三个方向的prompt
                                for key in refine_loss_functions:
                                    temp_loss_1 = refine_loss_ratio_dict[key] * refine_loss_func_dict[key](
                                        pred=prompt_list_pred[i], target=prompt_list_gt[i],
                                        device=local_rank)
                                    val_loss = val_loss + temp_loss_1
                                    #losses['%s_%s' % (task_str, key)] = temp_loss_1.data.cpu().numpy()


                        val_loss_epoch.append(val_loss.item())
                val_loss_mean = np.mean(val_loss_epoch)
                print(f"\n[✓] Validation Loss for PromptPredictor (Epoch {idx_epoch}): {val_loss_mean:.6f}")

                # Save logs
                with open(os.path.join(log_dir, 'PromptPredictor_val_logs.txt'), 'a') as f:
                    f.write(f"Epoch {idx_epoch:03d} - Prompt Val Loss: {val_loss_mean:.6f}\n")

        MT_Scheduler.step()

    if local_rank == 0:
        average = float(np.mean(all_iter_times)) if all_iter_times else float("nan")
        print(f"Average training iter time: {average:.4f}s")


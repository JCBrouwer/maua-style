import gc
import math
import os.path
import sys

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch import optim

import load
import loss
import models
from utils import info, wrapping_slice

PBAR = tqdm.tqdm(file=sys.stdout, smoothing=0.1)


def set_content_targets(net, content_image, opt):
    if opt.pbar:
        PBAR.set_description(f"Capturing content targets...")

    for i in net.content_losses:
        i.mode = "capture"

    net(content_image.type(opt.model.dtype))

    for i in net.content_losses:
        i.mode = "none"


def set_temporal_targets(net, warp_image, warp_weights=None, opt=None):
    if opt.pbar:
        PBAR.set_description(f"Capturing temporal targets...")

    for i in net.temporal_losses:
        i.mode = "capture"
        if warp_weights is not None:
            i.weights = warp_weights.type(opt.model.dtype)

    net(warp_image.type(opt.model.dtype))

    for i in net.temporal_losses:
        i.mode = "none"


def set_style_targets(net, style_images, opt):
    if opt.pbar:
        PBAR.set_description(f"Capturing style targets...")

    for j in net.style_losses:
        j.reset_targets()
        j.mode = "capture"

    for i, image in enumerate(style_images):

        for j in net.style_losses:
            j.blend_weight = opt.param.style_blend_weights[i]

        net(image.type(opt.model.dtype))

    for j in net.style_losses:
        j.mode = "none"


def set_style_video_targets(net, style_videos, opt):
    if opt.pbar:
        PBAR.set_description(f"Capturing style video targets...")

    for j in net.style_losses:
        j.reset_targets()
        j.mode = "capture"

    for i, video in enumerate(style_videos):

        for j in net.style_losses:
            # divide weight by number of windows videos average over
            # this also decreases strength of static image styles, though
            # TODO refactor? or just set style_video_factor accordingly?
            j.blend_weight = opt.param.style_blend_weights[i] / max(len(video) - opt.param.gram_frame_window + 1, 1)

        # average style targets over all windows of video
        for window_start in range(max(len(video) - opt.param.gram_frame_window + 1, 1)):
            net(video[window_start : window_start + opt.param.gram_frame_window].type(opt.model.dtype))

    for j in net.style_losses:
        j.mode = "none"


def optimize(content, styles, init, num_iters, opt, net=None, losses=None):
    if init.numel() <= 5 * 1664 * 1664 * 3:
        opt.model.gpu = 0
        opt.model.multidevice = False
    elif init.numel() <= 5 * 3000 * 3000 * 3:
        opt.model.gpu = "0,1"
        opt.model.multidevice = True
        opt.param.tv_weight = 0
        opt.model.model_file = "../style-transfer/models/vgg16-00b39a1b.pth"
        opt.optim.optimizer = "adam"
    else:
        opt.model.model_file = "../style-transfer/models/nin_imagenet.pth"
        opt.model.style_layers = "relu1,relu3,relu5,relu7,relu9,relu11"
        opt.model.content_layers = "relu8"
    opt.param.num_iterations = num_iters
    # opt.optim.print_iter = num_iters // 4

    # TODO make work for spatial tiling as well
    if "_vid" in opt.transfer_type:
        # linearly space start of windows over total time of styles given number of windows needed to cover pastiche
        num_windows = math.ceil(init.shape[0] / opt.param.gram_frame_window)
        framestep = (
            np.array([style.shape[0] - opt.param.gram_frame_window / 2 for style in [init] + styles]) / num_windows
        )
        windows = [
            [math.ceil(framestep[idx] * n) for n in range(num_windows + 1)]
            if ([init] + styles)[idx].shape[0] != 1
            else [0] * (num_windows + 1)
            for idx in range(len([init] + styles))
        ]
    else:
        windows = [[0]] * len(styles)
    # print(windows)

    if net is None or losses is None:
        net, losses = models.load_model(opt.model, opt.param)

    if opt.pbar:
        PBAR.reset()
        PBAR.total = len(windows[0]) * num_iters
        PBAR.refresh()
    # else:
    #     line = ""
    #     for mod in losses:
    #         line += f"{mod.name:<10s}"
    #     print(line)

    set_content_targets(net, content, opt)

    if "_vid" in opt.transfer_type:
        if opt.param.avg_frame_window == -1:
            set_style_video_targets(net, styles, opt)
            for mod in losses:
                mod.mode = "loss"
    else:
        set_style_targets(net, styles, opt)
        for mod in losses:
            mod.mode = "loss"

    output = init
    for w, window_start in enumerate(windows[0]):

        if "_vid" in opt.transfer_type:
            front_overlap = windows[0][w - 1] + opt.param.gram_frame_window - window_start
            if window_start + opt.param.gram_frame_window >= output.shape[0]:
                end_overlap = (window_start + opt.param.gram_frame_window) % output.shape[0]
            else:
                end_overlap = 0

            current_pastiche = wrapping_slice(output, window_start, opt.param.gram_frame_window)

            if not opt.param.avg_frame_window == -1:
                current_styles = [
                    wrapping_slice(style, windows[num + 1][w], opt.param.avg_frame_window)
                    for num, style in enumerate(styles)
                ]
                if "_vid" in opt.transfer_type:
                    set_style_video_targets(net, current_styles, opt)

                for mod in losses:
                    mod.mode = "loss"
        else:
            current_pastiche = init

        pastiche = nn.Parameter(current_pastiche.type(opt.model.dtype))

        # Maybe normalize target strengths; divide weights by channel size (only once, strengths aren't reset!)
        if w == 0 and opt.param.normalize_weights:
            for n, i in enumerate(net.content_losses + net.style_losses + net.temporal_losses):
                i.strength = i.strength / max(i.target.size())

        if opt.optim.optimizer == "lbfgs":
            if opt.pbar:
                PBAR.set_description(f"Running optimization with L-BFGS")
            optim_state = {
                "max_iter": num_iters,
                "tolerance_change": float(opt.optim.lbfgs_tolerance_change),
                "tolerance_grad": float(opt.optim.lbfgs_tolerance_grad),
            }
            if opt.optim.lbfgs_num_correction != 100:
                optim_state["history_size"] = opt.optim.lbfgs_num_correction
            optimizer = optim.LBFGS([pastiche], **optim_state)
            iters = 1
        elif opt.optim.optimizer == "adam":
            if opt.pbar:
                PBAR.set_description(f"Running optimization with ADAM")
            optimizer = optim.Adam([pastiche], lr=opt.optim.learning_rate)
            iters = num_iters

        i = [0]
        log_losses = [[0] * len(losses)]

        def feval():
            # info(pastiche)
            optimizer.zero_grad()

            net(pastiche)

            total_loss = 0
            for idx, mod in enumerate(losses):
                if mod.loss == 0:
                    continue
                log_losses[0][idx] += mod.loss.detach().cpu().item()
                total_loss += mod.loss.to(opt.model.backward_device)

            total_loss.backward()

            # disable gradients to frames already styled in previous windows
            if w != 0:
                pastiche.grad[:front_overlap] = 0
                if end_overlap > 0:
                    pastiche.grad[-end_overlap:] = 0

            for mod in losses:
                mod.loss = 0

            if opt.pbar:
                PBAR.update(1)
            i[0] += 1

            if opt.optim.print_iter > 0 and i[0] % opt.optim.print_iter == 0 and opt.model.verbose:
                print(f"Iteration {i[0]} / {opt.param.num_iterations}, Loss: {total_loss}")
            if opt.optim.save_iter > 0 and (i[0] % opt.optim.save_iter == 0 or i[0] == num_iters):
                load.save_tensor_to_file(
                    pastiche.detach().cpu(),
                    opt,
                    (w * num_iters) + i[0] if not (w * num_iters) + i[0] == len(windows[0]) * num_iters else None,
                    pastiche.size(3),
                )

            # if not opt.pbar:
            #     if i[0] % (num_iters // 8) == 0:
            #         line = ""
            #         for ll in log_losses[0]:
            #             line += f"{f'{ll / (num_iters // 8):.2E}':<10s}"
            #         print(line)
            #         log_losses[0] = [0] * len(losses)

            return total_loss

        while i[0] <= iters:
            optimizer.step(feval)

        # TODO figure out better way to blend frames into output
        if "_vid" in opt.transfer_type:
            output[
                wrapping_slice(output, window_start, opt.param.gram_frame_window, return_indices=True)
            ] = pastiche.cpu()
            if not opt.param.avg_frame_window == -1:
                del current_styles
        else:
            output = pastiche.cpu()

        del optimizer, current_pastiche, pastiche
        gc.collect()
        th.cuda.empty_cache()

    return output

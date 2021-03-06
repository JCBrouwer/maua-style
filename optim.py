import gc
import json
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


def set_content_targets(net, content_image, args):
    if not args.verbose:
        PBAR.set_description(f"Capturing content targets...")

    for i in net.content_losses:
        i.mode = "capture"

    net(content_image.type(args.dtype))

    for i in net.content_losses:
        i.mode = "none"


def set_temporal_targets(net, warp_image, warp_weights=None, args=None):
    if not args.verbose:
        PBAR.set_description(f"Capturing temporal targets...")

    for i in net.temporal_losses:
        i.mode = "capture"
        if warp_weights is not None:
            i.weights = warp_weights.type(args.dtype)

    net(warp_image.type(args.dtype))

    for i in net.temporal_losses:
        i.mode = "none"


def set_style_targets(net, style_images, args):
    if not args.verbose:
        PBAR.set_description(f"Capturing style targets...")

    for j in net.style_losses:
        j.reset_targets()
        j.mode = "capture"

    for i, image in enumerate(style_images):

        for j in net.style_losses:
            j.blend_weight = args.style_blend_weights[i]

        net(image.type(args.dtype))

    for j in net.style_losses:
        j.mode = "none"


def set_style_video_targets(net, style_videos, args):
    if not args.verbose:
        PBAR.set_description(f"Capturing style video targets...")

    for j in net.style_losses:
        j.reset_targets()
        j.mode = "capture"

    for i, video in enumerate(style_videos):

        for j in net.style_losses:
            # divide weight by number of windows videos average over
            # this also decreases strength of static image styles, though
            # TODO refactor? or just set style_video_factor accordingly?
            j.blend_weight = args.style_blend_weights[i] / max(len(video) - args.gram_frame_window + 1, 1)

        # average style targets over all windows of video
        for window_start in range(max(len(video) - args.gram_frame_window + 1, 1)):
            net(video[window_start : window_start + args.gram_frame_window].type(args.dtype))

    for j in net.style_losses:
        j.mode = "none"


def set_model_args(args, current_size):
    with open(args.scaling_args, "r") as f:
        scaling = json.load(f)

    found = False
    for size, params in scaling.items():
        if int(size) < current_size:
            continue  # skip options for sizes smaller than the current size
        if len(args.gpu.split(",")) < len(params["gpu"].split(",")):
            continue  # skip options which require more gpus than available
        found = True
        break
    if not found:
        print("Warning: no model configuration found for this size, out of memory error is likely...")
    for key, param in params.items():
        args.__dict__[key] = param


def optimize(content, styles, init, num_iters, args, net=None, losses=None):

    # TODO make work for spatial tiling as well
    if "_vid" in args.transfer_type:
        # linearly space start of windows over total time of styles given number of windows needed to cover pastiche
        num_windows = math.ceil(init.shape[0] / args.gram_frame_window)
        framestep = np.array([style.shape[0] - args.gram_frame_window / 2 for style in [init] + styles]) / num_windows
        windows = [
            [math.ceil(framestep[idx] * n) for n in range(num_windows + 1)]
            if ([init] + styles)[idx].shape[0] != 1
            else [0] * (num_windows + 1)
            for idx in range(len([init] + styles))
        ]
    else:
        windows = [[0]] * len(styles)

    if net is None or losses is None:
        set_model_args(args, max(*init.shape))
        net, losses = models.load_model(args)

    if not args.verbose:
        PBAR.reset()
        PBAR.total = len(windows[0]) * num_iters
        PBAR.refresh()

    set_content_targets(net, content, args)

    if "_vid" in args.transfer_type:
        if args.avg_frame_window == -1:
            set_style_video_targets(net, styles, args)
            for mod in losses:
                mod.mode = "loss"
    else:
        set_style_targets(net, styles, args)
        for mod in losses:
            mod.mode = "loss"

    output = init.clone()
    for w, window_start in enumerate(windows[0]):

        if "_vid" in args.transfer_type:
            front_overlap = windows[0][w - 1] + args.gram_frame_window - window_start
            if window_start + args.gram_frame_window >= output.shape[0]:
                end_overlap = (window_start + args.gram_frame_window) % output.shape[0]
            else:
                end_overlap = 0

            current_pastiche = wrapping_slice(output, window_start, args.gram_frame_window)

            if not args.avg_frame_window == -1:
                current_styles = [
                    wrapping_slice(style, windows[num + 1][w], args.avg_frame_window)
                    for num, style in enumerate(styles)
                ]
                if "_vid" in args.transfer_type:
                    set_style_video_targets(net, current_styles, args)

                for mod in losses:
                    mod.mode = "loss"
        else:
            current_pastiche = init

        pastiche = nn.Parameter(current_pastiche.type(args.dtype))

        # Maybe normalize target strengths; divide weights by channel size (only once, strengths aren't reset!)
        if w == 0 and args.normalize_weights:
            for n, i in enumerate(net.content_losses + net.style_losses + net.temporal_losses):
                i.strength = i.strength / max(i.target.size())

        if args.optimizer == "lbfgs":
            if not args.verbose:
                PBAR.set_description(f"Running optimization with L-BFGS")
            optim_state = {
                "max_iter": num_iters,
                "tolerance_change": float(args.lbfgs_tolerance_change),
                "tolerance_grad": float(args.lbfgs_tolerance_grad),
            }
            if args.lbfgs_num_correction != 100:
                optim_state["history_size"] = args.lbfgs_num_correction
            optimizer = optim.LBFGS([pastiche], **optim_state)
            iters = 1
        elif args.optimizer == "adam":
            if not args.verbose:
                PBAR.set_description(f"Running optimization with ADAM")
            optimizer = optim.Adam([pastiche], lr=args.learning_rate)
            iters = num_iters

        i = [0]
        log_losses = [[0] * len(losses)]

        def feval():
            optimizer.zero_grad()

            net(pastiche)

            total_loss = 0
            for idx, mod in enumerate(losses):
                if mod.loss == 0:
                    continue
                log_losses[0][idx] += mod.loss.detach().cpu().item()
                total_loss += mod.loss.to(args.backward_device)

            total_loss.backward()

            # disable gradients to frames already styled in previous windows
            if w != 0:
                pastiche.grad[:front_overlap] = 0
                if end_overlap > 0:
                    pastiche.grad[-end_overlap:] = 0

            for mod in losses:
                mod.loss = 0

            if not args.verbose and not (args.optimizer == "adam" and i[0] == 0):
                PBAR.update(1)
            i[0] += 1

            if args.print_iter > 0 and i[0] % args.print_iter == 0 and args.verbose:
                print(f"Iteration {i[0]} / {args.num_iters}, Loss: {total_loss}")
            if args.save_iter > 0 and (i[0] % args.save_iter == 0 or i[0] == num_iters):
                load.save_tensor_to_file(
                    pastiche.detach().cpu(),
                    args,
                    (w * num_iters) + i[0] if not (w * num_iters) + i[0] == len(windows[0]) * num_iters else None,
                    pastiche.size(3),
                )

            return total_loss

        while i[0] <= iters:
            optimizer.step(feval)

        # TODO figure out better way to blend frames into output
        if "_vid" in args.transfer_type:
            output[wrapping_slice(output, window_start, args.gram_frame_window, return_indices=True)] = pastiche.cpu()
            if not args.avg_frame_window == -1:
                del current_styles
        else:
            output = pastiche.cpu()

        del optimizer, current_pastiche, pastiche
        gc.collect()
        th.cuda.empty_cache()

    return output

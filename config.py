import argparse
import json
import uuid

import torch

from utils import name


def get_args():
    parser = argparse.ArgumentParser()

    # input options
    parser.add_argument("-transfer_type", default="img_img", choices=["img_img", "vid_img", "img_vid"])
    parser.add_argument("-output_dir", default="./output")
    parser.add_argument("-content", help="Content target image")
    parser.add_argument("-style", help="Style target image", nargs="*")
    parser.add_argument("-init", type=str, default="random")
    parser.add_argument("-seed", type=int, default=-1)

    # main parameters
    parser.add_argument("-image_sizes", default="256,512,724,1024,1448")
    parser.add_argument("-num_iters", default="500,400,300,200,100")
    parser.add_argument("-content_weight", type=float, default=5)
    parser.add_argument("-temporal_weight", type=float, default=50)
    parser.add_argument("-style_weight", type=float, default=100)
    parser.add_argument("-style_blend_weights", default=None)
    parser.add_argument("-style_scale", type=float, default=1.0)
    parser.add_argument("-tv_weight", type=float, default=1e-3)

    # model settings
    parser.add_argument(
        "-model_file",
        type=str,
        default="vgg19",
        help="Path to model .pth file or one of [prune, nyud, fcn32s, sod, vgg19, vgg16, nin]",
    )
    parser.add_argument("-content_layers", help="layers for content", default="relu4_2")
    parser.add_argument("-style_layers", help="layers for style", default="relu1_1,relu2_1,relu3_1,relu4_1,relu5_1")
    parser.add_argument("-pooling", choices=["avg", "max"], default="max")
    parser.add_argument("-disable_check", action="store_true")

    # switches
    parser.add_argument("-original_colors", action="store_true")
    parser.add_argument("-normalize_weights", action="store_true")
    parser.add_argument("-no_grad_norm", action="store_true")
    parser.add_argument("-no_hist_match", action="store_true")
    parser.add_argument("-use_covariance", action="store_true")

    # optimizer
    parser.add_argument("-optimizer", choices=["lbfgs", "adam"], default="lbfgs")
    parser.add_argument("-learning_rate", type=float, default=1)
    parser.add_argument("-lbfgs_num_correction", type=int, default=100)
    parser.add_argument("-lbfgs_tolerance_change", type=int, default=-1)
    parser.add_argument("-lbfgs_tolerance_grad", type=int, default=-1)

    # gpu
    parser.add_argument("-gpu", help="Zero-indexed ID of the GPU to use; for CPU mode set -gpu = c", default=0)
    parser.add_argument(
        "-backend", choices=["nn", "cudnn", "mkl", "mkldnn", "openmp", "mkl,cudnn", "cudnn,mkl"], default="cudnn"
    )
    parser.add_argument("-multidevice_strategy", default="5")
    parser.add_argument("-no_cudnn_autotune", action="store_true")

    # video content settings
    parser.add_argument("-flow_models", type=str, default="unflow,pwc,spynet,liteflownet")
    parser.add_argument("-no_check_occlusion", action="store_true")
    parser.add_argument("-passes_per_scale", type=int, default=4)
    parser.add_argument("-loop", action="store_true")
    parser.add_argument("-temporal_blend", type=float, default=0.5)
    parser.add_argument("-fps", type=float, default=24)

    # video style settings
    parser.add_argument("-num_frames", type=int, default=48)
    parser.add_argument("-video_style_factor", type=float, default=100)
    parser.add_argument("-gram_frame_window", type=str, default="18,9,7")
    parser.add_argument("-avg_frame_window", type=int, default=18)
    parser.add_argument("-shift_factor", type=float, default=0)

    # logging
    parser.add_argument("-verbose", action="store_true")
    parser.add_argument("-print_iter", type=int, default=0)
    parser.add_argument("-save_iter", type=int, default=0)
    parser.add_argument("-save_args", action="store_true")
    parser.add_argument("-load_args", type=str, default=None)
    parser.add_argument("-ffmpeg_args", type=str, default="config/ffmpeg-libx264.json")
    parser.add_argument(
        "-scaling_args",
        type=str,
        default="config/scaling-2x11GB",
        help="multi-network multi-scale model-parallel configuration",
    )
    parser.add_argument("-uniq", action="store_true")

    args = parser.parse_args()

    output = f"{name(args.content)}_{'_'.join([name(s) for s in args.style])}"
    if args.uniq:
        output += f"_{str(uuid.uuid4())[:6]}"

    if args.save_args:
        with open(f"config/{output}_args.json", "w") as f:
            json.dump(args.__dict__, f, indent=2)

    if args.load_args is not None:
        # load from file
        file_args = argparse.Namespace()
        with open(args.load_args, "r") as f:
            file_args.__dict__ = json.load(f)

        # store any specified cmdline arguments
        non_default = {}
        argdict = vars(args)
        for key in vars(args):
            if argdict[key] != parser.get_default(key):
                non_default[key] = argdict[key]
            if key not in file_args:
                non_default[key] = argdict[key]

        # override with non-default arguments
        for key in non_default:
            setattr(file_args, key, non_default[key])
        args = file_args

    args.output = f"{args.output_dir}/{output}"

    with open(args.ffmpeg_args, "r") as f:
        ffargs = json.load(f)
    ffargs["framerate"] = args.fps
    args.ffmpeg = ffargs

    args = postprocess(args)

    return args


def postprocess(args):
    args.normalize_gradients = not args.no_grad_norm
    args.match_histograms = not args.no_hist_match
    args.cudnn_autotune = not args.no_cudnn_autotune

    args.image_sizes = [int(s) for s in ("" + args.image_sizes).split(",")]
    args.num_iters = [int(s) for s in ("" + args.num_iters).split(",")]

    assert len(args.image_sizes) == len(
        args.num_iters
    ), "-image_sizes and -num_iters must have the same number of elements!"

    # Handle style blending weights for multiple style inputs
    style_blend_weights = []
    if args.style_blend_weights is None:
        # Style blending not specified, so use equal weighting
        if args.style is not None:
            for i in args.style:
                style_blend_weights.append(1.0)
        else:
            style_blend_weights.append(1.0)
    else:
        style_blend_weights = [float(x) for x in args.style_blend_weights.split(",")]
        assert len(style_blend_weights) == len(
            args.style
        ), "-style_blend_weights and -style_images must have the same number of elements!"
    # Normalize the style blending weights so they sum to 1
    style_blend_sum = sum(style_blend_weights)
    for i, blend_weight in enumerate(style_blend_weights):
        style_blend_weights[i] = blend_weight / style_blend_sum
    args.style_blend_weights = style_blend_weights

    args.dtype, args.multidevice, args.backward_device = setup_gpu(args)

    return args


def setup_gpu(args):
    def setup_cuda():
        if "cudnn" in args.backend:
            torch.backends.cudnn.enabled = True
            if args.cudnn_autotune:
                torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.enabled = False

    def setup_cpu():
        if "mkldnn" in args.backend:
            raise ValueError("MKL-DNN is not supported yet.")
        elif "mkl" in args.backend:
            torch.backends.mkl.enabled = True
        elif "openmp" in args.backend:
            torch.backends.openmp.enabled = True

    multidevice = False
    if "," in str(args.gpu):
        devices = args.gpu.split(",")
        multidevice = True
        if "c" in str(devices[0]).lower():
            backward_device = "cpu"
            setup_cuda(), setup_cpu()
            dtype = torch.FloatTensor
        else:
            backward_device = "cuda:" + devices[0]
            setup_cuda()
            dtype = torch.cuda.FloatTensor
    elif "c" not in str(args.gpu).lower():
        setup_cuda()
        dtype, backward_device = torch.cuda.FloatTensor, "cuda:" + str(args.gpu)
    else:
        setup_cpu()
        dtype, backward_device = torch.FloatTensor, "cpu"

    return dtype, multidevice, backward_device


def load_args(filepath):
    args = argparse.Namespace()
    with open(filepath, "r") as f:
        args.__dict__ = json.load(f)

    if not args.content is None and not args.style is None:
        output = f"{name(args.content)}_{'_'.join([name(s) for s in args.style])}"
        if args.uniq:
            output += f"_{str(uuid.uuid4())[:6]}"
        args.output = f"{args.output_dir}/{output}"

    args = postprocess(args)
    # args = handle_ffmpeg(args, ffargs) TODO

    return args

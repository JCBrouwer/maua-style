import gc
import math
import os.path
import re
import traceback
import uuid

import ffmpeg
import numpy as np
import scipy.ndimage as ndi
import torch as th
import torch.nn.functional as F

import config
import flow
import load
import models
import optim
from utils import info, match_histogram, name


def img_img(args):
    style_images_big = load.process_style_images(args)
    content_image_big = match_histogram(load.preprocess(args.content), style_images_big)
    content_size = np.array(content_image_big.size()[-2:])
    if args.init != "content" and args.init != "random":
        pastiche = load.preprocess(args.init)
    else:
        pastiche = None

    for (current_size, num_iters) in zip(args.image_sizes, args.num_iters):
        print("\nCurrent size {}px".format(current_size))
        if os.path.exists(f"{args.output}_{current_size}.png"):
            pastiche = load.preprocess(f"{args.output}_{current_size}.png")
            continue

        # scale content image
        content_scale = current_size / max(*content_size)
        content_image = F.interpolate(
            content_image_big, scale_factor=content_scale, mode="bilinear", align_corners=False
        )

        # scale style images
        style_images = []
        content_area = content_image.shape[2] * content_image.shape[3]
        for img in style_images_big:
            style_scale = math.sqrt(content_area / (img.size(3) * img.size(2))) * args.style_scale
            style_images.append(
                F.interpolate(th.clone(img), scale_factor=style_scale, mode="bilinear", align_corners=False)
            )

        # Initialize the pastiche image
        if args.init == "random" and pastiche is None:
            H, W = content_image.shape[2:]
            pastiche = th.randn(1, 3, H, W).mul(0.001)
        elif args.init == "content" and pastiche is None:
            pastiche = F.interpolate(
                content_image_big.clone(),
                tuple(np.int64(content_image.shape[2:])),
                mode="bilinear",
                align_corners=False,
            )
        else:
            pastiche = F.interpolate(
                pastiche.clone(), tuple(np.int64(content_image.shape[2:])), mode="bilinear", align_corners=False
            )
        pastiche = match_histogram(pastiche, style_images_big)

        output_image = optim.optimize(content_image, style_images, pastiche, num_iters, args)

        pastiche = match_histogram(output_image.detach().cpu(), style_images_big)

        load.save_tensor_to_file(pastiche.detach().cpu(), args, size=current_size)


def img_vid(args):
    # load style videos
    style_videos_big = load.process_style_videos(args)

    # load content image
    content_image_big = load.preprocess(args.content)
    if args.match_histograms != False:
        content_image_big = match_histogram(content_image_big, style_videos_big, mode=args.match_histograms)

    # determine frame settings
    if args.num_frames == -1:
        video_length = max([vid.shape[0] for vid in style_videos_big])
    else:
        video_length = args.num_frames
    delta_ts = args.gram_frame_window.split(",")

    # initialize pastiche video
    H, W = content_size = np.array(content_image_big.size()[-2:])
    if args.init == "random":
        pastiche = th.randn((video_length, 3, H, W)) * 255
        pastiche = th.from_numpy(ndi.gaussian_filter(pastiche.numpy(), [video_length, 0, H / 32, W / 32], mode="wrap"))
    elif args.init == "content":
        pastiche = F.interpolate(content_image_big.clone(), tuple(content_size), mode="bilinear", align_corners=False)
        pastiche = pastiche.repeat([video_length, 1, 1, 1])
        pastiche += th.randn((video_length, 3, H, W)) * 255
        pastiche = th.from_numpy(ndi.gaussian_filter(pastiche.numpy(), [video_length, 0, 4, 4], mode="wrap"))
    else:
        pastiche = load.preprocess_video(args.init, args.fps)
        pastiche = pastiche.repeat([video_length, 1, 1, 1])
    if args.match_histograms != False:
        pastiche = match_histogram(pastiche, style_videos_big, mode=args.match_histograms)

    for i, (current_size, num_iters) in enumerate(zip(args.image_sizes, args.num_iters)):
        if os.path.exists(f"{args.output}_{current_size}.mp4"):
            pastiche = load.preprocess_video(f"{args.output}_{current_size}.mp4", args.fps)
            continue
        print("\nCurrent size {}px".format(current_size))
        args.gram_frame_window = int(delta_ts[i])

        # scale content image
        content_image = F.interpolate(
            content_image_big, scale_factor=current_size / max(*content_size), mode="bilinear", align_corners=False
        )

        # scale style videos
        style_videos = []
        content_area = content_image.shape[2] * content_image.shape[3]
        for vid in style_videos_big:
            style_scale = math.sqrt(content_area / (vid.size(3) * vid.size(2))) * args.style_scale
            style_videos.append(
                F.interpolate(th.clone(vid), scale_factor=style_scale, mode="bilinear", align_corners=False)
            )

        # scale pastiche video
        pastiche = F.interpolate(
            pastiche.clone(), tuple(np.int64(content_image.shape[2:])), mode="bilinear", align_corners=False
        )

        pastiche = optim.optimize(content_image, style_videos, pastiche, num_iters, args).detach().cpu()

        pastiche = th.cat((pastiche[7:], pastiche[:7]))
        style_videos_big = [th.cat((svb[7:], svb[:7])) for svb in style_videos_big]

        if args.temporal_blend > 0:
            pastiche = th.from_numpy(ndi.gaussian_filter(pastiche, [args.temporal_blend, 0, 0, 0], mode="wrap"))
        if args.match_histograms != False:
            pastiche = match_histogram(pastiche, style_videos_big, mode=args.match_histograms)
        load.save_tensor_to_file(pastiche, args, filename=f"{args.output}_{current_size}")

    load.save_tensor_to_file(match_histogram(pastiche, style_videos_big, mode=args.match_histograms), args)


def vid_img(args):
    output_dir = args.output_dir + "/" + name(args.content) + "_" + "_".join([name(s) for s in args.style])

    flow_model = flow.get_flow_model(args)
    frames = load.process_content_video(flow_model, args)
    content_size = np.array(load.preprocess(frames[0]).size()[-2:])

    style_images_big = load.process_style_images(args)

    for size_n, (current_size, num_iters) in enumerate(zip(args.image_sizes, args.num_iters)):
        print("\nCurrent size {}px".format(current_size))
        os.makedirs(output_dir + "/" + str(current_size), exist_ok=True)
        content_scale = current_size / max(*content_size)

        # scale style images
        style_images = []
        content_area = content_scale ** 2 * content_size[0] * content_size[1]
        for img in style_images_big:
            style_scale = math.sqrt(content_area / (img.size(3) * img.size(2))) * args.style_scale
            style_images.append(
                F.interpolate(th.clone(img), scale_factor=style_scale, mode="bilinear", align_corners=False)
            )
            print(style_images[-1].shape)

        if current_size <= 1024:
            args.gpu = 0
            args.multidevice = False
        else:
            args.gpu = "0,1"
            args.multidevice = True
            args.tv_weight = 0
        net, losses = models.load_model(args)
        # optim.set_style_targets(net, style_images, args)

        for pass_n in range(args.passes_per_scale):
            pastiche = None
            for (prev_frame, this_frame) in zip(frames, frames[1:] + frames[:1]):
                # TODO add update_style() function to support changing styles per frame
                args.output = "%s/%s/%s_%s.png" % (output_dir, current_size, pass_n + 1, name(this_frame))
                if os.path.isfile(args.output):
                    print("Skipping pass: %s, frame: %s. File already exists." % (pass_n + 1, name(this_frame)))
                    continue
                print("Optimizing... size: %s, pass: %s, frame: %s" % (current_size, pass_n + 1, name(this_frame)))

                content_frames = [
                    F.interpolate(
                        load.preprocess(prev_frame), scale_factor=content_scale, mode="bilinear", align_corners=False
                    ),
                    F.interpolate(
                        load.preprocess(this_frame), scale_factor=content_scale, mode="bilinear", align_corners=False
                    ),
                ]
                content_frames = [match_histogram(frame, style_images_big[0]) for frame in content_frames]
                # optim.set_content_targets(net, content_frames[1], args)

                # Initialize the image
                # TODO make sure initialization correct even when continuing half way through video stylization
                if size_n == 0 and pass_n == 0:
                    if args.init == "random":
                        pastiche = th.randn(content_frames[1].size()).mul(0.001)
                    elif args.init == "prev_warp":
                        flo_file = "%s/flow/forward_%s_%s.flo" % (output_dir, name(prev_frame), name(this_frame))
                        flow_map = load.flow_warp_map(flo_file, current_size)
                        if pastiche is None:
                            pastiche = content_frames[0]
                        pastiche = F.grid_sample(pastiche, flow_map, padding_mode="border")
                    else:
                        pastiche = content_frames[1].clone()
                else:
                    if pass_n == 0:
                        # load images from last pass of previous size
                        if pastiche is None:
                            ifile = "%s/%s/%s_%s.png" % (output_dir, prev_size, args.passes_per_scale, name(prev_frame))
                            pastiche = load.preprocess(ifile)
                            pastiche = F.interpolate(
                                pastiche, size=content_frames[0].size()[2:], mode="bilinear", align_corners=False
                            )
                        bfile = "%s/%s/%s_%s.png" % (output_dir, prev_size, args.passes_per_scale, name(this_frame))
                        blend_image = load.preprocess(bfile)
                        blend_image = F.interpolate(
                            blend_image, size=content_frames[0].size()[2:], mode="bilinear", align_corners=False
                        )
                    else:
                        # load images from previous pass of current size
                        if pastiche is None:
                            ifile = "%s/%s/%s_%s.png" % (output_dir, current_size, pass_n, name(prev_frame))
                            pastiche = load.preprocess(ifile)
                        bfile = "%s/%s/%s_%s.png" % (output_dir, current_size, pass_n, name(this_frame))
                        blend_image = load.preprocess(bfile)

                    direction = "forward" if pass_n % 2 == 0 else "backward"
                    flo_file = f"{output_dir}/flow/{direction}_{name(prev_frame)}_{name(this_frame)}.flo"
                    flow_map = load.flow_warp_map(flo_file, current_size)
                    flow_map = F.interpolate(
                        flow_map.permute(0, 3, 1, 2), size=pastiche.size()[2:], mode="bilinear"
                    ).permute(0, 2, 3, 1)

                    warp_image = F.grid_sample(pastiche, flow_map, padding_mode="border")

                    flow_weight_file = f"{output_dir}/flow/{direction}_{name(prev_frame)}_{name(this_frame)}.png"
                    reliable_flow = load.reliable_flow_weighting(flow_weight_file)
                    reliable_flow = F.interpolate(
                        reliable_flow, size=pastiche.size()[2:], mode="bilinear", align_corners=False
                    )

                    optim.set_temporal_targets(net, warp_image, warp_weights=reliable_flow, args=args)

                    blend_pastiche = (1 - args.temporal_blend) * blend_image + args.temporal_blend * pastiche
                    warp_blend_pastiche = F.grid_sample(blend_pastiche, flow_map, padding_mode="border")
                    pastiche = warp_blend_pastiche

                output_image = optim.optimize(
                    content_frames[1], style_images, pastiche, num_iters // args.passes_per_scale, args, net, losses
                )

                pastiche = match_histogram(output_image.detach().cpu(), style_images_big[0])

                disp = load.deprocess(pastiche.clone())
                if args.original_colors == 1:
                    disp = load.original_colors(load.deprocess(content_frames[1].clone()), disp)
                disp.save(str(args.output))

            # clean up / prepare for next pass
            frames = frames[7:] + frames[:7]  # rotate frames
            frames = list(reversed(frames))

        ffmpeg.input(output_dir + "/" + str(current_size) + "/" + str(pass_n) + "_%05d.png").output(
            "%s/%s_%s.mp4" % (output_dir, name(output_dir), current_size), **args.ffmpeg
        ).overwrite_output().run()
        prev_size = current_size
        del net
        th.cuda.empty_cache()

    # ffmpeg.input(f"{output_dir}/{current_size}/{pass_n}_%05d.png").output(
    #     args.output, **args.ffmpeg
    # ).overwrite_output().run()


if __name__ == "__main__":
    args = config.get_args()

    if args.seed >= 0:
        th.manual_seed(args.seed)
        th.cuda.manual_seed_all(args.seed)
        if args.backend == "cudnn":
            th.backends.cudnn.deterministic = True

    eval(args.transfer_type)(args)

import math
import os.path
import load
import uuid
import style
from config import load_config
import torch as th
import torch.nn.functional as F
import numpy as np
import ffmpeg
import traceback
import scipy.ndimage as ndi
import flow
from utils import info, name, determine_scaling, match_histogram
import models
import re
import gc


def img_img(opt):
    style_images_big = load.process_style_images(opt)
    content_image_big = match_histogram(load.preprocess(opt.input.content), style_images_big)
    content_size = np.array(content_image_big.size()[-2:])
    if opt.input.init != "content" and opt.input.init != "random":
        pastiche = load.preprocess(opt.input.init)
    else:
        pastiche = None

    for (current_size, num_iters) in zip(*determine_scaling(opt.param)):
        print("\nCurrent size {}px".format(current_size))
        if os.path.exists(f"{opt.output}_{current_size}.png"):
            pastiche = load.preprocess(f"{opt.output}_{current_size}.png")
            continue

        # scale content image
        content_scale = current_size / max(*content_size)
        if not content_scale == 1:
            content_image = F.interpolate(
                content_image_big, scale_factor=content_scale, mode="bilinear", align_corners=False
            )

        # scale style images
        style_images = []
        content_area = content_image.shape[2] * content_image.shape[3]
        for img in style_images_big:
            style_scale = math.sqrt(content_area / (img.size(3) * img.size(2))) * opt.param.style_scale
            style_images.append(
                F.interpolate(th.clone(img), scale_factor=style_scale, mode="bilinear", align_corners=False)
            )

        # Initialize the pastiche image
        if opt.input.init == "random" and pastiche is None:
            H, W = content_image.shape[2:]
            pastiche = th.randn(1, 3, H, W).mul(0.001)
        elif opt.input.init == "content" and pastiche is None:
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

        output_image = style.optimize(content_image, style_images, pastiche, num_iters, opt)

        pastiche = match_histogram(output_image.detach().cpu(), style_images_big)

        load.save_tensor_to_file(pastiche.detach().cpu(), opt)


def img_vid(opt):
    # load style videos
    style_videos_big = load.process_style_videos(opt)
    # style_videos_big = [svb[:144] for svb in style_videos_big]

    # load content image
    content_image_big = load.preprocess(opt.input.content)
    if opt.param.match_histograms != False:
        content_image_big = match_histogram(content_image_big, style_videos_big, mode=opt.param.match_histograms)

    # determine frame settings
    if opt.param.num_frames == -1:
        video_length = max([vid.shape[0] for vid in style_videos_big])
    else:
        video_length = opt.param.num_frames
    delta_ts = opt.param.gram_frame_window.split(",")

    # initialize pastiche image
    content_size = np.array(content_image_big.size()[-2:])
    if opt.input.init == "random":
        H, W = content_size
        pastiche = th.randn((video_length, 3, H, W)) * 255
        pastiche = th.from_numpy(
            ndi.gaussian_filter(pastiche.numpy(), [video_length // 6, 0, H / 128, W / 128], mode="wrap")
        )
    elif opt.input.init == "content":
        pastiche = F.interpolate(content_image_big.clone(), tuple(content_size), mode="bilinear", align_corners=False)
        pastiche = pastiche.repeat([video_length, 1, 1, 1])
        H, W = content_size
        pastiche += th.randn((video_length, 3, H, W)) * 255
        pastiche = th.from_numpy(ndi.gaussian_filter(pastiche.numpy(), [video_length // 6, 0, 4, 4], mode="wrap"))
    else:
        pastiche = load.preprocess_video(opt.input.init, opt.ffmpeg.fps)
        pastiche = pastiche.repeat([video_length, 1, 1, 1])
    if opt.param.match_histograms != False:
        pastiche = match_histogram(pastiche, style_videos_big, mode=opt.param.match_histograms)

    for i, (current_size, num_iters) in enumerate(zip(*determine_scaling(opt.param))):
        print("\nCurrent size {}px".format(current_size))
        opt.param.gram_frame_window = int(delta_ts[i])

        # scale content image
        content_image = F.interpolate(
            content_image_big, scale_factor=current_size / max(*content_size), mode="bilinear", align_corners=False
        )

        # scale style videos
        style_videos = []
        content_area = content_image.shape[2] * content_image.shape[3]
        for vid in style_videos_big:
            style_scale = math.sqrt(content_area / (vid.size(3) * vid.size(2))) * opt.param.style_scale
            style_videos.append(
                F.interpolate(th.clone(vid), scale_factor=style_scale, mode="bilinear", align_corners=False)
            )

        # scale pastiche video
        pastiche = F.interpolate(
            pastiche.clone(), tuple(np.int64(content_image.shape[2:])), mode="bilinear", align_corners=False
        )

        pastiche = style.optimize(content_image, style_videos, pastiche, num_iters, opt).detach().cpu()

        # pastiche = th.cat((output[7:], output[:7]))
        # style_videos_big = [th.cat((svb[7:], svb[:7])) for svb in style_videos_big]

        if opt.param.temporal_smoothing > 0:
            pastiche = th.from_numpy(
                ndi.gaussian_filter(pastiche, [opt.param.temporal_smoothing, 0, 0, 0], mode="wrap")
            )
        if opt.param.match_histograms != False:
            pastiche = match_histogram(pastiche, style_videos_big, mode=opt.param.match_histograms)
        load.save_tensor_to_file(pastiche, opt, filename=f"{opt.output}_{current_size}")
        # load.save_tensor_to_file(pastiche, opt, filename=f"{opt.output}_raw_{current_size}")
        exit()

    load.save_tensor_to_file(match_histogram(pastiche, style_videos_big, mode=opt.param.match_histograms), opt)


# TODO fix vid_img to work with new optimize function, figure out how to setup network & style targets only once!
def vid_img(opt):
    output_dir = (
        opt.output_dir + "/" + name(opt.input.content) + "_" + "_".join([name(s) for s in opt.input.style.split(",")])
    )

    flow_model = flow.get_flow_model(opt)
    frames = load.process_content_video(flow_model, opt)
    content_size = np.array(load.preprocess(frames[0]).size()[-2:])

    style_images_big = load.process_style_images(opt)

    for size_n, (current_size, num_iters) in enumerate(zip(*determine_scaling(opt.param))):
        print("\nCurrent size {}px".format(current_size))
        os.makedirs(output_dir + "/" + str(current_size), exist_ok=True)
        content_scale = current_size / max(*content_size)

        if current_size <= 1024:
            opt.model.gpu = 0
            opt.model.multidevice = False
        else:
            opt.model.gpu = "0,1"
            opt.model.multidevice = True
            opt.param.tv_weight = 0
        net = models.load_model(opt.model, opt.param)
        net.set_style_targets(style_images_big, content_scale * content_size, opt)

        for pass_n in range(opt.param.passes_per_scale):
            init_image = None
            for (prev_frame, this_frame) in zip(frames, frames[1:] + frames[:1]):
                # TODO add update_style() function to support changing styles per frame
                opt.output = "%s/%s/%s_%s.png" % (output_dir, current_size, pass_n + 1, name(this_frame))
                if os.path.isfile(opt.output):
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
                net.set_content_targets(content_frames[1], opt)

                # Initialize the image
                # TODO make sure initialization correct even when continuing half way through video stylization
                if size_n == 0 and pass_n == 0:
                    if opt.input.init == "random":
                        init_image = th.randn(content_frames[1].size()).mul(0.001)
                    elif opt.input.init == "prev_warp":
                        flo_file = "%s/flow/forward_%s_%s.flo" % (output_dir, name(prev_frame), name(this_frame))
                        flow_map = load.flow_warp_map(flo_file)
                        if init_image is None:
                            init_image = content_frames[0]
                        init_image = F.grid_sample(init_image, flow_map, padding_mode="border")
                    else:
                        init_image = content_frames[1].clone()
                else:
                    if pass_n == 0:
                        # load images from last pass of previous size
                        if init_image is None:
                            ifile = "%s/%s/%s_%s.png" % (
                                output_dir,
                                prev_size,
                                opt.param.passes_per_scale,
                                name(prev_frame),
                            )
                            init_image = load.preprocess(ifile)
                            init_image = F.interpolate(
                                init_image, size=content_frames[0].size()[2:], mode="bilinear", align_corners=False
                            )
                        bfile = "%s/%s/%s_%s.png" % (
                            output_dir,
                            prev_size,
                            opt.param.passes_per_scale,
                            name(this_frame),
                        )
                        blend_image = load.preprocess(bfile)
                        blend_image = F.interpolate(
                            blend_image, size=content_frames[0].size()[2:], mode="bilinear", align_corners=False
                        )
                    else:
                        # load images from previous pass of current size
                        if init_image is None:
                            ifile = "%s/%s/%s_%s.png" % (output_dir, current_size, pass_n, name(prev_frame))
                            init_image = load.preprocess(ifile)
                        bfile = "%s/%s/%s_%s.png" % (output_dir, current_size, pass_n, name(this_frame))
                        blend_image = load.preprocess(bfile)

                    direction = "forward" if pass_n % 2 == 0 else "backward"
                    flo_file = f"{output_dir}/flow/{direction}_{name(prev_frame)}_{name(this_frame)}.flo"
                    flow_map = load.flow_warp_map(flo_file)
                    flow_map = F.interpolate(
                        flow_map.permute(0, 3, 1, 2), size=init_image.size()[2:], mode="bilinear"
                    ).permute(0, 2, 3, 1)

                    warp_image = F.grid_sample(init_image, flow_map, padding_mode="border")

                    flow_weight_file = f"{output_dir}/flow/{direction}_{name(prev_frame)}_{name(this_frame)}.png"
                    reliable_flow = load.reliable_flow_weighting(flow_weight_file)
                    reliable_flow = F.interpolate(
                        reliable_flow, size=init_image.size()[2:], mode="bilinear", align_corners=False
                    )

                    net.set_temporal_targets(warp_image, warp_weights=reliable_flow, opt=opt)

                    blend_init_image = (1 - opt.param.blend_weight) * blend_image + opt.param.blend_weight * init_image
                    warp_blend_init_image = F.grid_sample(blend_init_image, flow_map, padding_mode="border")
                    init_image = warp_blend_init_image

                output_image = style.optimize(
                    content_frames, style_images_big, init_image, num_iters // opt.param.passes_per_scale, opt
                )

                init_image = match_histogram(output_image.detach().cpu(), style_images_big[0])

                disp = load.deprocess(init_image.clone())
                if opt.param.original_colors == 1:
                    disp = load.original_colors(load.deprocess(content_frames[1].clone()), disp)
                disp.save(str(opt.output))

            # clean up / prepare for next pass
            frames = frames[7:] + frames[:7]  # rotate frames
            frames = list(reversed(frames))

        ffmpeg.input(output_dir + "/" + str(current_size) + "/" + str(pass_n) + "_%05d.png").output(
            "%s/%s_%s.mp4" % (output_dir, name(output_dir), current_size), **opt.ffmpeg
        ).overwrite_output().run()
        prev_size = current_size
        del net
        th.cuda.empty_cache()

    ffmpeg.input("{output_dir}/{current_size}/{pass_n}_%05d.png").output(
        opt.output, **opt.ffmpeg
    ).overwrite_output().run()


opt = load_config("config/ub96.yaml")
opt.output = f"{opt.output_dir}/{name(opt.input.content)}_{'_'.join([name(s) for s in opt.input.style.split(',')])}"
if opt.transfer_type == "img_img":
    img_img(opt)
elif opt.transfer_type == "vid_img":
    vid_img(opt)
elif opt.transfer_type == "img_vid":
    img_vid(opt)

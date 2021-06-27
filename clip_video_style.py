import os
import sys
from glob import glob
from random import randrange

import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F

import clip_vqgan
import config
import flow
import load
from utils import match_histogram, name

torch.backends.cudnn.benchmark = True

args = config.get_args()

if args.seed >= 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.backend == "cudnn":
        torch.backends.cudnn.deterministic = True

output_dir = args.output_dir + "/" + name(args.content) + "_" + "_".join([name(s) for s in args.style])

flow_model = flow.get_flow_model(args)
frames = load.process_content_video(flow_model, args)
content_size = np.array(load.preprocess(frames[0]).size()[-2:])

style_images_big = load.process_style_images(args)

for size_n, (current_size, num_iters) in enumerate(zip(args.image_sizes, args.num_iters)):

    if len(glob("%s/%s/*.png" % (output_dir, args.image_sizes[min(len(args.image_sizes) - 1, size_n + 1)]))) > 1:
        print("Skipping size: %s, already done." % current_size)
        prev_size = current_size
        continue

    print("\nCurrent size {}px".format(current_size))
    os.makedirs(output_dir + "/" + str(current_size), exist_ok=True)
    content_scale = current_size / max(*content_size)

    # scale style images
    style_images = []
    content_area = content_scale ** 2 * content_size[0] * content_size[1]
    for img in style_images_big:
        style_scale = (content_area / (img.size(3) * img.size(2))) ** 0.5 * args.style_scale
        style_images.append(
            F.interpolate(torch.clone(img), scale_factor=style_scale, mode="bilinear", align_corners=False)
        )

    if size_n != 0:
        clip_vqgan.update_styles(style_images, args.content_text, args.style_text)

    for pass_n in range(args.passes_per_scale):
        pastiche = None

        if args.loop:
            start_idx = randrange(0, len(frames) - 1)
            frames = frames[start_idx:] + frames[:start_idx]  # rotate frames

        if len(glob("%s/%s/%s_*.png" % (output_dir, current_size, pass_n + 2))) > 1:
            print(f"Skipping pass: {pass_n + 1}, already done.")
            frames = list(reversed(frames))
            continue

        for n, (prev_frame, this_frame) in enumerate(
            zip(frames + frames[: 11 if args.loop else 1], frames[1:] + frames[: 10 if args.loop else 1])
        ):
            # TODO add update_style() function to support changing styles per frame

            args.output = "%s/%s/%s_%s.png" % (output_dir, current_size, pass_n + 1, name(this_frame))
            if os.path.isfile(args.output) and not n >= len(frames):
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
            flow_direction = "forward" if pass_n % 2 == 0 else "backward"

            # Initialize the image
            # TODO make sure initialization correct even when continuing half way through video stylization
            if size_n == 0 and pass_n == 0:
                if args.init == "random":
                    pastiche = torch.randn(content_frames[1].size()).mul(0.001)
                elif args.init == "prev_warp":
                    if pastiche is None:
                        pastiche = content_frames[0]
                    flo_file = f"{output_dir}/flow/{flow_direction}_{name(prev_frame)}_{name(this_frame)}.flo"
                    flow_map = load.flow_warp_map(flo_file, pastiche.shape[2:])
                    pastiche = F.grid_sample(pastiche, flow_map, padding_mode="border")
                else:
                    pastiche = content_frames[1].clone()
                mask = None
            else:
                if pass_n == 0:
                    # load images from last pass of previous size
                    if pastiche is None:
                        ifile = "%s/%s/%s_%s.png" % (
                            output_dir,
                            prev_size if n <= len(frames) else current_size,
                            args.passes_per_scale if n <= len(frames) else pass_n + 1,
                            name(prev_frame),
                        )
                        pastiche = load.preprocess(ifile)
                        pastiche = F.interpolate(
                            pastiche, size=content_frames[0].size()[2:], mode="bilinear", align_corners=False
                        )
                    bfile = "%s/%s/%s_%s.png" % (
                        output_dir,
                        prev_size if n <= len(frames) else current_size,
                        args.passes_per_scale if n <= len(frames) else pass_n + 1,
                        name(this_frame),
                    )
                    blend_image = load.preprocess(bfile)
                    blend_image = F.interpolate(
                        blend_image, size=content_frames[0].size()[2:], mode="bilinear", align_corners=False
                    )
                else:
                    # load images from previous pass of current size
                    if pastiche is None:
                        ifile = "%s/%s/%s_%s.png" % (
                            output_dir,
                            current_size,
                            pass_n if n <= len(frames) else pass_n + 1,
                            name(prev_frame),
                        )
                        pastiche = load.preprocess(ifile)
                    bfile = "%s/%s/%s_%s.png" % (
                        output_dir,
                        current_size,
                        pass_n if n <= len(frames) else pass_n + 1,
                        name(this_frame),
                    )
                    blend_image = load.preprocess(bfile)

                flo_file = f"{output_dir}/flow/{flow_direction}_{name(prev_frame)}_{name(this_frame)}.flo"
                flow_map = load.flow_warp_map(flo_file, pastiche.shape[2:])

                warp_image = F.grid_sample(pastiche, flow_map, padding_mode="border")

                flow_weight_file = f"{output_dir}/flow/{flow_direction}_{name(prev_frame)}_{name(this_frame)}.png"
                reliable_flow = load.reliable_flow_weighting(flow_weight_file)
                reliable_flow = F.interpolate(
                    reliable_flow, size=pastiche.size()[2:], mode="bilinear", align_corners=False
                )

                mask = (reliable_flow + 1) / 2

                pastiche = (1 - args.temporal_blend) * blend_image + args.temporal_blend * pastiche

            output_image = clip_vqgan.optimize_cached(
                init=pastiche,
                content=content_frames[1],
                style=style_images,
                mask=None,
                content_text=args.content_text,
                style_text=args.style_text,
                content_weight=args.content_weight,
                style_weight=args.style_weight,
                text_weight=1,
                model_dir=args.vqgan_dir,
                clip_backbone=args.clip_backbone,
                iterations=num_iters // args.passes_per_scale,
            )

            pastiche = match_histogram(output_image.detach().cpu(), style_images_big[0])

            disp = load.deprocess(pastiche.clone())
            if args.original_colors == 1:
                disp = load.original_colors(load.deprocess(content_frames[1].clone()), disp)
            disp.save(str(args.output))

        # reverse frames for next pass
        frames = list(reversed(frames))

    ffmpeg.input(output_dir + "/" + str(current_size) + "/" + str(pass_n) + "_%05d.png").output(
        "%s/%s_%s.mp4" % (output_dir, name(output_dir), current_size), **args.ffmpeg
    ).overwrite_output().run()
    prev_size = current_size
    torch.cuda.empty_cache()

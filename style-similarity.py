import os
import glob
import tqdm
import load
import json
import torch
import PIL.Image
import config
import canvas
import models
import pathos
import numpy as np

dataset_folder = "/home/hans/datasets/trawt"
with torch.no_grad():
    opt = config.load_config("config/sim.yaml")
    net = models.load_model(opt.model, opt.param)

    images = []
    # for color in filter(lambda x: not "csv" in x, glob.glob(dataset_folder + "/*")):
    #     for img_file in glob.glob(color + "/*.png"):
    #           print(img_file)
    for img_file in glob.glob(dataset_folder + "/all/*"):
        images.append(img_file)
    print("got image files...")

    target_dict = {}
    for img_file in images:
        embedding_filename = (
            f"{dataset_folder}/vgg_embeddings/{img_file.split('trawt/all/')[-1].replace('/','_').split('.')[0]}"
        )
        if os.path.exists(f"{embedding_filename}_4.npy"):
            target_dict[img_file] = []
            for i, mod in enumerate(net.style_losses):
                target_dict[img_file].append(np.load(f"{embedding_filename}_{i}.npy"))
            continue
        print(embedding_filename)

        opt.input.content = "random"
        opt.input.style = img_file
        style_image = load.process_style_images(opt)
        net.set_style_targets(style_image, [1500, 1500], opt=opt)
        target_dict[img_file] = []
        for i, mod in enumerate(net.style_losses):
            target_dict[img_file].append(mod.target.to("cpu").numpy().copy())
            mod.target = torch.Tensor()
        np.save(f"{embedding_filename}_{i}.npy", mod.target.to("cpu").numpy())
    print("got embedding files...")

    if os.path.exists(f"{dataset_folder}/max_dists.npy"):
        maxima = np.load(f"{dataset_folder}/max_dists.npy")
        minima = np.load(f"{dataset_folder}/min_dists.npy")
    else:

        def get_max_dists(ims):
            max_dists = [0, 0, 0, 0, 0]
            min_dists = [np.inf, np.inf, np.inf, np.inf, np.inf]
            for img_file in ims:
                targets = target_dict[img_file]
                for other_file, other_targets in target_dict.items():
                    if img_file == other_file:
                        continue
                    dist = 0
                    for n, (t, ot) in enumerate(zip(targets, other_targets)):
                        sub_dist = np.sum((t - ot) ** 2)
                        dist += sub_dist
                        max_dists[n] = max(max_dists[n], sub_dist)
                        min_dists[n] = min(min_dists[n], sub_dist)
            return max_dists, min_dists

        results = pathos.multiprocessing.ProcessPool(nodes=24).map(
            get_max_dists,
            [images[start : start + len(images) // 24] for start in range(0, len(images), len(images) // 24)],
        )
        maxs = np.array([r[0] for r in results])
        mins = np.array([r[1] for r in results])
        maxima = maxs.max(axis=0)
        minima = mins.min(axis=0)

        print(minima)
        print(maxima)

        np.save(f"{dataset_folder}/min_dists.npy", minima)
        np.save(f"{dataset_folder}/max_dists.npy", maxima)
    print("got min/max distances...")

    if os.path.exists(f"{dataset_folder}/closest.json"):
        with open(f"{dataset_folder}/closest.json", "r") as j:
            closest = json.loads(j.read())
    else:
        top_n = 6

        def get_closest_imgs(ims):
            closest_imgs = {}
            all_files = list(target_dict.keys())
            for img_file in ims:
                targets = target_dict[img_file]
                dists = []
                for other_file, other_targets in target_dict.items():
                    if img_file == other_file:
                        continue
                    dist = 0
                    for n, (t, ot) in enumerate(zip(targets, other_targets)):
                        sub_dist = (np.sum((t - ot) ** 2) - minima[n]) / (maxima[n] - minima[n])
                        # sub_dist = np.sum((t - ot) ** 2)
                        dist += sub_dist
                    dists.append(dist)
                best_indices = np.argpartition(dists, -top_n)[-top_n:]
                closest_imgs[img_file] = [all_files[i] for i in best_indices]

            return closest_imgs

        closest = pathos.multiprocessing.ProcessPool(nodes=24).map(
            get_closest_imgs,
            [images[start : start + len(images) // 24] for start in range(0, len(images), len(images) // 24)],
        )
        closest = {k: v for d in closest for k, v in d.items()}

        with open(f"{dataset_folder}/closest.json", "w") as file:
            json.dump(closest, file)
    print("got closest perceptual style neighbors...")

    for og, closes in closest.items():
        grid = PIL.Image.new("RGB", (900, 900))

        im = PIL.Image.open(og)
        im.thumbnail((300, 300))
        grid.paste(im, (0, 0))

        index = 0
        for i in range(300, 900, 300):
            for j in range(0, 900, 300):
                im = PIL.Image.open(closes[index])
                im.thumbnail((300, 300))
                grid.paste(im, (i, j))
                index += 1

        grid.save(f"{dataset_folder}/grids/{og.split('/')[-1].split('.')[0]}.png")

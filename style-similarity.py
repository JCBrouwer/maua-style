import os
import glob
import load
import json
import torch
import config
import canvas
import models
import pathos
import PIL.Image
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import scipy.stats
import imageio
import numpy as np

dataset_name = "cyphis"
dataset_folder = f"/home/hans/datasets/{dataset_name}"

images = []
for img_file in glob.glob(dataset_folder + "/all/*"):
    images.append(img_file)


print("getting image histograms...")
if not os.path.exists(f"{dataset_folder}/hists.npy"):
    num_bins = 64
    hists = np.zeros((len(images), 3, num_bins))
    for i, img_file in enumerate(tqdm(images)):
        img = imageio.imread(img_file)
        for k in range(3):
            hists[i, k] = np.histogram(img[:, :, k], bins=num_bins)[0] / 3
    np.save(f"{dataset_folder}/hists.npy", hists)
else:
    hists = np.load(f"{dataset_folder}/hists.npy")


def chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])


print("getting distances between histograms...")
if not os.path.exists(f"{dataset_folder}/dists.npy"):
    dists = np.zeros((len(images), len(images)))
    for i, hist1 in enumerate(tqdm(hists)):
        for j, hist2 in enumerate(hists):
            if (hist1 == hist2).all():
                dists[i, j] = np.infty
                continue
            # dists[i, j] = scipy.stats.wasserstein_distance(hist1.flatten(), hist2.flatten())
            dists[i, j] = chi2_distance(hist1.flatten(), hist2.flatten())
    np.save(f"{dataset_folder}/dists.npy", dists)
else:
    dists = np.load(f"{dataset_folder}/dists.npy")

top_n = 6
best_indices = np.argpartition(dists, top_n, axis=1)[:, :top_n]

print(best_indices.shape, len(images))
closest = [[images[j] for j in best_indices[i]] for i in range(len(images))]

print("saving grids of closest neighbors...")
for ii, closest in enumerate(tqdm(closest)):
    grid = PIL.Image.new("RGB", (900, 900))

    im = PIL.Image.open(images[ii])
    im.thumbnail((300, 300))
    grid.paste(im, (0, 0))

    index = 0
    for i in range(300, 900, 300):
        for j in range(0, 900, 300):
            im = PIL.Image.open(closest[index])
            im.thumbnail((300, 300))
            grid.paste(im, (i, j))
            index += 1

    grid.save(f"{dataset_folder}/grids/{images[ii].split('/')[-1].split('.')[0]}.png")

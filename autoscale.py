import argparse
import gc
import json
import math

import torch

import config
import models
import optim

parser = argparse.ArgumentParser()
parser.add_argument("-min_size", type=int, default=256)
parser.add_argument("-models", nargs="*", default=["vgg19", "vgg16", "sod", "nyud", "prune", "nin"])
parser.add_argument("-precision", choices=["low", "med", "high"], default="low")
autoscale_args = parser.parse_args()

num_gpus = torch.cuda.device_count()

mods = autoscale_args.models
opts = ["lbfgs", "adam"]

if autoscale_args.precision == "low":
    growth_factor = math.sqrt(2)
if autoscale_args.precision == "med":
    growth_factor = math.sqrt(math.sqrt(2))
if autoscale_args.precision == "high":
    growth_factor = math.sqrt(math.sqrt(math.sqrt(2)))

min_size = autoscale_args.min_size

print("\n\n\n\nmodels:", mods)
print("optimizers:", opts)
print("#GPUs:", num_gpus)
print("starting from size:", min_size)
print("scaling with factor:", growth_factor, "\n\n\n")

im = lambda size: torch.rand(size=(1, 3, int(round(size)), int(round(size)))) * 255

args = config.load_args("config/img_img_2x11GB.json")
args.print_iter = -1
args.save_iter = -1

max_sizes = {}

for mod, opt, gpus in [(mod, opt, gpus) for gpus in range(1, num_gpus + 1) for opt in opts for mod in mods]:
    conf = f"{mod}+{opt}+{gpus}"
    print(f"\nmodel: {mod.upper()}   optimizer: {opt.upper()}   #GPUs: {gpus}")
    max_sizes[conf] = {}

    if opt == "lbfgs" and gpus == 1:
        size = min_size / growth_factor
    else:
        # size >= that of previous more memory hungry config
        size = max_sizes[f"{mod}+lbfgs+{max(gpus - 1, 1)}"]["safe_max_size"]

    args.model_file = mod
    if mod == "nin":
        args.content_layers = "relu8"
        args.style_layers = "relu1,relu3,relu5,relu7,relu9,relu11"
    else:
        args.content_layers = "relu4_2"
        args.style_layers = "relu1_1,relu2_1,relu3_1,relu4_1,relu5_1"

    args.optimizer = opt

    if gpus > 1:
        args.multidevice = True
    else:
        args.multidevice = False

    oom = False
    while not oom:
        size *= growth_factor
        print(f"{int(round(size))}x{int(round(size))}")
        try:
            optim.optimize(content=im(size), styles=[im(size)], init=im(size), num_iters=150, args=args)
        except RuntimeError as e:
            if not "out of memory" in str(e):
                raise e
            print("Ran out of memory...")
            max_sizes[conf]["safe_max_size"] = size / growth_factor
            max_sizes[conf]["true_max_size"] = size
            max_sizes[conf]["iters_b4_oom"] = optim.PBAR.n
            del optim
            gc.collect()
            torch.cuda.empty_cache()
            oom = True
            import optim
    print("safe max size:", int(round(max_sizes[conf]["safe_max_size"])))
    print("true max size:", int(round(max_sizes[conf]["true_max_size"])))
    print("iterations before OOM:", int(round(max_sizes[conf]["iters_b4_oom"])))

print("\n\n\n", max_sizes)

with open("config/scaling-2x11GB.json", "w") as fp:
    json.dump(max_sizes, fp)

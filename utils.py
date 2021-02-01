import numpy as np
import torch as th


def info(x, y=None, z=None):
    if z is None:
        if y is None:
            print(
                f"{x.min().detach().cpu().item():.2f}",
                f"{x.mean().detach().cpu().item():.2f}",
                f"{x.max().detach().cpu().item():.2f}",
                x.shape,
            )
        else:
            print(
                f"{x.min().detach().cpu().item():.2f}",
                f"{x.mean().detach().cpu().item():.2f}",
                f"{x.max().detach().cpu().item():.2f}",
                x.shape,
                f"{y.min().detach().cpu().item():.2f}",
                f"{y.mean().detach().cpu().item():.2f}",
                f"{y.max().detach().cpu().item():.2f}",
                y.shape,
            )
    else:
        if y is None:
            print(
                z,
                f"{x.min().detach().cpu().item():.2f}",
                f"{x.mean().detach().cpu().item():.2f}",
                f"{x.max().detach().cpu().item():.2f}",
                x.shape,
            )
        else:
            print(
                z,
                f"{x.min().detach().cpu().item():.2f}",
                f"{x.mean().detach().cpu().item():.2f}",
                f"{x.max().detach().cpu().item():.2f}",
                x.shape,
                f"{y.min().detach().cpu().item():.2f}",
                f"{y.mean().detach().cpu().item():.2f}",
                f"{y.max().detach().cpu().item():.2f}",
                y.shape,
            )


def name(s):
    return s.split("/")[-1].split(".")[0]


def wrapping_slice(tensor, start, length, return_indices=False):
    if start + length <= tensor.shape[0]:
        indices = th.arange(start, start + length)
    else:
        indices = th.cat((th.arange(start, tensor.shape[0]), th.arange(0, (start + length) % tensor.shape[0])))
    if tensor.shape[0] == 1:
        indices = th.zeros(1, dtype=th.int64)
    if return_indices:
        return indices
    return tensor[indices]


def get_histogram(tensor, eps):
    mu_h = tensor.mean(list(range(len(tensor.shape) - 1)))
    h = tensor - mu_h
    h = h.permute(0, 3, 1, 2).reshape(tensor.size(3), -1)
    Ch = th.mm(h, h.T) / h.shape[1] + eps * th.eye(h.shape[0])
    return mu_h, h, Ch


def match_histogram(target_tensor, source_tensor, eps=1e-2, mode="avg"):
    backup = target_tensor.clone()
    try:
        if mode == "avg":
            elementwise = True
            random_frame = False
        else:
            elementwise = False
            random_frame = True

        if not isinstance(source_tensor, list):
            source_tensor = [source_tensor]

        output_tensor = th.zeros_like(target_tensor)
        for source in source_tensor:
            target = target_tensor.permute(0, 3, 2, 1)  # Function expects b,w,h,c
            source = source.permute(0, 3, 2, 1)  # Function expects b,w,h,c
            if elementwise:
                source = source.mean(0).unsqueeze(0)
            if random_frame:
                source = source[np.random.randint(0, source.shape[0])].unsqueeze(0)

            matched_tensor = th.zeros_like(target)
            for idx in range(target.shape[0] if elementwise else 1):
                frame = target[idx].unsqueeze(0) if elementwise else target
                _, t, Ct = get_histogram(frame + 1e-3 * th.randn(size=frame.shape), eps)
                mu_s, _, Cs = get_histogram(source + 1e-3 * th.randn(size=source.shape), eps)

                # PCA
                eva_t, eve_t = th.symeig(Ct, eigenvectors=True, upper=True)
                Et = th.sqrt(th.diagflat(eva_t))
                Et[Et != Et] = 0  # Convert nan to 0
                Qt = th.mm(th.mm(eve_t, Et), eve_t.T)

                eva_s, eve_s = th.symeig(Cs, eigenvectors=True, upper=True)
                Es = th.sqrt(th.diagflat(eva_s))
                Es[Es != Es] = 0  # Convert nan to 0
                Qs = th.mm(th.mm(eve_s, Es), eve_s.T)

                ts = th.mm(th.mm(Qs, th.inverse(Qt)), t)

                match = ts.reshape(*frame.permute(0, 3, 1, 2).shape).permute(0, 2, 3, 1)
                match += mu_s

                if elementwise:
                    matched_tensor[idx] = match
                else:
                    matched_tensor = match
            output_tensor += matched_tensor.permute(0, 3, 2, 1) / len(source_tensor)
    except RuntimeError:
        traceback.print_exc()
        print("Skipping histogram matching...")
        output_tensor = backup
    return output_tensor


def determine_scaling(opt):
    def opt_to_ints(scales, scaling, num):
        scales = ("" + scales).split(",")
        if len(scales) == 2:
            up = max(float(scales[1]), float(scales[0]))
            low = min(float(scales[1]), float(scales[0]))
            if scaling == "linear":
                scales = [int(low + i * (up - low) / (num - 1)) for i in range(num)]
            elif scaling == "power":
                factor = (up / low) ** (1 / (num - 1))
                scales = [int(round(low * factor ** i)) for i in range(num)]
            else:
                print("Scaling type: " + str(scaling) + " not recognized. Options are: [linear, power]")
        else:
            scales = [int(s) for s in scales]
        return scales

    image_sizes = opt_to_ints(opt.image_sizes, opt.size_scaling, opt.num_scales)
    num_iters = opt_to_ints(opt.num_iterations, opt.iter_scaling, opt.num_scales)
    return image_sizes, reversed(num_iters)

"""
Batch Cellpose runner for nuclei (DAPI) segmentation.

Run this INSIDE a conda env that has PyTorch (with CUDA) + Cellpose installed -
not inside your StarDist/TensorFlow env. The notebook calls it as a subprocess,
exactly like the StarDist batch runner.

Manifest format (same as the StarDist runner):
    [{"tif_path": "...", "dapi_channel": 2, "output_path": "./seg_outputs/cellpose/<core>.npy"}, ...]

Usage:
    python cellpose_runner_batch.py manifest.json [--diameter 15] [--batch-size 8] [--cpu] [--overwrite]
    python cellpose_runner_batch.py --write-config run_config.json      # only record the settings, no segmentation

Outputs:
    * one int32 label image (.npy) per manifest entry, written to output_path
    * timings.json next to the outputs: seconds per image + device + cellpose version
      (the notebook uses this for the runtime comparison)
    * run_config.json next to the outputs: the exact model and EVERY eval() setting used (explicit ones and library
      defaults), plus library versions. Quote it in the Methods section.

Safety: the script REFUSES to run on CPU unless you pass --cpu explicitly, so you never
silently start an hours-long CPU run.
"""
import argparse
import inspect
import json
import os
import sys
import time

import numpy as np
from skimage import io


def load_channel(tif_path, channel):
    """Read one channel from an (H, W, C) or (C, H, W) image."""
    img = io.imread(tif_path)
    if img.ndim == 2:
        return img
    if img.shape[-1] <= 4 < img.shape[0]:      # channels-last (H, W, C)
        return img[..., channel]
    if img.shape[0] <= 4 < img.shape[-1]:      # channels-first (C, H, W)
        return img[channel]
    return img[..., channel]


def get_major_version():
    try:
        from importlib.metadata import version
        v = version("cellpose")
        return int(v.split(".")[0]), v
    except Exception:
        return 4, "unknown (assuming >=4)"


def build_model(major, use_gpu):
    from cellpose import models
    if major >= 4:
        # Cellpose v4+: single "Cellpose-SAM" model; model_type / channels are ignored.
        return models.CellposeModel(gpu=use_gpu)
    # Cellpose v3 and older: dedicated nuclei model.
    return models.CellposeModel(gpu=use_gpu, model_type="nuclei")


def segment(model, major, image, diameter, batch_size):
    kwargs = overrides_for(major, diameter, batch_size)
    out = model.eval(image, **kwargs)
    return out[0]                              # masks; flows/styles are not needed


def overrides_for(major, diameter, batch_size):
    """The eval() settings this script sets explicitly (everything else is a library default)."""
    o = dict(diameter=diameter, batch_size=batch_size, flow_threshold=0.4, cellprob_threshold=0.0)
    if major < 4:
        o["channels"] = [0, 0]
    return o


def collect_config(model, major, version_str, device, torch_version, diameter, batch_size):
    """Record the model and every eval() argument (explicit values + library defaults)."""
    explicit = overrides_for(major, diameter, batch_size)
    sig = inspect.signature(model.eval)
    defaults = {k: (None if v.default is inspect.Parameter.empty else v.default)
                for k, v in sig.parameters.items() if k not in ("self", "x")}
    return {
        "cellpose_version": version_str,
        "torch_version": torch_version,
        "device": device,
        "model_class": type(model).__name__,
        "model_selection": ("library default pretrained model (Cellpose-SAM family)" if major >= 4
                            else "model_type='nuclei'"),
        "pretrained_model": str(getattr(model, "pretrained_model", "unknown")),
        "input": "single DAPI channel, 2D, float32 (library normalisation applied inside eval)",
        "explicit_eval_settings": explicit,
        "all_eval_settings_used": {**defaults, **explicit},
        "post_processing": "none: labels are saved as returned (int32)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", nargs="?", help="JSON manifest (omit with --write-config)")
    ap.add_argument("--write-config", metavar="PATH", default=None,
                    help="Only write the run configuration JSON to PATH and exit.")
    ap.add_argument("--diameter", type=float, default=None,
                    help="Nucleus diameter in px. Leave unset for Cellpose-SAM (v4); "
                         "for v3 'nuclei' a value around 12-18 px fits 0.5 um/px data.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Lower this (4, 2, 1) if you run out of GPU memory.")
    ap.add_argument("--cpu", action="store_true", help="Allow running on CPU (very slow).")
    ap.add_argument("--overwrite", action="store_true", help="Recompute existing outputs.")
    args = ap.parse_args()

    import torch
    major, version_str = get_major_version()
    cuda_ok = torch.cuda.is_available()
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    use_gpu = cuda_ok or mps_ok

    if not use_gpu and not args.cpu:
        print("ERROR: no GPU visible to PyTorch (torch.cuda.is_available() is False).\n"
              "       Fix the CUDA install, or pass --cpu if you really want a (very slow) CPU run.")
        sys.exit(2)

    device = (torch.cuda.get_device_name(0) if cuda_ok else "mps") if use_gpu else "cpu"
    print(f"cellpose {version_str} | torch {torch.__version__} | device: {device}", flush=True)

    if args.write_config:
        model = build_model(major, use_gpu)
        cfg = collect_config(model, major, version_str, device, torch.__version__, args.diameter, args.batch_size)
        os.makedirs(os.path.dirname(os.path.abspath(args.write_config)), exist_ok=True)
        with open(args.write_config, "w") as fh:
            json.dump(cfg, fh, indent=1, default=str)
        print(f"configuration written to {args.write_config}", flush=True)
        return

    if not args.manifest:
        ap.error("a manifest is required unless --write-config is given")
    with open(args.manifest) as fh:
        manifest = json.load(fh)
    if not manifest:
        print("Empty manifest - nothing to do.")
        return

    model = build_model(major, use_gpu)

    out_dir = os.path.dirname(os.path.abspath(manifest[0]["output_path"]))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(collect_config(model, major, version_str, device, torch.__version__, args.diameter, args.batch_size),
                  fh, indent=1, default=str)
    timings_path = os.path.join(out_dir, "timings.json")
    if os.path.exists(timings_path):
        with open(timings_path) as fh:
            timings = json.load(fh)
    else:
        timings = {}
    timings.setdefault("seconds_per_image", {})
    timings.update({"cellpose_version": version_str, "device": device})

    batch_size = args.batch_size
    for i, item in enumerate(manifest, 1):
        out_path = item["output_path"]
        name = os.path.basename(out_path)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{i}/{len(manifest)}] {name}: exists, skipping", flush=True)
            continue

        image = load_channel(item["tif_path"], item["dapi_channel"]).astype(np.float32)

        t0 = time.time()
        while True:
            try:
                masks = segment(model, major, image, args.diameter, batch_size)
                break
            except RuntimeError as err:               # CUDA out-of-memory -> retry smaller
                if "out of memory" in str(err).lower() and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    print(f"    GPU out of memory, retrying with batch_size={batch_size}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                raise
        if cuda_ok:
            torch.cuda.synchronize()
        seconds = time.time() - t0

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.save(out_path, masks.astype(np.int32))
        timings["seconds_per_image"][name.replace(".npy", "")] = seconds
        with open(timings_path, "w") as fh:            # saved after every image: safe to interrupt
            json.dump(timings, fh, indent=1)
        print(f"[{i}/{len(manifest)}] {name}: {int(masks.max())} nuclei, {seconds:.1f}s", flush=True)
        if cuda_ok:
            torch.cuda.empty_cache()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

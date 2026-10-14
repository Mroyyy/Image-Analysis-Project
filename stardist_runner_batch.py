"""
Batch StarDist runner.

Runs inside the GPU-enabled StarDist conda environment. Takes a JSON manifest
listing every image to process, loads the pretrained model ONCE, then loops
over all images in a single process - this avoids paying the model-load cost
75 times (which would dominate total runtime even though inference itself is
fast).

Manifest format (JSON list of objects):
[
  {"tif_path": "./Data/.../Core[1,1,A]_....ome.tif", "dapi_channel": 2, "output_path": "./stardist_outputs/Core_1_1_A.npy"},
  ...
]

Usage (from the main notebook, via subprocess):
    python stardist_runner_batch.py manifest.json
"""

import sys
import json
import time
import numpy as np
from skimage import io as skio
from csbdeep.utils import normalize
from stardist.models import StarDist2D


def main(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Loading StarDist model (2D_versatile_fluo)...")
    t0 = time.time()
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    print(f"Model loaded in {time.time() - t0:.1f}s")

    n_ok, n_failed = 0, 0
    for i, item in enumerate(manifest):
        tif_path = item["tif_path"]
        dapi_channel = item["dapi_channel"]
        output_path = item["output_path"]

        print(f"[{i + 1}/{len(manifest)}] {tif_path}")
        t_img = time.time()
        try:
            img = skio.imread(tif_path)
            dapi = img[:, :, dapi_channel]
            labels, _ = model.predict_instances(normalize(dapi))
            np.save(output_path, labels)
            print(f"    -> {labels.max()} nuclei, {time.time() - t_img:.1f}s, saved to {output_path}")
            n_ok += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            n_failed += 1

    print(f"\nDone. {n_ok} succeeded, {n_failed} failed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stardist_runner_batch.py manifest.json")
        sys.exit(1)
    main(sys.argv[1])

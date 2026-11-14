"""Score images with an Anomalib OpenVINO export (pred_score, 0 = normal, 1 = anomalous, 0.5 = learned threshold).

Get a model:  viam packages export --org-id=<org> --name=sortit-patchcore --version=latest --type=ml_model --destination=/tmp/m
              (then untar; model.xml + model.bin + metadata.json)
Get frames:   viam data export binary filter --destination=/tmp/f --org-ids=<org> --part-id=<part> \
                --component-name=cam --mime-types=image/jpeg --tags=sortit-v1 --start=<iso time>
Run:          uv run python scripts/score_frames.py /tmp/m/model.xml /tmp/f/data/*.jpeg
"""
import sys

import cv2
import numpy as np
import openvino as ov


def main() -> None:
    model = ov.Core().compile_model(sys.argv[1], "CPU")
    scores = []
    for path in sys.argv[2:]:
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (256, 256)).astype(np.float32) / 255  # anomalib-train exports take [1,3,256,256] in 0..1
        s = float(np.asarray(model(x.transpose(2, 0, 1)[None])[model.output("pred_score")]).ravel()[0])
        scores.append(s)
        print(f"{s:.3f}  {path.rsplit('/', 1)[-1]}")
    if scores:
        print(f"n={len(scores)} mean={np.mean(scores):.3f} min={min(scores):.3f} max={max(scores):.3f}")


if __name__ == "__main__":
    main()

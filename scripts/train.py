"""Train an Anomalib model on *good* images only and export it to OpenVINO IR.

Dataset layout (only `good/` is required; `defect/` lets Anomalib calibrate the threshold properly):
    data/<item>/good/*.jpg
    data/<item>/defect/*.jpg

    uv run python scripts/train.py --data data/<item> --model patchcore
"""
from __future__ import annotations

import argparse
from pathlib import Path

from anomalib.data import Folder
from anomalib.deploy import CompressionType, ExportType
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Padim, Patchcore

MODELS = {
    "patchcore": lambda: Patchcore(),            # best accuracy with few images, no backprop
    "padim": lambda: Padim(),                    # fastest to train, tiny
    "efficientad": lambda: EfficientAd(),        # fastest inference, needs more images / epochs
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", default="patchcore", choices=MODELS)
    ap.add_argument("--out", default="models/anomaly", type=Path)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--compression", default="fp16", choices=["none", "fp16", "int8", "int8_ptq"])
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    abnormal = args.data / "defect"
    datamodule = Folder(
        name=args.data.name,
        root=args.data,
        normal_dir="good",
        abnormal_dir="defect" if abnormal.exists() else None,
        train_batch_size=1 if args.model == "efficientad" else 16,
        eval_batch_size=16,
        num_workers=args.workers,
        test_split_mode="from_dir" if abnormal.exists() else "synthetic",
        val_split_mode="same_as_test",
    )
    model = MODELS[args.model]()
    engine = Engine(max_epochs=args.epochs, default_root_dir=str(args.out / "_lightning"))
    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)

    compression = None if args.compression == "none" else CompressionType(args.compression)
    path = engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=args.out,
        compression_type=compression,
        datamodule=datamodule if compression in (CompressionType.INT8_PTQ,) else None,
    )
    print(f"\nOpenVINO model -> {path}")


if __name__ == "__main__":
    main()

"""Read / back up / patch a Viam machine part's config using the Viam CLI's login (no extra API key).

    uv run python tools/viam_config.py backup                       # writes runs/config-backups/<part>-<time>.json
    uv run python tools/viam_config.py patch-yolo-npu               # mlmodel-2 -> sortit-yolocls on viam:openvino NPU
    uv run python tools/viam_config.py restore <backup.json>        # put a backed-up config back

Uses optimistic concurrency: the update fails if someone else saved the config after we read it.
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
import time
from pathlib import Path

from grpclib.client import Channel
from viam.app.app_client import AppClient

ORG = "6d6c7293-bc67-43e4-9c8b-31c76aac27d3"
PART = "f08ae88f-5fa8-4775-a3bf-0f4fda96940f"
BACKUPS = Path("runs/config-backups")


def token() -> str:
    auth = json.loads((Path.home() / ".viam" / "cached_cli_config.json").read_text())["auth"]
    return auth["access_token"]


async def client() -> tuple[AppClient, Channel]:
    ch = Channel(host="app.viam.com", port=443, ssl=True)
    return AppClient(ch, {"authorization": f"Bearer {token()}"}), ch


async def read(app: AppClient):
    part = await app.get_robot_part(PART)
    return part, json.loads(part.robot_config_json) if hasattr(part, "robot_config_json") else part.robot_config


def backup(cfg: dict) -> Path:
    BACKUPS.mkdir(parents=True, exist_ok=True)
    p = BACKUPS / f"{PART[:8]}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(cfg, indent=2))
    return p


def patch_yolo_npu(cfg: dict) -> dict:
    new = copy.deepcopy(cfg)
    pkgs = new.setdefault("packages", [])
    if not any(p["name"] == "sortit-yolocls" for p in pkgs):
        pkgs.append({"name": "sortit-yolocls", "package": f"{ORG}/sortit-yolocls", "type": "ml_model", "version": "latest"})
    svc = next(s for s in new["services"] if s["name"] == "mlmodel-2")
    svc["model"] = "viam:openvino:mlmodel"
    svc["attributes"] = {
        "model_path": "${packages.ml_model.sortit-yolocls}/model.onnx",
        "label_path": "${packages.ml_model.sortit-yolocls}/labels.txt",
        "device": "NPU",
    }
    return new


def patch_yolo_preprocess(cfg: dict) -> dict:
    """Have OpenVINO take raw u8 RGB NHWC pixels and do layout + /255 itself (matches how the model was trained).

    No resize in the graph: the NPU needs static shapes, so the vision service resizes to 320x320 before calling.
    """
    new = patch_yolo_npu(cfg)
    svc = next(s for s in new["services"] if s["name"] == "mlmodel-2")
    svc["attributes"]["preprocess"] = {
        "input_name": "images",
        "tensor_element_type": "u8",
        "tensor_layout": "NHWC",
        "tensor_color_format": "RGB",
        "model_layout": "NCHW",
        "model_color_format": "RGB",
        "resize": "none",
        "mean": [0, 0, 0],
        "scale": [255, 255, 255],
    }
    return new


def add_objects_module(cfg: dict) -> dict:
    """Local module /root/viam_module (sortit:vision:objects) + an 'objects' vision service on ov + cam."""
    new = copy.deepcopy(cfg)
    mods = new.setdefault("modules", [])
    if not any(m.get("name") == "sortit" for m in mods):
        mods.append({"type": "local", "name": "sortit", "executable_path": "/root/viam_module/run.sh"})
    svcs = new.setdefault("services", [])
    if not any(s["name"] == "objects" for s in svcs):
        svcs.append({"name": "objects", "api": "rdk:service:vision", "model": "sortit:vision:objects",
                     "attributes": {"mlmodel_name": "ov", "camera_name": "cam"}})
    return new


def bump_objects_module(cfg: dict) -> dict:
    """Change the sortit module's env so viam-server restarts it (after copying new files to the NUC)."""
    new = copy.deepcopy(cfg)
    mod = next(m for m in new["modules"] if m.get("name") == "sortit")
    mod.setdefault("env", {})["SORTIT_REV"] = time.strftime("%H%M%S")
    return new


def pin_package(cfg: dict, name: str, version: str) -> dict:
    """Pin a top-level package (e.g. sortit-patchcore) to a version: 'latest', '1', '2', ..."""
    new = copy.deepcopy(cfg)
    pkg = next(p for p in new["packages"] if p["name"] == name)
    pkg["version"] = version
    for s in new.get("services", []):  # the app also mirrors the version inside the service's attributes
        if s.get("attributes", {}).get("name") == name and "version" in s["attributes"]:
            s["attributes"]["version"] = version
    return new


async def main() -> None:
    cmd = sys.argv[1]
    app, ch = await client()
    try:
        part, cfg = await read(app)
        print(f"backup -> {backup(cfg)}")
        if cmd == "backup":
            return
        if cmd == "patch-yolo-npu":
            new = patch_yolo_npu(cfg)
        elif cmd == "patch-yolo-preprocess":
            new = patch_yolo_preprocess(cfg)
        elif cmd == "add-objects-module":
            new = add_objects_module(cfg)
        elif cmd == "bump-objects-module":
            new = bump_objects_module(cfg)
        elif cmd == "pin":  # pin <package> <version>
            new = pin_package(cfg, sys.argv[2], sys.argv[3])
        elif cmd == "restore":
            new = json.loads(Path(sys.argv[2]).read_text())
        else:
            raise SystemExit(f"unknown command {cmd}")
        await app.update_robot_part(PART, part.name, robot_config=new, last_known_update=part.last_updated)
        _, after = await read(app)
        print("services:", [s["name"] for s in after["services"]])
        print("modules:", [m.get("name") for m in after.get("modules", [])])
    finally:
        ch.close()


if __name__ == "__main__":
    asyncio.run(main())

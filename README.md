# Spot it, sort it

Defect detection and response on an Intel NUC. An **Anomalib** model spots defective items. A **vision-language model** decides what to do with each one. A robot arm sorts good items into the box and defective ones into the reject bin. All models run through **OpenVINO**, split across NPU, GPU and CPU. See [docs/DESIGN.md](docs/DESIGN.md).

## Setup

```bash
# Python 3.12, managed by uv (https://docs.astral.sh/uv/)
uv sync
```

## Day-of workflow

```bash
# 1. Capture images into data/<item>/good (and a few in data/<item>/defect if possible)
# 2. Train and export to OpenVINO (a few minutes on CPU)
uv run python scripts/train.py --data data/<item> --model patchcore --compression fp16

# 3. (once, can be done ahead of time) export the VLM to OpenVINO INT4
uv sync --extra export && ./scripts/export_vlm.sh

# 4. Edit configs/default.yaml: decider.scenario, decider.actions, arm.driver

# 5. Run: live camera or a folder of images
uv run spotsort --source 0
uv run spotsort --source data/<item> --set detector.device=CPU

# 6. Benchmark the detector on each device for the write-up
uv run python scripts/benchmark.py
```

## Smoke test (no hardware)

```bash
uv run python scripts/make_synthetic.py
uv run python scripts/train.py --data data/synthetic_washer --model padim
uv run spotsort --source data/synthetic_washer
```

## Layout

| Path | What |
|-|-|
| `src/spot_it_sort_it/detector.py` | Anomalib OpenVINO inferencer (NPU) |
| `src/spot_it_sort_it/decider.py` | VLM decision via OpenVINO GenAI (GPU), with a rules fallback |
| `src/spot_it_sort_it/arm.py` | Arm interface. Add the real driver when the SDK is revealed |
| `src/spot_it_sort_it/pipeline.py` | Main loop and per-stage latency logging |
| `scripts/` | train/export, VLM export, benchmark, synthetic data |

## On the Viam machine (what actually runs for judging)

| Piece | Where | What |
|-|-|-|
| `ov` + `vision-1` | Viam registry modules | PatchCore (our `sortit-patchcore` model) on the iGPU, frame-level anomalous/normal |
| `objects` | `viam_module/` (local module `sortit:vision:objects`) | One box per part, each labelled normal / hold / anomalous; per-item log |
| `mlmodel-2` + `vision-2` | registry `viam:openvino:mlmodel` | YOLOv8n-cls (`sortit-yolocls`) on the NPU — measured, not used to decide |

`objects` DoCommands (call through the Viam SDK, CLI or MCP):

```
{"command": "new_run"}                start a fresh item log (right before the judged run)
{"command": "decide", "frames": 3}    score parts, LOG each new item, return new + pending items
{"command": "next"}                   oldest logged item not yet picked: centre pixel + destination bin
{"command": "arm_start", "item": N}   record first arm motion for item N (latency)
{"command": "log"}                    the run's log; the CSV lives on the NUC under the module data dir
```

Deploy changes to the module: `viam machines part cp --part=<part> -r -p viam_module machine:/root/`, then
`uv run python tools/viam_config.py bump-objects-module` to restart it.

## Tools
| Tool | What |
|-|-|
| `tools/capture_station.py` | LAN page: capture buttons + shared shot checklist + camera previews; tags frames and adds them to the dataset |
| `tools/viam_config.py` | Back up / patch / restore the machine config with the Viam CLI login (`backup`, `pin <pkg> <ver>`, `restore <file>`, …) |
| `scripts/score_frames.py` | Score images locally with a downloaded Anomalib OpenVINO export |

Design, measurements and what failed: [docs/DESIGN.md](docs/DESIGN.md).

## Licenses

Anomalib is Apache-2.0. Trained weights are not committed (models/ is git-ignored); the YOLO classifier was trained with Ultralytics (AGPL-3.0) and is kept out of this repo.

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

## Licenses

Anomalib is Apache-2.0. Check the license of the VLM you pick before you publish weights.

# Design: Spot it, sort it

## Pipeline

```
camera (CPU) ─▶ Anomalib detector (NPU) ─┬─ clear pass ───────────────────────▶ arm → box
                                          └─ anomalous / borderline ─▶ VLM (iGPU) ─▶ arm → box | reject
```

1. **Detect.** We train an Anomalib model (PatchCore, PaDiM or EfficientAD) on good items only, then export it to OpenVINO IR (FP16 or INT8). It returns an anomaly score and a heatmap for every frame.
2. **Decide.** The VLM runs only on items that are flagged or close to the threshold. It gets the frame, the heatmap overlay, the score and the event scenario. It returns JSON with `{action, defect, reason}`. If the output can't be parsed, the rules fallback takes over.
3. **Sort.** The arm puts `accept` items in the box and everything else in the reject bin.

## Compute split

| Device | Work | Why |
|-|-|-|
| NPU | Anomalib detector | Small static-shape CNN that runs on every frame. The NPU gives steady low latency at low power and leaves the GPU free. |
| iGPU | VLM (INT4, OpenVINO GenAI) | Generating text needs memory bandwidth and parallel compute, which the NPU doesn't handle well at this model size. It only runs on the small share of items that are flagged. |
| CPU | Capture, pre/post-processing, orchestration, arm control | Branchy work with low latency needs. It stays off the accelerators. |

Every stage falls back NPU → GPU → CPU (`devices.py`), so the same code runs on a dev laptop.

## Latency strategy

- Clean items skip the VLM, so most items pay only for detection.
- The detector is compiled with `PERFORMANCE_HINT=LATENCY`, and INT8 PTQ is available with `--compression int8_ptq`.
- The VLM is INT4 with a short `max_new_tokens` and a strict JSON output format.
- Every stage's time is logged to `runs/decisions.csv`. `scripts/benchmark.py` measures the detector on each device.

## Numbers (fill in on the NUC)

| Stage | Device | p50 ms |
|-|-|-|
| Detect | NPU | |
| Decide (flagged only) | GPU | |
| Arm move | — | |

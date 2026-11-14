# Design: Spot it, sort it

## What it does
A wrist camera looks down at the parts. Each part gets a box, an anomaly score and a decision: **box** (good),
**reject** (defective) or **hold** (not sure). Every decision is written to the per-item log *before* the arm moves.

```
cam (RealSense, wrist, at inspect-pose)
  └─▶ ov            PatchCore (Anomalib) on the iGPU, f32        ── one heatmap per frame
       ├─▶ vision-1 frame-level anomalous / normal               ── "is anything wrong in view?"
       └─▶ objects  (our module, sortit:vision:objects)
             find parts on the plain surface  (CPU, no training)
             score each part = heat on its pixels − heat on the empty surface
             label: < box_lo → normal/box · > reject_hi → anomalous/reject · else hold
             DoCommand decide → per-item log → arm / VLA picks by centre pixel
cam ─▶ mlmodel-2  YOLOv8n-cls on the NPU (vision-2)              ── measured NPU workload, not used to decide
```

## Why each choice

| Choice | Why | Evidence |
|-|-|-|
| PatchCore (unsupervised anomaly detection) as the decider | Learns "normal" from good parts only, so it flags defect types it has never seen — the brief guarantees at least one | Blind test on unseen shots: caught 6/7 defective shots incl. three foreign blocks; supervised YOLO caught 2/7 |
| PatchCore on the **iGPU at f32** | Fastest device for it; f16 zeroes the anomaly map (no heatmap, no boxes) | ~23–31 ms GPU vs ~75 ms CPU, NPU much slower (organiser + module benchmarks) |
| YOLO classifier on the **NPU** | Small static-shape CNN suits the NPU; frees the GPU | Compiles to `execution_devices ['NPU']` in ~0.3–0.7 s; preprocessing compiled into the graph (u8 → f32, NHWC → NCHW) |
| YOLO **not** used for decisions | Learned a shortcut ("small dark thing = defect"): flagged good small bolts, missed long blue-patch bolts and all blocks | Blind test, shots 18–28 |
| Per-object boxes by surface subtraction | Parts sit on plain cardboard, so no detector training is needed; works the moment a new part type appears | One box per part on every test frame incl. 4 parts at once |
| Score relative to the frame's own background | Cancels global shifts (lighting, cardboard moved) that otherwise push every part over the line | Background median rose from ~0.25 to ~0.41 under evening light |
| Three destinations with a **hold** band | Uncertain parts go to hold, not the wrong bin — protects "items ending where we decided" | Closest held-out calls (good small bolt 0.49, bad small bolt 0.44) both land in hold |
| Log written by the NUC module before the arm moves | The log is ~40% of the score; writing it where the model runs means no network hop can lose it | `decide` logs + returns in ~480 ms (3-frame average) |
| CPU | Camera handling, part finding, logging, arm control | — |

## What failed (and what we learned)
- **A model trained on another station's camera scored 1.00 on everything here.** Anomaly models learn the whole
  scene, not the part. Normal must come from *our* camera, pose, surface and gripper state.
- **v0, trained on good frames only (16), was degenerate** (AUROC 0.5, every score clipped to 1.0): the threshold is
  calibrated on real defects, so a few defective shots are required.
- **Mixed backgrounds and gripper states** (white table vs cardboard, gripper open vs closed) taught the model the
  wrong cue; we standardised on cardboard + open gripper and dropped the mismatched frames.
- **Evening light tripled colour saturation** and collapsed the per-part gap. Fix is physical: a fixed lamp, then a
  short on-site "teach-in" capture and retrain before the run.
- **vision-2 stuck on "normal" after moving to OpenVINO**: the module feeds tensors as-is; fixed by compiling the
  preprocessing into the model.
- **Our capture page's camera previews flooded the NUC with connections** (340 per 10 min) and made it look offline;
  fixed with shared, throttled frames.

## What we'd change
- Fixed lighting enclosure + locked camera exposure/white balance, so one model holds across the day.
- Many more defect types and good-part poses (our biggest accuracy lever), captured with the capture page.
- Crop tightly around each part and score crops, so small defects fill more of the input.
- Use depth (RealSense) for separating touching parts and for 3D grasp points.

## Numbers to fill in on the day
| Stage | Device | Measured |
|-|-|-|
| PatchCore inference | GPU f32 | `ov-diag` benchmark |
| YOLO classifier | NPU | `ov-diag` benchmark (point a diagnostics service at `mlmodel-2`) |
| `objects` decide (3 frames, incl. camera) | GPU + CPU | ~480 ms |
| Item enters view → first arm motion | — | `latency_ms` column of the item log |

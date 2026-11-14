# Handoff — Track 3 "Spot it, sort it" (updated 2026-11-13 ~20:45 local)

Public copy of the local handoff (team access code removed). Judging Sat 3:30 PM. Brief: `~/Downloads/Voltron brief - Track 3 Team Brief Spot it, sort it.pdf`.
Design write-up (public): `docs/DESIGN.md`. Commands: `README.md`.

## IDs
| What | Value |
|-|-|
| Org `hackathons` | `6d6c7293-bc67-43e4-9c8b-31c76aac27d3` |
| armfarm01 main part | `f08ae88f-5fa8-4775-a3bf-0f4fda96940f` |
| Dataset `sortit-armfarm01-v1` | `6af772edf64790cdeaa8ee06` — 129 frames on cardboard (83 normal / 46 anomalous); white-table shots 03/04 removed |
| Anomaly training | script `hackathons:anomalib-train` 0.4.0, container `9accd6e7-2992-47c9-9662-2ca5298fa1f1` |
| Models in registry | `sortit-patchcore` v1 (61/12 frames), v2 (83/46); `sortit-yolocls` v1 (ONNX) |

Viam CLI logged in on this laptop (token expires Sat 11:29 EST → `viam login`). `tools/viam_config.py` reuses that login.

## Live on armfarm01 right now
- `ov` → `sortit-patchcore` **pinned to v1** (team preferred it), GPU, f32. Revert/advance: `uv run python tools/viam_config.py pin sortit-patchcore <1|2|latest>`.
- `vision-1` → `ov`, cam, threshold 0.5 (frame-level).
- `objects` (local module `sortit`, files in `/root/viam_module` on the NUC, env built with uv) → `ov` + cam,
  `box_lo 0.30`, `reject_hi 0.50` (calibrated for v1's relative scores on unseen shots; v2 would need ~0.15/0.28).
  DoCommands: `objects`, `decide`, `next`, `arm_start`, `log`, `new_run`. Log CSV on the NUC:
  `/root/.viam/module-data/<part>/sortit/items-<run>.csv`.
- `mlmodel-2` → `sortit-yolocls` on **NPU** with compiled preprocessing (u8 NHWC); `vision-2` uses it. Not used for decisions.
- Config backups before every edit: `runs/config-backups/` (restore with `tools/viam_config.py restore <file>`).
- Capture page: `TEAM_CODE=<team code> uv run python tools/capture_station.py` → http://<laptop-ip>:8899/?code=<team code> (state in `runs/capture_state.json`).

## Key findings
- **Don't sort from `vision-1`.** Its "normal NN%" is 1 − frame score; under evening light it flickers across 0.5
  (0.38 → 0.60 on the same scene). Use `objects` → `decide` (per part, 3-frame average, hold band) instead.
- Per-object relative scores separate cleanly on frames matching the training setup; they collapse when the scene
  changes (cardboard moved/creased, evening light: saturation ~37 → ~105). Pipeline itself verified: machine output == local v2.
- YOLO-cls learned a shortcut; blind test 2/7 defects. PatchCore 6/7.
- Gripper state and background must match between capture and run (open gripper, cardboard).

## Tomorrow (before 3:30)
1. Fixed lamp over the inspection area, cardboard taped to fill the cam view, no marks. Same gripper state.
2. Teach-in: ~6 good + ~4 defective arrangements on the capture page (fresh dataset or add to existing).
3. Retrain (bump `--model-version`), pin `ov` to it, recalibrate `box_lo`/`reject_hi` on held-out shots
   (`/tmp`-style eval scripts are gone; re-run leave-one-shot-out: see `docs/DESIGN.md` numbers).
4. Wire the arm: `decide` → `next` → move to centre pixel (needs pixel→3D via depth + frame system, or VLA) → `arm_start` → drop at box/reject/hold pose (poses not saved yet).
5. `new_run` right before judging; pull the CSV afterwards (`viam machines part cp --part=<part> machine:<log path> .`).
6. `ov-diag` benchmarks for GPU and NPU models → fill the numbers table in `docs/DESIGN.md`.

## Not done
- Saved poses for box / reject / hold; gripper timing; pixel→grasp-point mapping; arm integration with `decide`.
- Separating touching parts (they merge into one box).

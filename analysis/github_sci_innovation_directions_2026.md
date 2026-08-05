# GitHub / Recent SCI Innovation Directions for Aircraft Skin Defect Detection

## Context

Current local evidence:

- Best training anchor: `N2`, YOLO11n, imgsz 800, no mosaic/erasing, mAP50-95 `0.352565`.
- Best inference setting: `N2 + TTA + iou=0.685`, mAP50-95 `0.356407`.
- Already weak/negative locally: simple P2/ECA variants, sliced inference only, synthetic patch copy-paste, Focaler-IoU, dataset replacement with clean/rebalanced variants.

Therefore, the next SCI-oriented innovation should not be a single attention plug-in or threshold trick. It should combine aircraft-defect domain priors with a structurally defensible small-object detector.

## Ranked Directions

### 1. Detail-Preserving Downsampling + Defect P2 Head

Inspired by YOLO-TLP and tiny-object work.

Core idea:

- Replace early stride-2 downsampling with Space-to-Depth / SPDConv-style downsampling.
- Keep a lightweight P2 detection head, but make it defect-aware instead of simply adding another generic YOLO head.
- Use this branch mainly for `missing-rivet` and thin `scratch` cases where spatial pixels vanish quickly.

Why it fits this project:

- Aircraft defects are tiny, weak-texture, and often lose discriminative cues after early downsampling.
- You already tried P2-like routes, but not a complete spatial-information-preserving stem plus defect-specific tiny head.
- This can be written as a real method: "detail-preserving feature extraction for aircraft skin micro-defects".

Risk:

- P2 alone can overfit or increase false positives.
- Must include ablations: baseline, P2 only, SPD only, SPD+P2, SPD+P2+TTA.

Priority: highest.

### 2. Bidirectional Dynamic Multi-Scale Fusion Neck

Inspired by BDMA-YOLO, LEAF-YOLO, PARE-YOLO, MFDA-YOLO, and multi-scale UAV detectors.

Core idea:

- Replace static concat in the YOLO neck with learnable bidirectional weighted fusion.
- Add a lightweight spatial/channel gate that preserves local defect texture instead of global-pooling-heavy attention.
- Keep the model small; do not simply stack CBAM/ECA.

Why it fits this project:

- Your defects vary from tiny missing rivets to elongated scratches/rust, so one fixed fusion path may under-serve some classes.
- Unlike pure TTA, this is a structural contribution and can support SCI-style ablation.

Risk:

- If the dataset is small, too many fusion parameters may overfit.
- Start with one neck block, not a full architecture rewrite.

Priority: high.

### 3. Dual-Branch Defect Morphology Network

Inspired by recent industrial defect detection papers using dual-backbone/dual-branch strategies.

Core idea:

- Branch A: normal YOLO feature path for global panel context.
- Branch B: shallow high-resolution texture/morphology branch for thin scratches, missing rivets, rust edges.
- Fuse before the head using a small gate.

Why it fits this project:

- Aircraft skin defects are not generic objects; they are local texture anomalies on a smooth surface.
- This gives the paper a domain-specific story rather than "YOLO + module".

Risk:

- More engineering work.
- Needs careful visualization: Grad-CAM/feature maps showing Branch B attends to defect edges.

Priority: high for SCI narrative, medium for quick experiments.

### 4. Hypergraph / High-Order Feature Correlation

Inspired by YOLOv13 and Hyper-YOLO.

Core idea:

- Use a compact high-order correlation module in the neck to model relationships between repeated rivets, panel texture, scratches, and background.
- Do not transplant full YOLOv13 unless compute and implementation time allow.

Why it fits this project:

- Aircraft surfaces have structured repeated patterns; high-order context could suppress rivet/texture false positives.

Risk:

- Complex, harder to debug, and more likely to overfit small data.
- Better as a later experiment or discussion baseline, not the first implementation.

Priority: medium.

### 5. Defect-Aware Localization Loss

Core idea:

- Instead of generic NWD/Focaler-IoU, design a morphology-aware localization term:
  - thin defects: penalize centerline/long-axis mismatch
  - round missing rivets: emphasize small-box center stability
  - rust: tolerate diffuse boundaries more than scratches

Why it fits this project:

- This is domain-specific and more novel than importing another IoU variant.
- It can be paired with the existing class-specific AP analysis.

Risk:

- Harder to implement correctly.
- If labels are noisy, a stricter morphology loss may hurt.

Priority: medium-high if the goal is SCI novelty, lower if the goal is fast mAP.

## Recommended Paper Mainline

Name candidate: `DPM-YOLO` / `AeroDefect-YOLO`.

Proposed contributions:

1. Detail-preserving stem using SPDConv-style downsampling for micro-defect cues.
2. Defect-aware high-resolution P2 branch for missing-rivet and scratch localization.
3. Lightweight dynamic multi-scale fusion neck for balancing local texture and global panel context.
4. TTA + tuned NMS as deployment-stage enhancement, not the main innovation.

Minimum ablation matrix:

| ID | Variant | Purpose |
| --- | --- | --- |
| A0 | N2 baseline | Strong local baseline |
| A1 | SPD stem only | Tests detail preservation |
| A2 | P2 defect head only | Confirms simple P2 is insufficient/sufficient |
| A3 | SPD + defect P2 | Main tiny-defect branch |
| A4 | SPD + defect P2 + dynamic fusion | Full model |
| A5 | A4 + TTA/iou=0.685 | Deployment mode |

## Not Recommended as Main SCI Innovation

- TTA/NMS tuning alone: useful but too close to engineering inference optimization.
- Generic CBAM/ECA insertion: too common, weak novelty.
- Simple SAHI/sliced inference: already locally weak and well-known.
- Blindly switching to YOLOv13: publishable only if adapted with a clear aircraft-defect mechanism and strong ablations.
- More synthetic copy-paste: existing local results suggest distribution shift risk.

## Sources Checked

- GitHub YOLO-TLP: spatial-information-preserving small-object detector with SPDConv and P2/P3/P4 heads.
- GitHub YOLOv13: HyperACE and FullPAD high-order feature fusion.
- GitHub LEAF-YOLO: lightweight edge-real-time aerial small-object detector.
- GitHub le-yolo: lightweight enhanced YOLOv8 for UAV aerial small objects.
- BDMA-YOLO, The Visual Computer 2026: bidirectional lightweight fusion and partial attention fusion for UAV small targets.
- FMV-YOLO, Electronics 2025: YOLOv11-based steel surface defect detection with fine-grained channel attention, multi-scale attention fusion, and NWD loss.

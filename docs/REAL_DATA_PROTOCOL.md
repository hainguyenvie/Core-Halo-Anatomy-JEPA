# Real-data validation protocol

This document separates a fast mechanism PoC from a defensible paper benchmark. Freeze the protocol before inspecting test masks.

## Phase 0 — executable sanity check

Run the deterministic falsification matrix in
[`FALSIFICATION_RUN.md`](FALSIFICATION_RUN.md). Continue with each component only
if its own three-seed decision is supported. A full-model win is not evidence for
the halo if `halo_h1` does not beat `ctx_r5`, and it is not evidence for anatomy
unless the correct mirror also beats both matched anatomy controls.

## Phase 1 — low-friction real-data PoC

Use one normal/lesion modality pair first:

1. **IXI T1 healthy controls → ATLAS v2 stroke lesions.** This is the cleanest first test of small unilateral lesions and contralateral context.
2. **IXI T2 healthy controls → BraTS21 tumor slices.** This stresses mass effect and weaker symmetry; it is a useful negative/control domain rather than an expected easy win.

Use the patient-level CSV converter in this repository. Do not randomly split slices. Select hyperparameters using healthy latent validation loss and the frozen synthetic geometry ablation; do not select them using test Dice.

## Phase 2 — published baseline comparison

### Main task/base implementation

- Patch2Loc paper: <https://arxiv.org/abs/2506.22504>
- Official implementation: <https://github.com/bakerhassan/Patch2Loc>

Use the official patient CSV splits and preprocessing where licensing permits. Run the official code and this repository from the same preprocessed arrays. A local coordinate-regression reimplementation must be labeled `Patch2Loc-like`, never reported as the official baseline.

### Benchmark harness/checkpoints

- BMAD benchmark: <https://github.com/DorisBao/BMAD>
- Conditioned diffusion UAD: <https://github.com/FinnBehrendt/Conditioned-Diffusion-Models-UAD>
- MAD-AD: <https://github.com/farzad-bz/MAD-AD>
- I-JEPA reference implementation: <https://github.com/facebookresearch/ijepa>

BMAD is useful to validate metric code against released checkpoints. It is not a substitute for the Patch2Loc split protocol. I-JEPA is the representation-learning mechanism reference; reproducing its large ImageNet configurations is not required for this MRI PoC.

## Frozen comparison matrix

Run every row with seeds 0, 1, and 2 and identical train/calibration/test subjects:

| Family | Required row |
|---|---|
| Published base | official Patch2Loc |
| Strong reconstruction baseline | official cDDPM or released BMAD checkpoint |
| Generic representation baseline | `local_context` |
| Scale ablation | `wide_context` |
| Leakage ablation | `core_halo` |
| Proposed | `core_halo_anatomy` |

Report mean ± standard deviation. Primary endpoints are pixel AUPRC and calibrated Dice; pixel AUROC, oracle Dice, image AUROC, healthy FPR, wall-clock inference, peak memory, and lesion-size strata are secondary.

## Required falsification tests

1. **Fixed target, context sweep:** radii 2, 3, 5, 7.
2. **Fixed context, halo sweep:** halo widths 0, 1, 2.
3. **Anatomy rule:** no mirror, correct mirror, a token-count-matched non-homologous same-subject region, and another subject's mirror.
4. **Core-size sweep:** 1×1, 2×2, 4×4 tokens to expose JEPA's small-target failure mode.
5. **Lesion size:** small/medium/large with bootstrap confidence intervals at patient level.
6. **Registration stress:** controlled left–right shifts/rotations on healthy scans.
7. **Bilateral pathology:** report separately; do not hide it in the aggregate.
8. **Site leakage:** stratify healthy FPR by scanner/site when metadata are available.

The random and cross-subject mirror controls are essential. If they match the correct homologous mirror, improvement comes from extra tokens or site/style information, not anatomy.

## Go/no-go rule for a paper

Continue to a full paper only if the proposed model:

- beats the official Patch2Loc result after re-running both methods on at least two lesion datasets with the same protocol;
- improves pixel AUPRC and calibrated Dice, not only AUROC;
- has a reproducible small-lesion gain across seeds;
- beats `wide_context` and `core_halo`, demonstrating that both halo and anatomy contribute;
- shows correct mirror > random/cross-subject mirror;
- does not increase healthy FPR by more than one absolute percentage point;
- has a credible speed or memory advantage over the selected diffusion baseline.

If only the wide-context row wins, the publishable conclusion is scale decoupling, not anatomy awareness. If only the anatomy row wins on ATLAS but fails under small registration perturbations, the method needs alignment uncertainty rather than a stronger novelty claim.

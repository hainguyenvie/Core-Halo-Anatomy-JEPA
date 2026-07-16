# Core–Halo Anatomy JEPA

A falsifiable proof-of-concept for **unsupervised brain MRI anomaly localization**. The method predicts a small target core in latent space from a larger, target-free context. A masked halo prevents boundary leakage, and an optional left–right homologous region supplies an explicit anatomy rule.

This repository is a research prototype. The synthetic benchmark tests mechanism only; it is not evidence of clinical performance.

## Research question

Patch-localization methods face a coupled scale choice: a small patch gives precise localization but is anatomically ambiguous, while a large patch supplies location context but can dilute small lesions or learn shortcuts. The pre-registered hypothesis here is:

> Holding the target core fixed, larger target-free context should make healthy latent targets more predictable; a halo should reduce local leakage; and contralateral homologous context should improve localization when anatomy is approximately symmetric.

The legacy four-row matrix tests each clause separately:

| Experiment | Core | Context radius | Halo | Contralateral rule | Isolates |
|---|---:|---:|---:|---:|---|
| `local_context` | 2×2 tokens | 2 | 0 | No | coupled/local reference |
| `wide_context` | 2×2 tokens | 5 | 0 | No | context-scale effect |
| `core_halo` | 2×2 tokens | 5 | 1 | No | halo effect |
| `core_halo_anatomy` | 2×2 tokens | 5 | 1 | Yes | anatomy-rule effect |

All variants use the same context encoder, EMA target encoder, predictor, training data, optimizer, and anomaly calibration. Only the context geometry changes.

```text
visible local context ─┐
                       ├─> context encoder ─> latent predictor ─> predicted core ẑ
mirrored hemisphere ───┘                                      │
                                                              │ residual
full healthy target ─────> EMA target encoder ────────────────>│ z

core = prediction target; halo = absent from every context token sequence
```

At inference, the feature residual is calibrated with coordinate-wise diagonal Mahalanobis statistics estimated only from healthy calibration slices. Overlapping core scores are averaged into a dense anomaly map.

## Quick start

Use Python 3.10+ and install the appropriate PyTorch build for your CPU/CUDA system first. Then:

```bash
git clone https://github.com/hainguyenvie/Core-Halo-Anatomy-JEPA.git
cd Core-Halo-Anatomy-JEPA
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,medical]"
pytest
```

Run a tiny end-to-end smoke experiment (it checks plumbing, not the hypothesis):

```bash
python scripts/run_falsification.py \
  --suite all \
  --smoke \
  --seeds 0 \
  --output-root outputs/falsification_smoke
```

Run the actual synthetic falsification study with three seeds:

```bash
python scripts/run_falsification.py \
  --suite all \
  --seeds 0 1 2 \
  --output-root outputs/falsification
```

This runs 10 unique variants: five context radii, two nonzero halo widths, the
correct within-image mirror, a token-count-matched non-homologous control, and a
mirror-coordinate donor from another synthetic subject. The aggregate table,
paired bootstrap intervals, and joint component decisions are written to:

```text
outputs/falsification/falsification_summary.md
outputs/falsification/falsification_summary.json
```

Runs resume completed, configuration-compatible variants by default. Use `--rerun`
only when you intentionally want to overwrite a run. One- or two-seed runs are
always labeled `insufficient_seeds` and must not be interpreted as evidence.

See [docs/FALSIFICATION_RUN.md](docs/FALSIFICATION_RUN.md) for staged commands,
expected outputs, decision rules, and the lightweight result package to commit.

The original four-row command remains available for backward compatibility:

```bash
python scripts/run_synthetic_poc.py --seeds 0 1 2
```

## Train and evaluate one configuration

```bash
core-halo-jepa train \
  --config configs/synthetic/core_halo_anatomy.yaml \
  --set train.epochs=20 \
  --set seed=0

core-halo-jepa evaluate \
  --config configs/synthetic/core_halo_anatomy.yaml \
  --checkpoint outputs/core_halo_anatomy/best.pt
```

Outputs include the resolved configuration, training history, best/last checkpoint,
healthy residual calibrator, aggregate and per-image metrics, a compressed sample,
and `qualitative_grid.png`. Data, model weights, and outputs are ignored by Git.

## Real MRI data contract

The training split must contain healthy controls only. The calibration split must also be healthy and is used to estimate residual statistics plus a fixed 99.5th-percentile detection threshold. Lesion masks are used only in the test split for metrics.

Create a patient-level `volumes.csv`:

```csv
image,mask,split,patient_id
/data/IXI/IXI001_T1.nii.gz,,train,IXI001
/data/IXI/IXI002_T1.nii.gz,,calibration,IXI002
/data/ATLAS/sub-r001_T1.nii.gz,/data/ATLAS/sub-r001_mask.nii.gz,test,ATLAS001
```

Convert canonical NIfTI volumes to portable 2-D arrays:

```bash
python scripts/prepare_nifti.py \
  --volumes-csv data/volumes.csv \
  --output-dir data/slices \
  --axis 2 \
  --image-size 240
```

Then edit `configs/real/core_halo_anatomy.yaml` and run the same train/evaluate commands. The converter rejects patient IDs assigned to multiple splits and rejects image/mask grid mismatches. Registration, skull stripping, modality matching, and dataset licenses remain the researcher's responsibility.

See [docs/REAL_DATA_PROTOCOL.md](docs/REAL_DATA_PROTOCOL.md) for the proposed Patch2Loc/BMAD comparison and go/no-go criteria.

## Metrics

Primary metrics are pixel AUPRC and pixel AUROC inside the nonzero brain region. The repository also reports:

- Dice at a threshold fixed from healthy calibration data (primary Dice);
- oracle/test-set Dice only as a clearly labeled secondary diagnostic;
- image AUROC, healthy pixel false-positive rate, and Dice by lesion size;
- paired per-image bootstrap confidence intervals for every causal comparison;
- latent variance diagnostics to reveal representation collapse.

## What this PoC does and does not establish

It directly tests whether decoupling target and context scales helps and whether improvement comes from context size, the halo, or the anatomy rule. It does **not** yet establish superiority to the official Patch2Loc, cDDPM, MAD-AD, or BMAD implementations. Those must be re-run on identical patient-level splits and preprocessing; cross-paper table numbers are not a valid comparison.

Known risks are deliberate and testable:

- a 2×2-token target may be too small for semantic JEPA learning;
- contralateral context assumes consistent left–right orientation and may fail for bilateral disease or major mass effect;
- an excessively wide context may introduce scanner/site shortcuts;
- diagonal residual calibration ignores cross-feature covariance;
- a 2-D model cannot exploit through-plane anatomy.

## Reproducibility

- Every checkpoint stores the fully resolved configuration.
- Synthetic samples are deterministic by `(split, seed, index)`.
- Thresholding uses healthy calibration data, not test masks.
- The CI workflow runs leakage tests, shape/EMA tests, calibration tests, and a CPU end-to-end smoke experiment.
- The repository never redistributes medical images or third-party checkpoints.

## License

MIT. Dataset and third-party baseline licenses apply separately.

# Falsification run guide

This is the recommended synthetic experiment for deciding whether the paper can
claim wide context, a halo, and/or an anatomical mirror. It is designed to remove
a component when its control test fails, rather than tune until every component wins.

## 1. Install and verify

Install the PyTorch build appropriate for the machine first, then install this
package and run the unit tests:

```bash
git switch main
git pull origin main
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,medical]"
pytest
```

## 2. Plumbing smoke test

```bash
python scripts/run_falsification.py \
  --suite all \
  --smoke \
  --seeds 0 \
  --bootstrap-samples 200 \
  --output-root outputs/falsification_smoke
```

Expected behavior:

- all 10 variants train, calibrate, and evaluate without target leakage;
- each run creates quantitative, per-image, and qualitative outputs;
- the root creates `falsification_summary.json` and `.md`;
- every comparison is labeled `insufficient_seeds`, because smoke is not a
  hypothesis test.

## 3. Full three-seed test

```bash
python scripts/run_falsification.py \
  --suite all \
  --seeds 0 1 2 \
  --bootstrap-samples 5000 \
  --output-root outputs/falsification
```

This is 30 train/evaluate runs. Runtime depends strongly on the GPU. Completed
runs are reused only if their `resolved_config.yaml` exactly matches the requested
configuration, so an interrupted command can be run again safely. Use a different
output root for smoke and full runs. `--rerun` intentionally ignores completed
artifacts and overwrites them.

For staged execution, use the same three seeds and separate roots:

```bash
python scripts/run_falsification.py --suite context --seeds 0 1 2 \
  --output-root outputs/falsification_context
python scripts/run_falsification.py --suite halo --seeds 0 1 2 \
  --output-root outputs/falsification_halo
python scripts/run_falsification.py --suite anatomy --seeds 0 1 2 \
  --output-root outputs/falsification_anatomy
```

## 4. What is being compared

| Component | Proposed comparison | Negative controls |
|---|---|---|
| Context scale | `ctx_r5 > ctx_r2` | radii 1, 3, and 7 expose a non-monotonic optimum |
| Halo | `halo_h1 > ctx_r5` | `halo_h2` tests sensitivity to a wider exclusion ring |
| Anatomy | `anat_mirror > halo_h1` | `anat_random` and `anat_cross_subject` |

`anat_random` adds the same number of explicitly typed tokens from deterministic,
non-homologous coordinates in the same image. `anat_cross_subject` uses the correct
mirror coordinates but takes their intensities from another independently generated
synthetic subject. Thus, correct-mirror superiority cannot be explained only by
extra tokens, token type, or generic anatomy at the same coordinates.

## 5. Decision rule

For one comparison, `supported` requires all three conditions:

1. mean seed-level pixel-AUPRC gain is at least `0.001`;
2. the proposed side wins in at least two of three seeds;
3. the 95% paired per-image bootstrap interval has a lower bound above zero.

The joint anatomy decision is `supported` only if `anat_mirror` beats all three of
`halo_h1`, `anat_random`, and `anat_cross_subject` under that rule. A one-seed smoke
run can never satisfy the decision rule.

The previous four-row result suggests that `ctx_r5` may beat `ctx_r2`. It did not
provide convincing evidence for the halo or mirror anatomy. Therefore the honest
expected outcome is:

- wide context is the most likely component to survive;
- halo and anatomy may remain `negligible` or `inconclusive`;
- if the joint anatomy decision fails, remove the anatomy novelty claim and frame
  the next study around scale-decoupled JEPA context;
- retain the anatomy claim only if the correct mirror defeats both matched controls.

These are expectations, not target numbers. The generated verdicts determine the
next model, including a negative result.

## 6. Expected files

Each `<variant>/seed_<n>/` directory contains:

```text
best.pt
last.pt
calibrator.pt
metrics.json
per_image_metrics.json
resolved_config.yaml
train_history.json
training_summary.json
examples.npz
qualitative_grid.png
```

The root contains `falsification_summary.json` for machine review and
`falsification_summary.md` for a compact table. Inspect `qualitative_grid.png` for
edge/background shortcuts even when a quantitative verdict is positive.

## 7. Package only reviewable outputs

Do not commit checkpoints or compressed raw arrays. Create a small Git-friendly
copy instead:

```bash
python scripts/package_results.py \
  --input outputs/falsification \
  --output results/falsification_run1
git add results/falsification_run1
git commit -m "Add full falsification results"
git push origin main
```

The package includes summaries, resolved configurations, histories, per-image
metrics, and PNG grids. It excludes `.pt` and `.npz` files. Choose a new output
directory for each result package so existing review files are never silently
deleted or mixed.

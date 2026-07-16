# Core-Halo Anatomy JEPA PoC results

| Experiment | Pixel AUROC | Pixel AUPRC | Calibrated Dice | Small-lesion Dice | Healthy FPR |
|---|---:|---:|---:|---:|---:|
| local_context | 0.9865 | 0.8301 | 0.7292 | 0.5065 | 0.0061 |
| wide_context | 0.9879 | 0.8384 | 0.7325 | 0.5153 | 0.0063 |
| core_halo | 0.9879 | 0.8390 | 0.7323 | 0.5152 | 0.0063 |
| core_halo_anatomy | 0.9878 | 0.8388 | 0.7324 | 0.5167 | 0.0063 |

## Pre-registered directional checks

- [x] Wider target-free context improves pixel AUPRC over local context
- [x] Adding a halo improves pixel AUPRC over the same wide context
- [ ] Contralateral anatomy improves pixel AUPRC over Core-Halo alone
- [x] Full model improves small-lesion Dice over local context

A failed check is informative: it falsifies that component under this setup and should not be hidden.

# Issue #28 final evaluation: portable publication provenance

This is a documentation summary transcribed from the existing local
`preflight.json` and public [`reveal/manifest.json`](reveal/manifest.json).
It is not an execution manifest, a new evaluation, or an independent reproduction.
No model was fitted, predictions regenerated, or metrics recomputed to prepare it.
The existing reports and operational markers are preserved unchanged.

The original preflight contains absolute local paths in `artifacts_dir`,
`reports_dir`, and `execution.inputs.*.path`. It remains local and is omitted from
the publication set. This separate summary uses repository-relative references
and selected recorded provenance; the original retains the full source-hash map.

## Recorded outcome and execution

| Recorded field | Value |
|---|---|
| Source commit | `fa201bc741dc448b3d3d62dfc8bf7d948c4bbcd5` |
| Source dirty | `true` (as recorded; the commit alone does not describe the evaluated working tree) |
| Preflight status | `passed_blind_structural_preflight` |
| Preflight fits / realized trees | 1 / 1,743 |
| Training-label dates | 2015-10-19 through 2016-01-03 |
| Training rows / unique training series | 670,292 / 2,996 |
| Evaluation series | 2,996, fixed across all four methods and windows |
| Evaluation origins | 2016-01-03, 2016-01-31, 2016-02-28, 2016-03-27 |
| Forecast horizon | 28 days per origin |
| Daily predictions per method / total | 335,552 / 1,342,208 |
| Reveal `accuracy_calculations` | 1 (one error-metric scoring pass) |
| Reveal `retrained` | `false` |
| Preflight exceptions / failures; reveal exceptions | `[]` / `[]`; `[]` |
| Independent reproduction | `not performed` |

The full frozen protocol is retained in the reveal manifest. WAPE measures error,
not accuracy. The four windows had already been exposed through Phase 1 baseline
evaluation. This final run does not establish statistical significance, universal
model superiority, production readiness, or inventory benefit. See the
[README final results](../../../README.md#completed-final-results) for the mixed
ML/MA28 outcome and the planned Phase 3 comparison under identical assumptions.

Recorded software: Python 3.13.1, LightGBM 4.7.0, NumPy 2.5.3, Pandas 3.0.6,
PyArrow 26.0.0, scikit-learn 1.9.1, SciPy 1.18.1; macOS ARM64. Preflight runtime
was 151.11900650011376 seconds with 2,527.03 MiB peak process RSS; reveal runtime
was 4.470825041178614 seconds with 1,179.55 MiB peak process RSS. RSS is the
whole-process cumulative high-water mark. These describe the original local
execution and are not portable performance guarantees.

## Recorded identity hashes

The original preflight's byte hash was checked against the saved reveal manifest
while preparing this documentation. The other identities below are copied from
the saved JSON; private prediction/model/data files were not opened to recheck
their hashes.

| Identity | Recorded SHA-256 |
|---|---|
| Original `preflight.json` | `1c3bafd437e173042650065c20b4054d33163d97372413c21338e5c993783371` |
| Cohort | `261f89bd6b5a911733b065afc4b1daff16702bab89084bd67f45a79236020c69` |
| Aligned observations | `d81e3ca10f8f013ca50249fde08ae9260a92cd34b28a8e8d54529cd04572353b` |
| Predictions | `5b5317ef50bf2fcde8e635ac4380cf4f12d164c37b21e409d714cb2254ee0211` |
| Grouping lookup | `3329b90439b40559ca98c3556ad6b8ece16cb4519ffc2e1a0f3aeecdc6a434b0` |
| Model vocabulary/schema | `4e696538479a06a79dc1499f6be22d1625570b6e3c1de183264deaf03788c736` |
| Model trees | `917b6c125def2fd17d0035dc3319f4514249695407fc84bd655be35a6b3e12b8` |

Selected source and input hashes copied from the original preflight:

| Repository-relative reference | Recorded SHA-256 |
|---|---|
| `src/ml_final_evaluation.py` | `aeea0a7ca74f22725672f431995006a8773a5e0717a02e90dafe7f9951208b73` |
| `scripts/run_ml_final_evaluation.py` | `3adfb1d2037f40b9b827f01a73c4b44ee530feee439a9d34295861fdddf05ac1` |
| `config.yaml` | `03056e0d73e1dd12b59ecec9c0f1e31813a00540d580832b41b113b5ec64fa07` |
| `requirements.txt` | `d3387ee2e91aec2f5f081e44bd671d21d55bd2498ccd633c1f1c1856cafaebf3` |
| Original evaluated `README.md` | `8eb56efee39a0578f2cad86e33918a2e74a78a2507fc97cb71d717fb466c1355` |
| `reports/cycle2_calendar/first_run/protocol_selection.json` | `e0cc570f56209e5e88d7b4092e6dfa8de5b902640d2dbcc860dd328bf0778997` |
| `reports/sample_series.csv` | `c36ccd3576cc3d1c098ca8262f6ff7ee127d341da71c50d2cb701dae9c83233f` |
| `data/processed/sales.parquet` | `76d327c8d6fd13a7b1f98fc598fa7ae341f2148b2171a83f1b5868dca24b852b` |
| `data/processed/series.parquet` | `63122912c8ebfe32213e80deb770166de11717a633c82f927c4e3c0490c85f82` |
| `data/raw/calendar.csv` | `d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5` |
| `data/processed/m5_backtest_predictions.parquet` | `54a5332996f3c0c90f5d2a0b9aa129df9be5fde5f85a3f5a1d1efe953328812b` |

The README publication edit occurs after the completed reveal; its new bytes are
not the evaluated README bytes identified above. Original manifests are unchanged.

## Minimal publication set

Publish the README, this documentation summary, and only these six existing
report files (11,103 bytes total):

- [`reveal/overall.csv`](reveal/overall.csv)
- [`reveal/by_window.csv`](reveal/by_window.csv)
- [`reveal/by_category.csv`](reveal/by_category.csv)
- [`reveal/by_demand_group.csv`](reveal/by_demand_group.csv)
- [`reveal/by_horizon_week.csv`](reveal/by_horizon_week.csv)
- [`reveal/manifest.json`](reveal/manifest.json)

The proposed public JSON (`reveal/manifest.json`) was inspected for absolute
paths and machine identifiers: none were found. Its runtime and peak RSS remain
original local execution measurements.

Keep `preflight.json` and the operational markers `../primary_reserved.json`,
`reveal_started.json`, and `score_completed.json` local and unchanged. The markers
control experiment/reveal lifecycle and are not public result tables. Keep raw
data, predictions, model binaries, and private `scored_result/` storage out of Git.
The existing `reports/first_run/`, `reports/repro_run/`, and both `.patch` files
remain untouched and outside this publication set. Publication requires no rerun
of preflight, reveal, recovery, or reproduction commands.

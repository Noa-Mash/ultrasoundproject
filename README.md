# Ultrasound CEUS Bolus Pipeline

## Where Are Caches Created?

The project uses **two caches** to avoid repeating expensive computations.
Both are created automatically when you run the pipeline
(`python us_model/run_bolus_pipeline.py`).

### 1. SVD Cache — `us_model/cache_utils.py`

| What | Details |
|------|---------|
| **Purpose** | Stores SVD-filtered spatial arrays so you can change ROIs without re-running SVD |
| **Created by** | `save_svd_file()` and `save_cache_metadata()` in **`us_model/cache_utils.py`** |
| **Called from** | `process_and_cache()` in **`us_model/ceus_processor.py`** (line 55–61) |
| **Triggered in pipeline** | Step 1 of `run_pipeline()` in **`us_model/run_bolus_pipeline.py`** (line 80) |
| **Directory on disk** | `svd_cache_v2_<hash>_n<n_svd>/` |
| **Validation** | `validate_svd_cache()` in `cache_utils.py` |

### 2. Bolus Cache — `us_model/bolus_cache.py`

| What | Details |
|------|---------|
| **Purpose** | Stores lognormal bolus-fitting results so you can re-visualize without re-fitting |
| **Created by** | `save_bolus_result()` and `save_bolus_cache_metadata()` in **`us_model/bolus_cache.py`** |
| **Called from** | Step 4 of `run_pipeline()` in **`us_model/run_bolus_pipeline.py`** (lines 160–164) |
| **Directory on disk** | `bolus_cache_v1_<hash>/` |
| **Validation** | `validate_bolus_cache()` in `bolus_cache.py` |

### Quick Reference

```
run_bolus_pipeline.py          ← entry point; orchestrates everything
  ├─ Step 1 → ceus_processor.py  → cache_utils.py   (SVD cache)
  ├─ Step 2 → select_roi.py                         (interactive ROI)
  ├─ Step 3 → ceus_processor.py                     (extract + smooth)
  ├─ Step 4 → LogNormalModel.py  → bolus_cache.py   (bolus cache)
  └─ Step 5 → visualization                         (plots)
```

Both cache directories are git-ignored (see `.gitignore`).

## Running the Pipeline

```bash
cd us_model
python run_bolus_pipeline.py
```

Edit the configuration block at the bottom of `run_bolus_pipeline.py` to set
`DATA_DIR`, `ROI_COORDS`, `N_SVD`, and other parameters.

Set `USE_CACHE = True` (default) to enable caching, or `FORCE_REPROCESS = True`
to rebuild the SVD cache from scratch.

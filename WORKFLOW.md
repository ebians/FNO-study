# FNO Training & Evaluation Workflow

This document describes the sequential workflow for establishing baseline evaluation, visualization, hyperparameter search, physical feature improvements, and batch inference.

## 1. Baseline Evaluation

**Purpose**: Extract and display key metrics from trained model summary.

```bash
python eval_baseline.py \
  --summary artifacts/training/training_summary.json \
  --history artifacts/training/training_history.json
```

**Output**:
- Displays training configuration, model performance, training progress in console.

**Typical output**:
```
======================================================================
FNO Baseline Evaluation Report
======================================================================

[Training Configuration]
  Dataset:         artifacts/fno_dataset.npz
  Checkpoint:      artifacts/training/fno_best.pt
  ...
  
[Model Performance]
  Best Epoch:      25
  Best Val MSE:    1.234567e-03
  Test MSE:        1.456789e-03
  Test MAE:        3.456789e-04
```

---

## 2. Prediction Visualization & Error Maps

**Purpose**: Generate visual comparison of predicted vs target fields, and error maps.

**Prerequisites**:
- Ensure `matplotlib` is installed: `pip install matplotlib`
- Must have trained checkpoint and dataset

```bash
python compare_predictions.py \
  --checkpoint artifacts/training/fno_best.pt \
  --dataset artifacts/fno_dataset.npz \
  --num-cases 4 \
  --output-dir artifacts/comparison \
  --device auto
```

**Output Files**:
- `artifacts/comparison/case_0000_comparison.png` - 3-column plot (target | predicted | error)
- `artifacts/comparison/case_0001_comparison.png`, etc.
- `artifacts/comparison/comparison_summary.json` - Per-case metrics in JSON

**Example PNG layout**:
```
[Target Field]  [Predicted Field]  [Error Map (Predicted - Target)]
```

**Summary JSON**:
```json
{
  "num_cases": 4,
  "metrics_per_case": [
    {"case_index": 0, "mae": 1.2e-04, "rmse": 2.3e-04, "max_error": 5.6e-04},
    ...
  ]
}
```

---

## 3. Hyperparameter Grid Search

**Purpose**: Train multiple FNO models with different hyperparameters and compare results.

**Grid Configuration**:
- **Width**: [16, 32, 64]
- **Depth**: [2, 4, 6]
- **Learning Rate**: [1e-4, 1e-3, 1e-2]
- **6 selected configs** for efficient search (not full grid)

```bash
python hp_search.py \
  --dataset artifacts/fno_dataset.npz \
  --split-file artifacts/splits/train_val_test_split.json \
  --epochs 50 \
  --device auto \
  --output-report artifacts/hp_search_report.json
```

**Output Files**:
- `artifacts/hp_search_report.json` - Summary of all configs + best config
- `artifacts/hp_search/narrow_shallow/` - Checkpoint, history, summary for each config
- `artifacts/hp_search/medium_medium/`, etc.

**Report JSON Structure**:
```json
{
  "num_configs": 6,
  "results": [
    {
      "config": {"width": 16, "depth": 2, "learning_rate": 0.001, "label": "narrow_shallow"},
      "summary": {"best_val_mse": 1.23e-03, "test_mse": 1.45e-03, "test_mae": 3.45e-04}
    },
    ...
  ],
  "best_config": {
    "label": "medium_medium",
    "config": {...},
    "test_mse": 1.10e-03,
    "test_mae": 2.50e-04
  }
}
```

---

## 4. Physical Feature Improvements (Copper Density Maps)

**Purpose**: Generate spatially-aware copper density maps to replace uniform layer properties.

**Supported Patterns**:
- `uniform` - Constant density across layer
- `radial` - Higher density at center, lower at edges
- `striped` - Alternating high/low density columns (models wiring tracks)
- `corner` - Density biased toward corners (e.g., power/ground regions)

### Step 4a: Generate Copper Density Map

```bash
# Radial pattern (center denser, edges sparse)
python generate_copper_density_map.py \
  --height 4 \
  --width 4 \
  --pattern radial \
  --center-density 0.8 \
  --edge-density 0.2 \
  --output artifacts/copper_density_radial.json

# Striped pattern (alternating tracks)
python generate_copper_density_map.py \
  --height 4 \
  --width 4 \
  --pattern striped \
  --high-density 0.8 \
  --low-density 0.2 \
  --output artifacts/copper_density_striped.json
```

**Output**:
```json
{
  "pattern": "radial",
  "height": 4,
  "width": 4,
  "density_map": [
    [0.8, 0.75, 0.75, 0.8],
    [0.75, 0.9, 0.9, 0.75],
    [0.75, 0.9, 0.9, 0.75],
    [0.8, 0.75, 0.75, 0.8]
  ]
}
```

### Step 4b: Enhance Case YAML with Copper Density (Manual)

Update your case YAML to reference the density map:

```yaml
# fno_input_case_enhanced.yaml
target:
  quantity: "Top-Surface Z Displacement"
  unit: "um"

grid:
  cell_demo_shape: [4, 4]
  x_cell_centers_demo: [0.1, 0.3, 0.5, 0.7]
  y_cell_centers_demo: [0.1, 0.3, 0.5, 0.7]

global_conditions:
  delta_temperature_C: 50.0
  copper_density_map_path: "artifacts/copper_density_radial.json"  # NEW

layers:
  - thickness_um: 10.0
    youngs_modulus_gpa: 130.0
    poisson_ratio: 0.3
    cte_ppm_per_C: 3.5
    copper_ratio_demo_4x4: null  # Will be loaded from copper_density_map_path
    
  - thickness_um: 20.0
    youngs_modulus_gpa: 4.5
    poisson_ratio: 0.35
    cte_ppm_per_C: 3.0
    copper_ratio_demo_4x4: null
```

### Step 4c: Build Features with Enhanced Map

```bash
python build_fno_features.py \
  --input examples/fno_input_case_enhanced.yaml \
  --output artifacts/fno_features_enhanced.json
```

**Expected Behavior**:
- Layer copper_ratio channels now reflect spatial distribution from density map
- Center cells have higher copper, edges have lower (for radial pattern)
- Improved physical meaning for thermal/structural coupling

---

## 5. Batch Inference & Auto-Report

**Purpose**: Run inference on multiple test cases and generate evaluation report.

```bash
python batch_infer_fno.py \
  --checkpoint artifacts/training/fno_best.pt \
  --dataset artifacts/fno_dataset.npz \
  --output-dir artifacts/batch_inference \
  --device auto \
  --split-type test
```

**Output Files**:
- `artifacts/batch_inference/batch_inference_summary.json` - Aggregated metrics
- `artifacts/batch_inference/report.md` - Markdown report with per-case table

**Report Example** (report.md):
```markdown
# FNO Batch Inference Report

## Summary
- **Total Cases**: 20
- **Mean MAE**: 2.345e-04
- **Mean RMSE**: 3.456e-04
- **Mean Max Error**: 8.765e-04
- **Mean R²**: 0.9234

## Per-Case Results

| Case | MAE | RMSE | Max Error | R² |
|------|-----|------|-----------|-----|
| 0 | 2.345e-04 | 3.456e-04 | 8.765e-04 | 0.9234 |
| 1 | 2.123e-04 | 3.234e-04 | 8.234e-04 | 0.9301 |
| ...
```

**Summary JSON**:
```json
{
  "num_cases": 20,
  "split_type": "test",
  "mean_mae": 2.345e-04,
  "mean_rmse": 3.456e-04,
  "mean_max_error": 8.765e-04,
  "mean_r2": 0.9234,
  "per_case_metrics": [
    {"case_index": 0, "mae": ..., "rmse": ..., "max_error": ..., "r2": ...},
    ...
  ]
}
```

---

## Complete Sequential Workflow Example

```bash
# 0. Train baseline model (prerequisite)
python train_fno.py --epochs 50 --batch-size 8 --width 32 --depth 4

# 1. Evaluate baseline
python eval_baseline.py

# 2. Visualize predictions
python compare_predictions.py --num-cases 4

# 3. Hyperparameter search (takes longer)
python hp_search.py --epochs 30

# 4. Generate and integrate copper density maps
python generate_copper_density_map.py --pattern radial
# Then manually update case YAML with copper_density_map_path
python build_fno_features.py --input examples/fno_input_case_enhanced.yaml

# 5. Batch inference on test set
python batch_infer_fno.py --split-type test
# Review artifacts/batch_inference/report.md
```

---

## Output Directory Structure

```
artifacts/
├── training/
│   ├── fno_best.pt               (trained model)
│   ├── training_summary.json     (metrics summary)
│   └── training_history.json     (epoch-by-epoch loss)
├── comparison/
│   ├── case_0000_comparison.png
│   ├── case_0001_comparison.png
│   └── comparison_summary.json   (per-case metrics)
├── hp_search/
│   ├── hp_search_report.json     (grid search summary)
│   ├── narrow_shallow/           (config dir)
│   ├── medium_medium/
│   └── ...
├── batch_inference/
│   ├── batch_inference_summary.json
│   └── report.md                 (markdown report)
└── copper_density_*.json         (generated density maps)
```

---

## Notes

- **Visualization**: Requires `matplotlib`. Install with: `pip install matplotlib`
- **Device**: Use `--device cuda` for GPU acceleration, or `--device cpu` for CPU-only
- **Reproducibility**: Train/val/test split is saved in checkpoint for consistent evaluation across runs
- **Physical Features**: Copper density maps reflect actual wiring distributions, improving model interpretability
- **Early Stopping**: Training respects early stopping and LR scheduling from original training config

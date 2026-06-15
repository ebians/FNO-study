# FNO-study

Japanese README: [README_ja.md](README_ja.md)

Simple utilities to load, validate, and transform an FNO case YAML into
model-ready spatial channels.

## What Is Implemented

- YAML case loading from `examples/fno_input_case_example.yaml`
- Basic schema validation for key fields and tensor-like map shapes
- FNO input channel construction (layer maps + global channels)
- CLI to export the channel tensor and metadata as JSON

## Quick Start

1. Install dependency:

```bash
pip install pyyaml
```

2. Build features from the sample:

PowerShell:

```powershell
python build_fno_features.py --input examples/fno_input_case_example.yaml --output artifacts/fno_features.json
```

Bash:

```bash
python build_fno_features.py \
  --input examples/fno_input_case_example.yaml \
  --output artifacts/fno_features.json
```

3. Inspect the generated file `artifacts/fno_features.json`.

## Notes

- This implementation currently targets the demo map path (`4x4`) in the sample.
- Boundary-condition masks are included when present in the YAML.

## Next Step: Training Dataset

To start learning the gridded top-surface displacement field, create a target JSON from the FrontISTR result after post-processing, then pack feature/target pairs into a training dataset.

Generate the target JSON from a surface-point export:

```powershell
python build_fno_target_from_surface_points.py --feature-json artifacts/fno_features.json --surface-points examples/fno_surface_points_example.csv --output artifacts/fno_surface_target.json
```

Example surface-point export: `examples/fno_surface_points_example.csv`

The generated target file is `artifacts/fno_surface_target.json`.

Example manifest: `examples/fno_training_manifest_example.json`

Build the dataset:

```powershell
python build_fno_training_dataset.py --manifest examples/fno_training_manifest_example.json --output artifacts/fno_dataset.npz --metadata-output artifacts/fno_dataset_metadata.json
```

This produces:

- `artifacts/fno_dataset.npz` with `inputs` and `targets`
- `artifacts/fno_dataset_metadata.json` with shapes, channels, and case IDs

Practical note:

- If you have raw FrontISTR AVS output, first export the top-surface node values to a table that has at least x, y, and uz.
- This script then interpolates that point cloud onto the feature grid and writes the target JSON.

Direct AVS/UCD path:

```powershell
python build_fno_target_from_avs_ucd.py --feature-json artifacts/fno_features.json --avs-ucd examples/frontistr_ucd_example.txt --output artifacts/fno_surface_target_from_avs_ucd.json --value-field uz
```

Example AVS/UCD input: `examples/frontistr_ucd_example.txt`

This variant reads nodal data directly from an ASCII AVS/UCD file, keeps the top surface by max z, and grids the result onto the feature mesh.

## Train FNO

Train directly from the generated dataset:

```powershell
python train_fno.py --dataset artifacts/fno_dataset.npz --output-dir artifacts/training --split-file artifacts/splits/train_val_test_split.json --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 --epochs 200 --batch-size 8 --learning-rate 1e-3 --early-stopping-patience 20 --scheduler-patience 8
```

Outputs:

- `artifacts/training/fno_best.pt` (best checkpoint)
- `artifacts/training/training_history.json` (per-epoch metrics)
- `artifacts/training/training_summary.json` (run summary)

Notes:

- The train/val/test split is fixed by `--split-file`.
- If the split file does not exist, it is created once and reused in future runs.
- Early stopping and ReduceLROnPlateau scheduler are enabled by default.

## Inference (Single Case)

Predict one case field using the best checkpoint:

```powershell
python infer_fno.py --checkpoint artifacts/training/fno_best.pt --dataset artifacts/fno_dataset.npz --case-index 0 --output artifacts/prediction_case.json
```

Output:

- `artifacts/prediction_case.json` (predicted field, and target/metrics if target exists)

## Baseline Evaluation

View key metrics from the trained model summary:

```powershell
python eval_baseline.py --summary artifacts/training/training_summary.json --history artifacts/training/training_history.json
```

Console output displays:
- Training configuration (dataset, epochs, learning rate, model width/depth)
- Best validation MSE and test MSE/MAE
- Training progress (initial vs final loss)

## Prediction Visualization & Error Maps

Generate side-by-side plots (target | predicted | error) for visual inspection:

```powershell
python compare_predictions.py --checkpoint artifacts/training/fno_best.pt --dataset artifacts/fno_dataset.npz --num-cases 4 --output-dir artifacts/comparison
```

Outputs:
- PNG plots for each case (3-column layout)
- `artifacts/comparison/comparison_summary.json` with per-case MAE, RMSE, max error

Requires: `pip install matplotlib`

## Hyperparameter Search

Run grid search over FNO width, depth, and learning rate:

```powershell
python hp_search.py --dataset artifacts/fno_dataset.npz --split-file artifacts/splits/train_val_test_split.json --epochs 50 --device auto
```

Tests 6 configurations and outputs:
- `artifacts/hp_search_report.json` with all results and best config
- Per-config checkpoint directories under `artifacts/hp_search/`

Searches: width [16, 32, 64] × depth [2, 4, 6] × learning_rate [1e-4, 1e-3, 1e-2]

## Physical Feature Improvements (Copper Density Maps)

Generate spatially-aware copper density maps to enhance model interpretability:

```powershell
# Radial pattern (higher density at center)
python generate_copper_density_map.py --height 4 --width 4 --pattern radial --center-density 0.8 --edge-density 0.2 --output artifacts/copper_density_radial.json

# Striped pattern (alternating high/low columns for wire tracks)
python generate_copper_density_map.py --height 4 --width 4 --pattern striped --high-density 0.8 --low-density 0.2 --output artifacts/copper_density_striped.json
```

Supported patterns: `uniform`, `radial`, `striped`, `corner`

Integration: Update your case YAML to reference the generated density map and rebuild features:

```powershell
python build_fno_features.py --input examples/fno_input_case_enhanced.yaml --output artifacts/fno_features_enhanced.json
```

## Batch Inference & Evaluation Report

Run inference on multiple test cases and generate a markdown summary report:

```powershell
python batch_infer_fno.py --checkpoint artifacts/training/fno_best.pt --dataset artifacts/fno_dataset.npz --split-type test --output-dir artifacts/batch_inference
```

Outputs:
- `artifacts/batch_inference/batch_inference_summary.json` (aggregated metrics)
- `artifacts/batch_inference/report.md` (markdown table with per-case metrics)

Metrics per case: MAE, RMSE, max error, R²
Aggregate metrics: mean values across all cases

## Complete Sequential Workflow

For a full baseline→optimization flow, see [WORKFLOW.md](WORKFLOW.md) for detailed step-by-step instructions.

## README Sync Policy

To prevent drift between English and Japanese documentation, update both README files together.

- Always update `README.md` and `README_ja.md` in the same commit.
- Keep section structure and command examples aligned.
- Use the checklist in [README_SYNC_CHECKLIST.md](README_SYNC_CHECKLIST.md) before commit.

Sync Version: 2026-06-14 (updated with 5 new evaluation scripts)
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


class DatasetValidationError(ValueError):
    """Raised when dataset inputs do not satisfy required constraints."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _shape_of_matrix(matrix: Any) -> Tuple[int, int]:
    if not isinstance(matrix, list) or not matrix:
        raise DatasetValidationError("Matrix must be a non-empty 2D list.")
    if not all(isinstance(row, list) for row in matrix):
        raise DatasetValidationError("Matrix rows must be lists.")

    row_lengths = {len(row) for row in matrix}
    if len(row_lengths) != 1:
        raise DatasetValidationError("Matrix rows must have identical lengths.")

    return len(matrix), len(matrix[0])


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise DatasetValidationError(f"JSON root must be an object: {path}")
    return payload


def _resolve_path(base_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (base_path.parent / candidate).resolve()


def _validate_feature_case(feature_case: Dict[str, Any]) -> Tuple[int, int, int]:
    required_keys = ["case_id", "target", "grid_shape", "num_channels", "channels", "tensor_chw"]
    for key in required_keys:
        if key not in feature_case:
            raise DatasetValidationError(f"Missing required feature key: {key}")

    grid_shape = feature_case["grid_shape"]
    if not isinstance(grid_shape, dict):
        raise DatasetValidationError("Feature grid_shape must be an object.")

    height = grid_shape.get("height")
    width = grid_shape.get("width")
    if not isinstance(height, int) or height <= 0 or not isinstance(width, int) or width <= 0:
        raise DatasetValidationError("Feature grid_shape must contain positive integer height/width.")

    num_channels = feature_case["num_channels"]
    if not isinstance(num_channels, int) or num_channels <= 0:
        raise DatasetValidationError("Feature num_channels must be a positive integer.")

    channels = feature_case["channels"]
    tensor_chw = feature_case["tensor_chw"]
    if not isinstance(channels, list) or len(channels) != num_channels:
        raise DatasetValidationError("Feature channels must be a list whose length matches num_channels.")
    if not isinstance(tensor_chw, list) or len(tensor_chw) != num_channels:
        raise DatasetValidationError("Feature tensor_chw length must match num_channels.")

    for idx, channel_matrix in enumerate(tensor_chw, start=1):
        shape = _shape_of_matrix(channel_matrix)
        if shape != (height, width):
            raise DatasetValidationError(
                f"Feature tensor_chw[{idx}] shape {shape} does not match grid_shape {(height, width)}."
            )
        for row in channel_matrix:
            for value in row:
                if not _is_number(value):
                    raise DatasetValidationError(f"Feature tensor_chw[{idx}] contains a non-numeric value.")

    return height, width, num_channels


def _validate_target_field(target_case: Dict[str, Any], expected_case_id: str, expected_shape: Tuple[int, int]) -> None:
    required_keys = ["case_id", "quantity", "unit", "grid_shape", "field_hw"]
    for key in required_keys:
        if key not in target_case:
            raise DatasetValidationError(f"Missing required target key: {key}")

    if target_case["case_id"] != expected_case_id:
        raise DatasetValidationError(
            f"Target case_id {target_case['case_id']} does not match feature case_id {expected_case_id}."
        )

    grid_shape = target_case["grid_shape"]
    if not isinstance(grid_shape, dict):
        raise DatasetValidationError("Target grid_shape must be an object.")

    height = grid_shape.get("height")
    width = grid_shape.get("width")
    if (height, width) != expected_shape:
        raise DatasetValidationError(
            f"Target grid_shape {(height, width)} does not match feature grid_shape {expected_shape}."
        )

    field_hw = target_case["field_hw"]
    if _shape_of_matrix(field_hw) != expected_shape:
        raise DatasetValidationError("Target field_hw shape does not match feature grid_shape.")

    for row in field_hw:
        for value in row:
            if not _is_number(value):
                raise DatasetValidationError("Target field_hw contains a non-numeric value.")


def _load_manifest_samples(manifest: Dict[str, Any]) -> List[Dict[str, str]]:
    samples = manifest.get("samples")
    if isinstance(samples, list):
        return samples
    if isinstance(manifest.get("cases"), list):
        return manifest["cases"]
    raise DatasetValidationError("Manifest must contain a samples list.")


def build_dataset(manifest_path: Path) -> Dict[str, Any]:
    manifest = load_json(manifest_path)
    samples = _load_manifest_samples(manifest)
    if not samples:
        raise DatasetValidationError("Manifest must contain at least one sample.")

    input_tensors: List[np.ndarray] = []
    target_tensors: List[np.ndarray] = []
    case_ids: List[str] = []
    channels: Sequence[str] | None = None
    target_quantity: str | None = None
    target_unit: str | None = None
    input_grid_shape: Tuple[int, int] | None = None

    for sample in samples:
        if not isinstance(sample, dict):
            raise DatasetValidationError("Each manifest sample must be an object.")

        feature_path_value = sample.get("features_path")
        target_path_value = sample.get("target_path")
        if not isinstance(feature_path_value, str) or not feature_path_value.strip():
            raise DatasetValidationError("Each sample must define features_path.")
        if not isinstance(target_path_value, str) or not target_path_value.strip():
            raise DatasetValidationError("Each sample must define target_path.")

        feature_path = _resolve_path(manifest_path, feature_path_value)
        target_path = _resolve_path(manifest_path, target_path_value)

        feature_case = load_json(feature_path)
        feature_height, feature_width, feature_channels = _validate_feature_case(feature_case)

        target_case = load_json(target_path)
        _validate_target_field(target_case, feature_case["case_id"], (feature_height, feature_width))

        if channels is None:
            channels = list(feature_case["channels"])
            input_grid_shape = (feature_height, feature_width)
            target_quantity = str(target_case["quantity"])
            target_unit = str(target_case["unit"])
        else:
            if list(feature_case["channels"]) != list(channels):
                raise DatasetValidationError("All samples must share the same input channel order.")
            if (feature_height, feature_width) != input_grid_shape:
                raise DatasetValidationError("All samples must share the same input grid shape.")
            if str(target_case["quantity"]) != target_quantity or str(target_case["unit"]) != target_unit:
                raise DatasetValidationError("All samples must share the same target quantity and unit.")

        x = np.asarray(feature_case["tensor_chw"], dtype=np.float32)
        y = np.asarray(target_case["field_hw"], dtype=np.float32)[np.newaxis, ...]

        if x.shape != (feature_channels, feature_height, feature_width):
            raise DatasetValidationError(f"Unexpected input tensor shape: {x.shape}")
        if y.shape != (1, feature_height, feature_width):
            raise DatasetValidationError(f"Unexpected target tensor shape: {y.shape}")

        input_tensors.append(x)
        target_tensors.append(y)
        case_ids.append(str(feature_case["case_id"]))

    inputs = np.stack(input_tensors, axis=0)
    targets = np.stack(target_tensors, axis=0)

    return {
        "inputs": inputs,
        "targets": targets,
        "case_ids": case_ids,
        "channels": list(channels or []),
        "input_grid_shape": input_grid_shape,
        "target_quantity": target_quantity,
        "target_unit": target_unit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an FNO training dataset from feature and surface-field JSON files.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("examples/fno_training_manifest_example.json"),
        help="Dataset manifest JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/fno_dataset.npz"),
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("artifacts/fno_dataset_metadata.json"),
        help="Output metadata JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = build_dataset(args.manifest)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, inputs=dataset["inputs"], targets=dataset["targets"], case_ids=np.array(dataset["case_ids"]))

    metadata = {
        "num_samples": int(dataset["inputs"].shape[0]),
        "input_shape": list(dataset["inputs"].shape),
        "target_shape": list(dataset["targets"].shape),
        "case_ids": dataset["case_ids"],
        "channels": dataset["channels"],
        "input_grid_shape": list(dataset["input_grid_shape"] or []),
        "target_quantity": dataset["target_quantity"],
        "target_unit": dataset["target_unit"],
    }

    with args.metadata_output.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Built dataset with {metadata['num_samples']} samples")
    print(f"Output: {args.output}")
    print(f"Metadata: {args.metadata_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
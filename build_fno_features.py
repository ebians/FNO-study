import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml


class CaseValidationError(ValueError):
    """Raised when the input case YAML does not satisfy required constraints."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _shape_of_matrix(matrix: Any) -> Tuple[int, int]:
    if not isinstance(matrix, list) or not matrix:
        raise CaseValidationError("Matrix must be a non-empty 2D list.")
    if not all(isinstance(row, list) for row in matrix):
        raise CaseValidationError("Matrix rows must be lists.")

    row_lengths = {len(row) for row in matrix}
    if len(row_lengths) != 1:
        raise CaseValidationError("Matrix rows must have identical lengths.")

    return len(matrix), len(matrix[0])


def _broadcast_scalar(value: float, height: int, width: int) -> List[List[float]]:
    return [[float(value) for _ in range(width)] for _ in range(height)]


def load_case(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise CaseValidationError("Top-level YAML structure must be a mapping.")
    return payload


def validate_case(case: Dict[str, Any]) -> None:
    required_top_keys = ["target", "grid", "global_conditions", "layers"]
    for key in required_top_keys:
        if key not in case:
            raise CaseValidationError(f"Missing required top-level key: {key}")

    target = case["target"]
    if not isinstance(target, dict):
        raise CaseValidationError("target must be a mapping.")
    for key in ["quantity", "unit"]:
        if key not in target or not isinstance(target[key], str) or not target[key].strip():
            raise CaseValidationError(f"target.{key} must be a non-empty string.")

    grid = case["grid"]
    if not isinstance(grid, dict):
        raise CaseValidationError("grid must be a mapping.")

    cell_demo_shape = grid.get("cell_demo_shape")
    if (
        not isinstance(cell_demo_shape, list)
        or len(cell_demo_shape) != 2
        or not all(isinstance(v, int) and v > 0 for v in cell_demo_shape)
    ):
        raise CaseValidationError("grid.cell_demo_shape must be [height, width] with positive integers.")

    global_conditions = case["global_conditions"]
    if not isinstance(global_conditions, dict):
        raise CaseValidationError("global_conditions must be a mapping.")
    if not _is_number(global_conditions.get("delta_temperature_C")):
        raise CaseValidationError("global_conditions.delta_temperature_C must be numeric.")

    layers = case["layers"]
    if not isinstance(layers, list) or not layers:
        raise CaseValidationError("layers must be a non-empty list.")

    expected_h, expected_w = cell_demo_shape
    for i, layer in enumerate(layers, start=1):
        if not isinstance(layer, dict):
            raise CaseValidationError(f"layers[{i}] must be a mapping.")

        for scalar_key in ["thickness_um", "youngs_modulus_gpa", "poisson_ratio", "cte_ppm_per_C"]:
            if not _is_number(layer.get(scalar_key)):
                raise CaseValidationError(f"layers[{i}].{scalar_key} must be numeric.")

        matrix = layer.get("copper_ratio_demo_4x4")
        shape = _shape_of_matrix(matrix)
        if shape != (expected_h, expected_w):
            raise CaseValidationError(
                f"layers[{i}].copper_ratio_demo_4x4 shape {shape} does not match cell_demo_shape {(expected_h, expected_w)}."
            )

        for row in matrix:
            for value in row:
                if not _is_number(value):
                    raise CaseValidationError(f"layers[{i}].copper_ratio_demo_4x4 contains non-numeric value.")
                if value < 0.0 or value > 1.0:
                    raise CaseValidationError(
                        f"layers[{i}].copper_ratio_demo_4x4 value {value} is outside [0.0, 1.0]."
                    )


def _extract_bc_masks(
    global_conditions: Dict[str, Any], expected_h: int, expected_w: int
) -> Dict[str, List[List[float]]]:
    masks = global_conditions.get("boundary_condition_component_masks_demo_5x5_nodes", {})
    if not isinstance(masks, dict):
        return {}

    out: Dict[str, List[List[float]]] = {}
    for dof_key in ["ux", "uy", "uz"]:
        node_mask = masks.get(dof_key)
        if not isinstance(node_mask, list):
            continue

        # The sample stores node-grid masks (5x5). We crop to cell-grid resolution (4x4).
        # For production, replace this with interpolation/projection that matches your solver.
        try:
            node_h, node_w = _shape_of_matrix(node_mask)
        except CaseValidationError:
            continue

        if node_h < expected_h or node_w < expected_w:
            continue

        cell_mask: List[List[float]] = []
        for y in range(expected_h):
            row: List[float] = []
            for x in range(expected_w):
                row.append(float(node_mask[y][x]))
            cell_mask.append(row)

        out[f"bc_{dof_key}_mask"] = cell_mask

    return out


def build_feature_channels(case: Dict[str, Any]) -> Dict[str, Any]:
    grid = case["grid"]
    layers = case["layers"]
    global_conditions = case["global_conditions"]

    h, w = grid["cell_demo_shape"]
    x_coords = grid.get("x_cell_centers_demo")
    y_coords = grid.get("y_cell_centers_demo")

    if not isinstance(x_coords, list) or len(x_coords) != w:
        raise CaseValidationError("grid.x_cell_centers_demo length must match cell width.")
    if not isinstance(y_coords, list) or len(y_coords) != h:
        raise CaseValidationError("grid.y_cell_centers_demo length must match cell height.")

    channels: List[str] = []
    tensor: List[List[List[float]]] = []

    for idx, layer in enumerate(layers, start=1):
        layer_prefix = f"layer{idx:02d}"

        channels.append(f"{layer_prefix}_copper_ratio")
        tensor.append([[float(v) for v in row] for row in layer["copper_ratio_demo_4x4"]])

        channels.append(f"{layer_prefix}_thickness_um")
        tensor.append(_broadcast_scalar(float(layer["thickness_um"]), h, w))

        channels.append(f"{layer_prefix}_youngs_modulus_gpa")
        tensor.append(_broadcast_scalar(float(layer["youngs_modulus_gpa"]), h, w))

        channels.append(f"{layer_prefix}_poisson_ratio")
        tensor.append(_broadcast_scalar(float(layer["poisson_ratio"]), h, w))

        channels.append(f"{layer_prefix}_cte_ppm_per_C")
        tensor.append(_broadcast_scalar(float(layer["cte_ppm_per_C"]), h, w))

    channels.append("delta_temperature_C")
    tensor.append(_broadcast_scalar(float(global_conditions["delta_temperature_C"]), h, w))

    bc_masks = _extract_bc_masks(global_conditions, h, w)
    for key in ["bc_ux_mask", "bc_uy_mask", "bc_uz_mask"]:
        if key in bc_masks:
            channels.append(key)
            tensor.append(bc_masks[key])

    x_map = [[float(x_coords[x]) for x in range(w)] for _ in range(h)]
    y_map = [[float(y_coords[y]) for _ in range(w)] for y in range(h)]

    channels.append("x_coord")
    tensor.append(x_map)
    channels.append("y_coord")
    tensor.append(y_map)

    return {
        "case_id": case.get("case_id"),
        "target": case["target"],
        "grid_shape": {"height": h, "width": w},
        "num_channels": len(channels),
        "channels": channels,
        "tensor_chw": tensor,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FNO feature channels from case YAML.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("examples/fno_input_case_example.yaml"),
        help="Input case YAML path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/fno_features.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case = load_case(args.input)
    validate_case(case)
    payload = build_feature_channels(case)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Built {payload['num_channels']} channels for case_id={payload['case_id']}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

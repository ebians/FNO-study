import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


class TargetBuildError(ValueError):
    """Raised when surface-point inputs do not satisfy required constraints."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _shape_of_matrix(matrix: Any) -> Tuple[int, int]:
    if not isinstance(matrix, list) or not matrix:
        raise TargetBuildError("Matrix must be a non-empty 2D list.")
    if not all(isinstance(row, list) for row in matrix):
        raise TargetBuildError("Matrix rows must be lists.")

    row_lengths = {len(row) for row in matrix}
    if len(row_lengths) != 1:
        raise TargetBuildError("Matrix rows must have identical lengths.")

    return len(matrix), len(matrix[0])


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise TargetBuildError(f"JSON root must be an object: {path}")
    return payload


def load_csv_points(path: Path, value_column: str) -> List[Dict[str, float]]:
    points: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"x", "y", value_column}
        if not required_columns.issubset(reader.fieldnames or []):
            raise TargetBuildError(
                f"CSV must contain columns: {', '.join(sorted(required_columns))}"
            )

        for row in reader:
            try:
                points.append(
                    {
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "value": float(row[value_column]),
                    }
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise TargetBuildError(f"Invalid numeric value in CSV row: {row}") from exc

    if not points:
        raise TargetBuildError(f"No surface points were loaded from {path}")
    return points


def _find_channel_index(channels: Sequence[str], channel_name: str) -> int:
    try:
        return list(channels).index(channel_name)
    except ValueError as exc:
        raise TargetBuildError(f"Missing required channel: {channel_name}") from exc


def _extract_grid_coordinates(feature_case: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    required_keys = ["channels", "tensor_chw", "grid_shape", "target"]
    for key in required_keys:
        if key not in feature_case:
            raise TargetBuildError(f"Missing required feature key: {key}")

    channels = feature_case["channels"]
    tensor_chw = feature_case["tensor_chw"]
    x_index = _find_channel_index(channels, "x_coord")
    y_index = _find_channel_index(channels, "y_coord")

    x_map = tensor_chw[x_index]
    y_map = tensor_chw[y_index]
    height, width = _shape_of_matrix(x_map)
    if _shape_of_matrix(y_map) != (height, width):
        raise TargetBuildError("x_coord and y_coord channel shapes must match.")

    x_coords = [float(x_map[0][col]) for col in range(width)]
    y_coords = [float(y_map[row][0]) for row in range(height)]
    return x_coords, y_coords


def _interpolate_idw(points: List[Dict[str, float]], x: float, y: float, power: float = 2.0) -> float:
    epsilon = 1e-12
    exact_matches = [point["value"] for point in points if abs(point["x"] - x) < epsilon and abs(point["y"] - y) < epsilon]
    if exact_matches:
        return float(exact_matches[0])

    numerator = 0.0
    denominator = 0.0
    for point in points:
        dx = point["x"] - x
        dy = point["y"] - y
        distance_sq = dx * dx + dy * dy
        if distance_sq < epsilon:
            return float(point["value"])

        weight = 1.0 / (distance_sq ** (power / 2.0))
        numerator += weight * point["value"]
        denominator += weight

    if denominator == 0.0:
        raise TargetBuildError("Interpolation denominator became zero.")
    return float(numerator / denominator)


def _build_field_hw(points: List[Dict[str, float]], x_coords: List[float], y_coords: List[float]) -> List[List[float]]:
    field_hw: List[List[float]] = []
    for y in y_coords:
        row: List[float] = []
        for x in x_coords:
            row.append(_interpolate_idw(points, x, y))
        field_hw.append(row)
    return field_hw


def build_target_json(feature_path: Path, surface_points_path: Path, value_column: str, quantity: str | None, unit: str | None) -> Dict[str, Any]:
    feature_case = load_json(feature_path)
    x_coords, y_coords = _extract_grid_coordinates(feature_case)

    points = load_csv_points(surface_points_path, value_column=value_column)
    field_hw = _build_field_hw(points, x_coords, y_coords)

    target_info = feature_case.get("target", {})
    output_quantity = quantity or str(target_info.get("quantity", "top_surface_warpage"))
    output_unit = unit or str(target_info.get("unit", "mm"))

    return {
        "case_id": feature_case.get("case_id"),
        "quantity": output_quantity,
        "unit": output_unit,
        "grid_shape": {
            "height": len(y_coords),
            "width": len(x_coords),
        },
        "field_hw": field_hw,
        "source": {
            "surface_points_path": str(surface_points_path),
            "method": "inverse_distance_weighting",
            "value_column": value_column,
            "notes": "Use this after exporting the top surface node values from FrontISTR AVS results to CSV or a similar tabular format.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a gridded top-surface target JSON from surface point data.")
    parser.add_argument(
        "--feature-json",
        type=Path,
        default=Path("artifacts/fno_features.json"),
        help="Feature JSON path used to recover the target grid.",
    )
    parser.add_argument(
        "--surface-points",
        type=Path,
        default=Path("examples/fno_surface_points_example.csv"),
        help="CSV file containing x, y, and value columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/fno_surface_target.json"),
        help="Output target JSON path.",
    )
    parser.add_argument(
        "--value-column",
        type=str,
        default="uz",
        help="CSV column containing the surface displacement value.",
    )
    parser.add_argument(
        "--quantity",
        type=str,
        default=None,
        help="Optional target quantity override.",
    )
    parser.add_argument(
        "--unit",
        type=str,
        default=None,
        help="Optional target unit override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_target_json(args.feature_json, args.surface_points, args.value_column, args.quantity, args.unit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Built target JSON for case_id={payload['case_id']}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
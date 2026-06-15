import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class TargetBuildError(ValueError):
    """Raised when AVS/UCD inputs do not satisfy required constraints."""


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


def _clean_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("#") or stripped.startswith("$") or stripped.startswith("//"):
        return ""
    return stripped


def _iter_clean_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            cleaned = _clean_line(raw_line)
            if cleaned:
                lines.append(cleaned)
    return lines


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise TargetBuildError(f"JSON root must be an object: {path}")
    return payload


def _find_channel_index(channels: Sequence[str], channel_name: str) -> int:
    try:
        return list(channels).index(channel_name)
    except ValueError as exc:
        raise TargetBuildError(f"Missing required channel: {channel_name}") from exc


def _extract_grid_coordinates(feature_case: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    channels = feature_case.get("channels")
    tensor_chw = feature_case.get("tensor_chw")
    if not isinstance(channels, list) or not isinstance(tensor_chw, list):
        raise TargetBuildError("Feature JSON must contain channels and tensor_chw.")

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


def _parse_avs_counts(first_line: str) -> Tuple[int, int, int, int]:
    tokens = first_line.split()
    if len(tokens) < 4:
        raise TargetBuildError(
            "AVS/UCD first line must contain at least: num_nodes num_elems num_node_data num_cell_data"
        )

    try:
        return int(tokens[0]), int(tokens[1]), int(tokens[2]), int(tokens[3])
    except ValueError as exc:
        raise TargetBuildError("AVS/UCD first line must contain integer counts.") from exc


def _parse_node_line(line: str) -> Tuple[int, float, float, float]:
    tokens = line.split()
    if len(tokens) < 4:
        raise TargetBuildError(f"Invalid node line: {line}")

    try:
        node_id = int(tokens[0])
        x = float(tokens[1])
        y = float(tokens[2])
        z = float(tokens[3])
    except ValueError as exc:
        raise TargetBuildError(f"Invalid numeric value in node line: {line}") from exc
    return node_id, x, y, z


def _parse_scalar_field_from_data_lines(lines: List[str], start_index: int, num_nodes: int) -> Tuple[List[float], int, str]:
    if start_index >= len(lines):
        raise TargetBuildError("Missing AVS/UCD nodal data section.")

    header_tokens = lines[start_index].split()
    if len(header_tokens) < 2:
        raise TargetBuildError("Nodal data header must contain a field name and component count.")

    field_name = header_tokens[0]
    try:
        component_count = int(header_tokens[1])
    except ValueError as exc:
        raise TargetBuildError("Nodal data component count must be an integer.") from exc

    if component_count <= 0:
        raise TargetBuildError("Nodal data component count must be positive.")

    values: List[float] = []
    cursor = start_index + 1
    while cursor < len(lines) and len(values) < num_nodes:
        tokens = lines[cursor].split()
        if len(tokens) < component_count:
            raise TargetBuildError(f"Nodal data line has fewer than {component_count} values: {lines[cursor]}")

        try:
            component_values = [float(tokens[i]) for i in range(component_count)]
        except ValueError as exc:
            raise TargetBuildError(f"Invalid nodal data value in line: {lines[cursor]}") from exc

        values.append(component_values[-1])
        cursor += 1

    if len(values) != num_nodes:
        raise TargetBuildError(
            f"Nodal data field {field_name} has {len(values)} entries but expected {num_nodes}."
        )

    return values, cursor, field_name


def _infer_top_surface_nodes(nodes: List[Tuple[int, float, float, float]], tolerance: float = 1e-8) -> List[Tuple[int, float, float, float]]:
    max_z = max(node[3] for node in nodes)
    top_nodes = [node for node in nodes if abs(node[3] - max_z) <= tolerance]
    if not top_nodes:
        raise TargetBuildError("Unable to identify top-surface nodes from AVS/UCD coordinates.")
    return top_nodes


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


def parse_avs_ucd_surface_points(avs_path: Path, value_field: str | None) -> Dict[str, Any]:
    lines = _iter_clean_lines(avs_path)
    if len(lines) < 2:
        raise TargetBuildError(f"AVS/UCD file is too short: {avs_path}")

    num_nodes, num_elems, num_node_data, num_cell_data = _parse_avs_counts(lines[0])
    _ = num_elems
    _ = num_cell_data

    cursor = 1
    nodes: List[Tuple[int, float, float, float]] = []
    for _ in range(num_nodes):
        if cursor >= len(lines):
            raise TargetBuildError("Unexpected end of file while reading nodes.")
        nodes.append(_parse_node_line(lines[cursor]))
        cursor += 1

    if not nodes:
        raise TargetBuildError("No nodes were read from AVS/UCD.")

    if cursor < len(lines):
        token_count = len(lines[cursor].split())
        if token_count == 1 and num_node_data > 0:
            cursor += 1

    top_nodes = _infer_top_surface_nodes(nodes)

    field_name = value_field
    extracted_values: Dict[int, float] = {}
    if num_node_data > 0:
        while cursor < len(lines):
            header_tokens = lines[cursor].split()
            if len(header_tokens) < 2:
                break

            candidate_field_name = header_tokens[0]
            try:
                component_count = int(header_tokens[1])
            except ValueError:
                break

            if component_count <= 0:
                raise TargetBuildError("Node data component count must be positive.")

            cursor += 1
            field_values: List[Tuple[int, float]] = []
            for node_index in range(num_nodes):
                if cursor >= len(lines):
                    raise TargetBuildError("Unexpected end of file while reading nodal data values.")

                tokens = lines[cursor].split()
                if len(tokens) < component_count:
                    raise TargetBuildError(
                        f"Nodal data line has fewer than {component_count} values: {lines[cursor]}"
                    )

                try:
                    values = [float(tokens[i]) for i in range(component_count)]
                except ValueError as exc:
                    raise TargetBuildError(f"Invalid nodal data line: {lines[cursor]}") from exc

                field_values.append((nodes[node_index][0], values[-1]))
                cursor += 1

            if field_name is None or candidate_field_name == field_name:
                extracted_values = {node_id: value for node_id, value in field_values}
                field_name = candidate_field_name
                break

        if not extracted_values:
            raise TargetBuildError(
                f"Requested nodal field {field_name or '<unspecified>'} was not found in AVS/UCD data."
            )
    else:
        raise TargetBuildError("AVS/UCD file does not contain nodal data.")

    surface_points = []
    for node_id, x, y, z in top_nodes:
        if node_id not in extracted_values:
            raise TargetBuildError(f"Missing nodal value for node {node_id} in field {field_name}.")
        surface_points.append({"x": x, "y": y, "value": extracted_values[node_id], "z": z})

    return {
        "field_name": field_name,
        "surface_points": surface_points,
        "node_count": num_nodes,
        "top_node_count": len(surface_points),
    }


def build_target_json(feature_path: Path, avs_path: Path, quantity: str | None, unit: str | None, value_field: str | None) -> Dict[str, Any]:
    feature_case = load_json(feature_path)
    x_coords, y_coords = _extract_grid_coordinates(feature_case)

    parsed = parse_avs_ucd_surface_points(avs_path, value_field=value_field)
    field_hw = _build_field_hw(parsed["surface_points"], x_coords, y_coords)

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
            "avs_ucd_path": str(avs_path),
            "method": "inverse_distance_weighting_on_top_surface_nodes",
            "value_field": parsed["field_name"],
            "node_count": parsed["node_count"],
            "top_node_count": parsed["top_node_count"],
            "notes": "Directly read AVS/UCD nodal values, keep the top-surface nodes by max z, and grid them onto the feature mesh.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a gridded top-surface target JSON from a FrontISTR AVS/UCD file.")
    parser.add_argument(
        "--feature-json",
        type=Path,
        default=Path("artifacts/fno_features.json"),
        help="Feature JSON path used to recover the target grid.",
    )
    parser.add_argument(
        "--avs-ucd",
        type=Path,
        default=Path("examples/frontistr_ucd_example.txt"),
        help="FrontISTR AVS/UCD ASCII file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/fno_surface_target_from_avs_ucd.json"),
        help="Output target JSON path.",
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
    parser.add_argument(
        "--value-field",
        type=str,
        default=None,
        help="Optional AVS/UCD nodal data field name to extract. If omitted, the first field is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_target_json(args.feature_json, args.avs_ucd, args.quantity, args.unit, args.value_field)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Built target JSON for case_id={payload['case_id']}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
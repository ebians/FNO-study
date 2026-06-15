import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class DensityMapError(ValueError):
    """Raised when density map generation fails."""


def generate_uniform_density_map(height: int, width: int, density: float) -> List[List[float]]:
    """Generate uniform copper density map."""
    if not 0.0 <= density <= 1.0:
        raise DensityMapError(f"Density must be in [0, 1], got {density}")
    return [[float(density) for _ in range(width)] for _ in range(height)]


def generate_radial_density_map(height: int, width: int, center_density: float, edge_density: float) -> List[List[float]]:
    """Generate radial copper density map (higher in center, lower at edges)."""
    if not (0.0 <= center_density <= 1.0 and 0.0 <= edge_density <= 1.0):
        raise DensityMapError("Densities must be in [0, 1]")

    center_y, center_x = height / 2.0, width / 2.0
    max_dist = np.sqrt(center_y ** 2 + center_x ** 2)

    result = []
    for y in range(height):
        row = []
        for x in range(width):
            dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
            normalized_dist = dist / max_dist
            density = center_density + (edge_density - center_density) * normalized_dist
            row.append(float(np.clip(density, 0.0, 1.0)))
        result.append(row)
    return result


def generate_striped_density_map(height: int, width: int, high_density: float, low_density: float, stripe_width: int = 1) -> List[List[float]]:
    """Generate striped copper density map (alternating high/low density columns)."""
    if not (0.0 <= high_density <= 1.0 and 0.0 <= low_density <= 1.0):
        raise DensityMapError("Densities must be in [0, 1]")

    result = []
    for y in range(height):
        row = []
        for x in range(width):
            stripe_idx = (x // stripe_width) % 2
            density = high_density if stripe_idx == 0 else low_density
            row.append(float(density))
        result.append(row)
    return result


def generate_corner_density_map(height: int, width: int, corner_densities: Dict[str, float]) -> List[List[float]]:
    """Generate density map with higher density in specified corners."""
    defaults = {"tl": 0.5, "tr": 0.5, "bl": 0.5, "br": 0.5}
    for key in defaults:
        if key in corner_densities:
            if not 0.0 <= corner_densities[key] <= 1.0:
                raise DensityMapError(f"{key} density must be in [0, 1]")
            defaults[key] = corner_densities[key]

    result = []
    for y in range(height):
        row = []
        for x in range(width):
            y_norm = y / (height - 1) if height > 1 else 0.5
            x_norm = x / (width - 1) if width > 1 else 0.5

            tl_val = defaults["tl"] * (1 - x_norm) * (1 - y_norm)
            tr_val = defaults["tr"] * x_norm * (1 - y_norm)
            bl_val = defaults["bl"] * (1 - x_norm) * y_norm
            br_val = defaults["br"] * x_norm * y_norm

            density = tl_val + tr_val + bl_val + br_val
            row.append(float(np.clip(density, 0.0, 1.0)))
        result.append(row)
    return result


def load_copper_density_patterns(config_file: Path) -> Dict[str, Any]:
    """Load copper density patterns from a JSON config."""
    with config_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate copper density maps for FNO features.")
    parser.add_argument("--height", type=int, default=4, help="Map height (cells).")
    parser.add_argument("--width", type=int, default=4, help="Map width (cells).")
    parser.add_argument("--pattern", type=str, default="uniform", help="Pattern: uniform, radial, striped, corner.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/copper_density_map.json"), help="Output JSON path.")
    parser.add_argument("--density", type=float, default=0.5, help="Density for uniform pattern.")
    parser.add_argument("--center-density", type=float, default=0.8, help="Center density for radial pattern.")
    parser.add_argument("--edge-density", type=float, default=0.2, help="Edge density for radial pattern.")
    parser.add_argument("--high-density", type=float, default=0.8, help="High density for striped pattern.")
    parser.add_argument("--low-density", type=float, default=0.2, help="Low density for striped pattern.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.pattern == "uniform":
        density_map = generate_uniform_density_map(args.height, args.width, args.density)
    elif args.pattern == "radial":
        density_map = generate_radial_density_map(args.height, args.width, args.center_density, args.edge_density)
    elif args.pattern == "striped":
        density_map = generate_striped_density_map(args.height, args.width, args.high_density, args.low_density)
    elif args.pattern == "corner":
        density_map = generate_corner_density_map(args.height, args.width, {})
    else:
        raise DensityMapError(f"Unknown pattern: {args.pattern}")

    output = {
        "pattern": args.pattern,
        "height": args.height,
        "width": args.width,
        "density_map": density_map,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {args.pattern} copper density map: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

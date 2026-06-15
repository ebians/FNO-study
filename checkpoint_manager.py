import argparse
import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CheckpointMetadata:
    """Metadata for a managed checkpoint."""
    checkpoint_id: str
    name: str
    description: str
    checkpoint_path: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    created_at: str
    tags: List[str]


class CheckpointManagerError(ValueError):
    """Raised when checkpoint management fails."""


class CheckpointManager:
    def __init__(self, catalog_path: Path = Path("artifacts/checkpoint_catalog.json")):
        self.catalog_path = catalog_path
        self.catalog: Dict[str, Dict[str, Any]] = {}
        self.load_catalog()

    def load_catalog(self) -> None:
        """Load existing catalog or create new one."""
        if self.catalog_path.exists():
            with self.catalog_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                self.catalog = data.get("checkpoints", {})
        else:
            self.catalog = {}

    def save_catalog(self) -> None:
        """Save catalog to disk."""
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with self.catalog_path.open("w", encoding="utf-8") as f:
            json.dump({"checkpoints": self.catalog, "last_updated": datetime.now().isoformat()}, f, indent=2)

    def register_checkpoint(
        self,
        checkpoint_id: str,
        checkpoint_path: Path,
        name: str,
        description: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[List[str]] = None,
    ) -> None:
        """Register a checkpoint in the catalog."""
        if not checkpoint_path.exists():
            raise CheckpointManagerError(f"Checkpoint not found: {checkpoint_path}")

        if checkpoint_id in self.catalog:
            raise CheckpointManagerError(f"Checkpoint ID already exists: {checkpoint_id}")

        metadata = {
            "checkpoint_id": checkpoint_id,
            "name": name,
            "description": description,
            "checkpoint_path": str(checkpoint_path.absolute()),
            "config": config,
            "metrics": metrics,
            "created_at": datetime.now().isoformat(),
            "tags": tags or [],
        }
        self.catalog[checkpoint_id] = metadata
        self.save_catalog()
        print(f"✓ Registered checkpoint: {checkpoint_id}")

    def list_checkpoints(self, tag_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered checkpoints, optionally filtered by tag."""
        results = []
        for checkpoint_id, metadata in self.catalog.items():
            if tag_filter:
                if tag_filter not in metadata.get("tags", []):
                    continue
            results.append(metadata)
        return sorted(results, key=lambda x: x["created_at"], reverse=True)

    def get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get path to a checkpoint by ID."""
        if checkpoint_id not in self.catalog:
            raise CheckpointManagerError(f"Checkpoint not found: {checkpoint_id}")
        return Path(self.catalog[checkpoint_id]["checkpoint_path"])

    def compare_checkpoints(self, checkpoint_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across multiple checkpoints."""
        results = []
        for cid in checkpoint_ids:
            if cid not in self.catalog:
                print(f"Warning: Checkpoint not found: {cid}")
                continue
            metadata = self.catalog[cid]
            results.append(
                {
                    "id": cid,
                    "name": metadata.get("name"),
                    "metrics": metadata.get("metrics", {}),
                    "config": {k: v for k, v in metadata.get("config", {}).items() if k in ["width", "depth", "learning_rate", "batch_size"]},
                }
            )
        return {"comparisons": results}

    def copy_checkpoint(self, checkpoint_id: str, dest_path: Path, new_id: str) -> None:
        """Copy a checkpoint to a new location and register it."""
        src_path = self.get_checkpoint_path(checkpoint_id)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path, dest_path)
        
        src_metadata = self.catalog[checkpoint_id]
        self.register_checkpoint(
            checkpoint_id=new_id,
            checkpoint_path=dest_path,
            name=f"Copy of {src_metadata['name']}",
            description=f"Copied from {checkpoint_id}",
            config=src_metadata.get("config", {}),
            metrics=src_metadata.get("metrics", {}),
            tags=src_metadata.get("tags", []) + ["copy"],
        )
        print(f"✓ Copied checkpoint to: {dest_path}")

    def tag_checkpoint(self, checkpoint_id: str, tags: List[str]) -> None:
        """Add tags to a checkpoint."""
        if checkpoint_id not in self.catalog:
            raise CheckpointManagerError(f"Checkpoint not found: {checkpoint_id}")
        current_tags = set(self.catalog[checkpoint_id].get("tags", []))
        current_tags.update(tags)
        self.catalog[checkpoint_id]["tags"] = list(current_tags)
        self.save_catalog()
        print(f"✓ Tagged checkpoint {checkpoint_id}: {tags}")

    def delete_checkpoint(self, checkpoint_id: str, remove_file: bool = False) -> None:
        """Remove checkpoint from catalog (optionally delete file too)."""
        if checkpoint_id not in self.catalog:
            raise CheckpointManagerError(f"Checkpoint not found: {checkpoint_id}")
        
        if remove_file:
            ckpt_path = Path(self.catalog[checkpoint_id]["checkpoint_path"])
            if ckpt_path.exists():
                ckpt_path.unlink()
                print(f"✓ Deleted checkpoint file: {ckpt_path}")
        
        del self.catalog[checkpoint_id]
        self.save_catalog()
        print(f"✓ Removed from catalog: {checkpoint_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage FNO model checkpoints catalog.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # register command
    register_parser = subparsers.add_parser("register", help="Register a checkpoint")
    register_parser.add_argument("checkpoint_id", help="Unique checkpoint ID")
    register_parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to checkpoint file")
    register_parser.add_argument("--name", type=str, required=True, help="Friendly name")
    register_parser.add_argument("--description", type=str, default="", help="Description")
    register_parser.add_argument("--config-json", type=Path, help="Config JSON file")
    register_parser.add_argument("--metrics-json", type=Path, help="Metrics JSON file")
    register_parser.add_argument("--tags", type=str, nargs="*", default=[], help="Tags")

    # list command
    list_parser = subparsers.add_parser("list", help="List all checkpoints")
    list_parser.add_argument("--tag", type=str, help="Filter by tag")
    list_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare checkpoints")
    compare_parser.add_argument("checkpoint_ids", nargs="+", help="Checkpoint IDs to compare")
    compare_parser.add_argument("--output", type=Path, help="Save comparison to JSON")

    # info command
    info_parser = subparsers.add_parser("info", help="Show checkpoint info")
    info_parser.add_argument("checkpoint_id", help="Checkpoint ID")

    # copy command
    copy_parser = subparsers.add_parser("copy", help="Copy checkpoint")
    copy_parser.add_argument("checkpoint_id", help="Source checkpoint ID")
    copy_parser.add_argument("--dest", type=Path, required=True, help="Destination path")
    copy_parser.add_argument("--new-id", type=str, required=True, help="New checkpoint ID")

    # tag command
    tag_parser = subparsers.add_parser("tag", help="Tag a checkpoint")
    tag_parser.add_argument("checkpoint_id", help="Checkpoint ID")
    tag_parser.add_argument("--add", type=str, nargs="+", required=True, help="Tags to add")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete checkpoint from catalog")
    delete_parser.add_argument("checkpoint_id", help="Checkpoint ID")
    delete_parser.add_argument("--remove-file", action="store_true", help="Also delete checkpoint file")

    parser.add_argument("--catalog", type=Path, default=Path("artifacts/checkpoint_catalog.json"), help="Catalog file path")

    return parser.parse_args()


def format_table(checkpoints: List[Dict[str, Any]]) -> str:
    """Format checkpoint list as table."""
    if not checkpoints:
        return "No checkpoints found."
    
    lines = ["ID | Name | Test MSE | Test MAE | Config | Tags"]
    lines.append("-" * 80)
    
    for ckpt in checkpoints:
        ckpt_id = ckpt.get("checkpoint_id", "N/A")[:15]
        name = ckpt.get("name", "N/A")[:20]
        test_mse = ckpt.get("metrics", {}).get("test_mse", "N/A")
        test_mae = ckpt.get("metrics", {}).get("test_mae", "N/A")
        
        config_str = "w={},d={}".format(
            ckpt.get("config", {}).get("width", "?"),
            ckpt.get("config", {}).get("depth", "?"),
        )
        tags_str = ",".join(ckpt.get("tags", []))[:20]
        
        test_mse_str = f"{test_mse:.2e}" if isinstance(test_mse, float) else str(test_mse)
        test_mae_str = f"{test_mae:.2e}" if isinstance(test_mae, float) else str(test_mae)
        
        lines.append(f"{ckpt_id} | {name} | {test_mse_str} | {test_mae_str} | {config_str} | {tags_str}")
    
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    manager = CheckpointManager(args.catalog)

    if args.command == "register":
        config = {}
        metrics = {}
        
        if args.config_json and args.config_json.exists():
            with args.config_json.open("r", encoding="utf-8") as f:
                config = json.load(f)
        
        if args.metrics_json and args.metrics_json.exists():
            with args.metrics_json.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
        
        manager.register_checkpoint(
            checkpoint_id=args.checkpoint_id,
            checkpoint_path=args.checkpoint_path,
            name=args.name,
            description=args.description,
            config=config,
            metrics=metrics,
            tags=args.tags,
        )

    elif args.command == "list":
        checkpoints = manager.list_checkpoints(tag_filter=args.tag)
        if args.format == "table":
            print(format_table(checkpoints))
        else:
            print(json.dumps(checkpoints, indent=2))

    elif args.command == "compare":
        result = manager.compare_checkpoints(args.checkpoint_ids)
        print(json.dumps(result, indent=2))
        if args.output:
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"Comparison saved: {args.output}")

    elif args.command == "info":
        if args.checkpoint_id not in manager.catalog:
            print(f"Checkpoint not found: {args.checkpoint_id}")
            return 1
        info = manager.catalog[args.checkpoint_id]
        print(json.dumps(info, indent=2))

    elif args.command == "copy":
        manager.copy_checkpoint(args.checkpoint_id, args.dest, args.new_id)

    elif args.command == "tag":
        manager.tag_checkpoint(args.checkpoint_id, args.add)

    elif args.command == "delete":
        manager.delete_checkpoint(args.checkpoint_id, remove_file=args.remove_file)

    else:
        parser = argparse.ArgumentParser()
        parser.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

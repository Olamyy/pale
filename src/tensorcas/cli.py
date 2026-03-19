import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from tensorcas.errors import tensorcasError
from tensorcas.manifest import ManifestReader
from tensorcas.store import TensorCasStore
from tensorcas.cas.filesystem import FilesystemBackend
from tensorcas.registry.registry import Registry


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024**2:.1f} MB"


def _die(msg: str) -> None:
    print(f"tensorcas: error: {msg}", file=sys.stderr)
    sys.exit(1)


def _confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")


def _open_registry(root: Path):

    db = root / "registry.db"
    if not db.exists():
        _die(f"no tensorcas store at {root} (registry.db not found)")
    return Registry.from_path(db)


def _steps_summary(steps: list) -> str:
    if not steps:
        return "(none)"
    if len(steps) <= 10:
        return "  ".join(str(s) for s in steps)
    return f"{steps[0]}..{steps[-1]}  ({len(steps)} checkpoints)"


def _cmd_list(args: argparse.Namespace) -> None:
    root: Path = args.root
    run_id: Optional[str] = args.run
    fmt: str = args.format

    registry = _open_registry(root)

    if run_id is None:
        runs = registry.list_runs()
        if fmt == "json":
            data = {rid: registry.list_checkpoints(rid) for rid in runs}
            print(json.dumps(data))
            return
        if not runs:
            print("(no runs)")
            return
        for rid in runs:
            steps = registry.list_checkpoints(rid)
            print(f"{rid:<20}  steps: {_steps_summary(steps)}")
    else:
        steps = registry.list_checkpoints(run_id)
        if fmt == "json":
            print(json.dumps(steps))
            return
        if not steps:
            print(f"(no checkpoints for run {run_id!r})")
            return
        print(_steps_summary(steps))


def _print_stats(s: Dict[str, Any], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(s))
        return
    for key, val in s.items():
        label = f"{key}:"
        if key == "total_bytes":
            print(f"{label:<20} {_fmt_bytes(val)}")
        elif key == "dedup_ratio":
            print(f"{label:<20} {val:.4f}")
        elif key == "steps":
            print(f"{label:<20} {', '.join(str(x) for x in val)}")
        else:
            print(f"{label:<20} {val}")


def _cmd_stats(args: argparse.Namespace) -> None:
    root: Path = args.root
    run_id: Optional[str] = args.run
    fmt: str = args.format

    if run_id is None:
        db = root / "registry.db"
        if not db.exists():
            _die(f"no tensorcas store at {root} (registry.db not found)")

        _print_stats(TensorCasStore.store_stats(root), fmt)
    else:
        registry = _open_registry(root)

        steps = registry.list_checkpoints(run_id)
        total_chunks = 0
        unique_chunks: set = set()
        total_bytes = 0
        for step in steps:
            path = registry.get_manifest_path(run_id, step)
            if path and path.exists():
                manifest = ManifestReader.read(path)
                for record in manifest.tensors.values():
                    total_bytes += record.byte_length
                    for ref in record.chunks:
                        total_chunks += 1
                        unique_chunks.add(ref.hash)
        dedup_ratio = len(unique_chunks) / total_chunks if total_chunks else 1.0
        s = {
            "run_id": run_id,
            "checkpoints": len(steps),
            "steps": steps,
            "total_chunks": total_chunks,
            "unique_chunks": len(unique_chunks),
            "dedup_ratio": round(dedup_ratio, 4),
            "total_bytes": total_bytes,
        }
        _print_stats(s, fmt)


def _cmd_gc(args: argparse.Namespace) -> None:
    root: Path = args.root
    grace: int = args.grace
    yes: bool = args.yes

    registry = _open_registry(root)

    if not yes:
        if not _confirm(f"Sweep orphaned blobs older than {grace}h?"):
            print("Aborted.")
            return

    backend = FilesystemBackend(root)
    report = registry.gc(grace_period_hours=grace)
    for h in report.swept_hashes:
        backend.delete(h)
    print(
        f"Deleted {report.deleted_blobs} blob(s), freed {_fmt_bytes(report.freed_bytes)}"
    )


def _cmd_delete(args: argparse.Namespace) -> None:
    root: Path = args.root
    run_id: str = args.run
    step: int = args.step
    yes: bool = args.yes

    if not yes:
        if not _confirm(f"Delete checkpoint {run_id}/step={step}?"):
            print("Aborted.")
            return

    registry = _open_registry(root)
    registry.delete_checkpoint(run_id, step)

    manifest_path = root / "manifests" / run_id / f"step_{step:06d}.json"
    manifest_path.unlink(missing_ok=True)

    print("Deleted.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tensorcas",
        description="Inspect and manage a tensorcas checkpoint store.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        metavar="ROOT",
        help="Root directory of the tensorcas store (default: current directory)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List runs and checkpoints")
    p_list.add_argument(
        "--run", default=None, metavar="RUN_ID", help="Filter to a specific run"
    )
    p_list.add_argument("--format", choices=["text", "json"], default="text")

    p_stats = sub.add_parser("stats", help="Show dedup statistics")
    p_stats.add_argument(
        "--run", default=None, metavar="RUN_ID", help="Scope stats to a run"
    )
    p_stats.add_argument("--format", choices=["text", "json"], default="text")

    p_gc = sub.add_parser("gc", help="Sweep orphaned blobs")
    p_gc.add_argument(
        "--grace",
        type=int,
        default=24,
        metavar="HOURS",
        help="Grace period in hours (default: 24)",
    )
    p_gc.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    p_del = sub.add_parser("delete", help="Delete a single checkpoint")
    p_del.add_argument("--run", required=True, metavar="RUN_ID")
    p_del.add_argument("--step", type=int, required=True, metavar="STEP")
    p_del.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    try:
        if args.command == "list":
            _cmd_list(args)
        elif args.command == "stats":
            _cmd_stats(args)
        elif args.command == "gc":
            _cmd_gc(args)
        elif args.command == "delete":
            _cmd_delete(args)
    except tensorcasError as exc:
        _die(str(exc))

"""Whitelist-only private dynamic dataset and standalone code handoff builder."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from build_highd_handoff import module_closure, json_read, digest, write_new


def assets(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh archive destination")
    collection = "outputs/dynamic_pairs_seed1000_x5_v1"
    training = "outputs/dynamic_pairs_ppo_smoke_seed7_v1"
    files = {collection + "/collection_protocol.json", collection + "/collection_summary.json"}
    for name in ("sequences.json", "sequence_manifest.json", "audit_summary.json"):
        files.add(collection + "/sequence_audit/" + name)
    summary = json_read(root / collection / "collection_summary.json")
    for run in summary["runs"]:
        for field in ("episode_path", "summary_path"):
            files.add(Path(run[field]).resolve().relative_to(root).as_posix())
    for path in (root / training).rglob("*"):
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    contract = json_read(root / "configs/highd_dynamic_pairs_freeze_v1.json")
    if summary["natural_target_sha256"] != contract["natural_target_sha256"]:
        raise ValueError("Source data differs from frozen dynamic target")
    manifest = {"schema_version": 1, "source_root": root.as_posix(),
                "scope": "Authorized team transfer: generated simulation records, new sequences and smoke checkpoint; no raw driving datasets",
                "natural_target_sha256": contract["natural_target_sha256"],
                "sequence_manifest": collection + "/sequence_audit/sequence_manifest.json",
                "training_root": training, "collection_root": collection,
                "files": {name: {"sha256": digest(root / name), "size": (root / name).stat().st_size} for name in sorted(files)}}
    payload = json.dumps(manifest, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dynamic_assets/manifest.json", payload)
        for name in sorted(files):
            archive.write(root / name, "dynamic_assets/files/" + name)
    lock = {"schema_version": 1, "archive": output.name, "sha256": digest(output),
            "private_manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "file_count": len(files), "uncompressed_bytes": sum(v["size"] for v in manifest["files"].values()),
            "natural_target_sha256": contract["natural_target_sha256"],
            "requires": "Unchanged highd_frozen_assets.zip / configs/highd_portable_assets.lock.json"}
    write_new(root / "configs/highd_dynamic_assets.lock.json", lock)
    print(json.dumps(lock, indent=2))


def code(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh source archive")
    files, _ = module_closure(root, ["scenario_reconstruction/highd_dynamic_handoff.py",
                                   "scenario_reconstruction/highd_portable.py",
                                   "scenario_reconstruction/highd_critical_sequence_train.py"])
    files = set(files) | set(json_read(root / "configs/highd_portable_assets.lock.json")["public_files"])
    files.update([".gitignore", ".gitattributes", "LICENSE", "HANDOFF.md",
        "configs/highd_portable_assets.lock.json", "configs/highd_checkpoint_compatibility_v1.json",
        "configs/highd_dynamic_assets.lock.json", "configs/highd_dynamic_pairs_v1.json",
        "configs/highd_dynamic_pairs_freeze_v1.json", "configs/highd_dual_bv_meanprecision_ppo_v1.json",
        "configs/requirements-highd-sequence-ppo.txt", "configs/requirements-highd-portable.txt",
        "configs/setup_highd_macos.sh", "configs/run_highd_dynamic_handoff.sh", "configs/run_highd_dynamic_handoff.ps1",
        "configs/run_highd_dynamic_pairs.ps1", "docs/highD动态多对版本交接.md",
        "docs/highD动态多对自然驾驶与单对干预.md", "docs/highD双BV完整闭环交接.md",
        "results/highd_dynamic_pairs_batch40_v1.json", "results/highd_dynamic_handoff_v1.json",
        "tests/test_highd_dynamic_handoff.py", "tests/test_highd_dynamic_pairs.py", "tests/test_highd_portable.py",
        "tests/test_highd_mean_precision_policy.py", "tools/build_highd_dynamic_handoff.py", "tools/build_highd_handoff.py"])
    output.parent.mkdir(parents=True, exist_ok=True)
    for rel in files:
        if not (root / rel).is_file():
            raise ValueError("Missing packaged source: " + rel)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for rel in sorted(files):
            archive.write(root / rel, "highd_dynamic_handoff/" + rel)
    print(json.dumps({"archive": str(output), "sha256": digest(output), "file_count": len(files), "bytes": output.stat().st_size}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--code", action="store_true")
    args = parser.parse_args()
    (code if args.code else assets)(args.root, args.output)

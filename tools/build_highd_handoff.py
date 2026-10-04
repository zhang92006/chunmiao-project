"""Build a scoped handoff from a research checkout; never collect raw datasets.

Copies are byte-preserving because the frozen reference checks source hashes.
The private archive is for authorized team transfer, not Git publication.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def module_closure(root, entries):
    pending, found, external = list(entries), set(), set()
    while pending:
        rel = pending.pop()
        if rel in found:
            continue
        found.add(rel)
        path = root / rel
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        package = list(Path(rel).parent.parts)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[:len(package) - node.level + 1]
                    if node.module:
                        base += node.module.split(".")
                    names = [".".join(base)] + [".".join(base + [a.name]) for a in node.names]
                elif node.module:
                    names = [node.module] + [node.module + "." + a.name for a in node.names]
            for name in names:
                candidate = Path(*name.split("."))
                choices = [candidate.with_suffix(".py"), candidate / "__init__.py"]
                match = next((p for p in choices if (root / p).is_file()), None)
                if match:
                    pending.append(match.as_posix())
                else:
                    external.add(name.split(".")[0])
        for parent in path.relative_to(root).parents:
            init = parent / "__init__.py"
            if parent != Path(".") and (root / init).is_file():
                pending.append(init.as_posix())
    return sorted(found), sorted(external - {Path(p).parts[0] for p in found})


def build(source, destination, output):
    source, destination, output = map(lambda p: Path(p).resolve(), (source, destination, output))
    if source == destination or output.exists():
        raise ValueError("Use separate source/destination and a fresh assets output")
    freeze_path = "configs/highd_dual_ndd_research_freeze_v1.json"
    freeze = json_read(source / freeze_path)
    for path, expected in freeze["pins"].items():
        if digest(source / path) != expected:
            raise ValueError("Frozen source changed: " + path)
    entries = ["scenario_reconstruction/highd_dual_ndd_behavior.py",
               "scenario_reconstruction/highd_critical_sequences.py"]
    sources, external = module_closure(source, entries)
    public = set(sources) | {freeze_path, "configs/highd_dual_ndd_behavior_v1.json"}
    public.update(p for p in freeze["pins"] if p.startswith("maps/"))
    for name in ("test_highd_dual_ndd_criticality.py", "test_highd_critical_sequences.py"):
        public.add("tests/" + name)
    def local(path):
        p = Path(path)
        return p.relative_to(source).as_posix() if p.is_absolute() else p.as_posix()
    assets = set()
    def add(path):
        rel = local(path)
        if ".." in Path(rel).parts or not (source / rel).is_file():
            raise ValueError("Missing/out-of-root asset: " + rel)
        assets.add(rel)
        return json_read(source / rel) if rel.endswith(".json") else None
    bundle = add(freeze["bundle_manifest"])
    add(freeze["pilot_config"])
    interface = add(bundle["interface_config"]["path"])
    add(bundle["lane_pair_model"]["path"])
    add(bundle["lane_pair_model"]["train_summary"])
    add(bundle["motion_model"]["path"])
    candidate = add(interface["lane_reference_contract"])
    add(interface["following_model"]["path"])
    add(interface["onset_model"]["path"])
    add(candidate["model"]["path"])
    add(candidate["config"])
    single = add(candidate["reference_ndd"])
    add(single["model"]["path"])
    sequence = "data_analysis/raw_data/highd_cav_first_sequences_train80_v1"
    for name in ("sequences.json", "sequence_manifest.json", "audit_summary.json"):
        add(sequence + "/" + name)
    training = "data_analysis/raw_data/highd_cav_first_ppo_meanprecision_seed7_iter200_v1"
    for path in (source / training).rglob("*"):
        if path.is_file():
            add(path)
    # Preflight before copying: never silently overwrite another task's code.
    maintained = {"scenario_reconstruction/templates.py", "d2rl_training/conditional_chain.py",
                  "d2rl_training/highd_event_contract.py"}
    for rel in public - maintained:
        target = destination / rel
        if target.exists() and digest(target) != digest(source / rel):
            raise ValueError("Destination differs; review explicitly before replacing: " + rel)
    for rel in sorted(public):
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source / rel, target)
    manifest = {"schema_version": 1, "source_root": source.as_posix(),
                "scope": "private frozen fitted models, generated sequences, checkpoint; no raw highD/SHRP2",
                "files": {rel: {"sha256": digest(source / rel), "size": (source / rel).stat().st_size}
                          for rel in sorted(assets)},
                "frozen_source_hashes": {rel: digest(destination / rel) for rel in sorted(public)},
                "sequence_manifest": sequence + "/sequence_manifest.json", "training_root": training,
                "source_sumo_version": "1.27.0"}
    output.mkdir(parents=True)
    private = output / "highd_frozen_assets.zip"
    with zipfile.ZipFile(private, "x", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("portable_assets/manifest.json", json.dumps(manifest, indent=2))
        for rel in sorted(assets):
            archive.write(source / rel, "portable_assets/files/" + rel)
    lock = {"schema_version": 1, "asset_archive": private.name, "sha256": digest(private),
            "private_manifest_sha256": hashlib.sha256(json.dumps(manifest, indent=2).encode()).hexdigest(),
            "asset_count": len(assets), "asset_uncompressed_bytes": sum(s["size"] for s in manifest["files"].values()),
            "public_files": sorted(public), "external_import_roots": external,
            "source_sumo_version": "1.27.0",
            "transfer": "Authorized team transfer only; do not commit the private archive or extracted assets"}
    write_new(destination / "configs/highd_portable_assets.lock.json", lock)
    print(json.dumps({"public_file_count": len(public), "asset_count": len(assets),
                      "asset_archive": str(private), "archive_bytes": private.stat().st_size,
                      "external_import_roots": external}, indent=2))


def code_archive(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh code archive path")
    lock = json_read(root / "configs/highd_portable_assets.lock.json")
    modules, _ = module_closure(root, ["scenario_reconstruction/highd_portable.py",
                                     "scenario_reconstruction/highd_critical_sequence_train.py"])
    files = set(modules) | set(lock["public_files"])
    files.update({".gitattributes", ".gitignore", "LICENSE", "HANDOFF.md",
        "configs/highd_portable_assets.lock.json", "configs/highd_checkpoint_compatibility_v1.json",
        "configs/highd_dual_bv_meanprecision_ppo_v1.json", "configs/requirements-highd-sequence-ppo.txt",
        "configs/requirements-highd-portable.txt", "configs/setup_highd_macos.sh", "configs/run_highd_portable.sh",
        "docs/highD双BV完整闭环交接.md", "docs/highD双BV PPO最小协作包.md",
        "docs/highD双BV策略修复与200轮训练结果.md", "results/highd_cav_first_ppo_meanprecision_iter200_summary.json",
        "results/highd_portable_handoff_validation_v1.json",
        "tests/test_highd_portable.py", "tests/test_highd_mean_precision_policy.py",
        "scenario_reconstruction/highd_sequence_policy_diagnostic.py",
        "configs/highd_cav_first_sequences_train80_manifest.template.json", "tools/build_highd_handoff.py"})
    for rel in files:
        if not (root / rel).is_file():
            raise ValueError("Missing source-package file: " + rel)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for rel in sorted(files):
            archive.write(root / rel, "highd_dual_bv_handoff/" + rel)
    print(json.dumps({"code_archive": str(output), "file_count": len(files),
                      "sha256": digest(output), "bytes": output.stat().st_size}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-archive", action="store_true")
    args = parser.parse_args()
    if args.code_archive:
        code_archive(args.destination, args.output)
    else:
        if not args.source_root:
            parser.error("--source-root required to collect private assets")
        build(args.source_root, args.destination, args.output)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze question-bank families and materialize explicitly reviewed Gym rows.

Run from the repository root with ``python -m resources_servers.stem_judge.prepare``.
No candidate answers or historical verdicts are read into the preparation logic.
Without --contracts this produces a complete unresolved ledger and empty Gym files.
The output is an engineering preparation artifact, never training admission evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ADAPTER_REVISION = "question-bank-contract-v1"
SPLIT_SEED = "stem-judge-family-split-v1"
DOMAIN_ALIASES = {"phys": "physics", "bio": "biology", "chem": "chemistry / materials"}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _object(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _text(row: dict, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def family_splits(rows: list[dict], *, development_ids: set[str], seed: str = SPLIT_SEED) -> dict[str, dict]:
    """Join known lineage duplicates before assigning a deterministic 20% holdout.

    This groups recorded identities, not unidentified semantic duplicates. All rows
    of each component share a split; supplied conflict IDs force development.
    """
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        keys = [("id", _text(row, "id"))]
        for field in ("original_question_sha256", "version_sha256"):
            if row.get(field):
                keys.append((field, row[field]))
        for field in ("qa_variant_ids", "review_ids"):
            keys.extend((field, value) for value in row.get(field, []) if value)
        upstream = _object(row.get("upstream") or {})
        if upstream.get("source_record_sha256"):
            keys.append(("source_record_sha256", upstream["source_record_sha256"]))
        record_id = upstream.get("original_record_id")
        if upstream.get("dataset") and record_id is not None and record_id != "":
            keys.append(("upstream_record", _json([upstream["dataset"], record_id])))
        for membership in row.get("source_memberships", []):
            if membership.get("cohort") and membership.get("source_record_id"):
                keys.append(("membership", _json([membership["cohort"], membership["source_record_id"]])))
        for key in keys:
            if key in seen:
                a, b = root(i), root(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = i

    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[root(i)].append(row["id"])
    result = {}
    for members in components.values():
        members = sorted(members)
        family_id = _hash(_json(members))
        forced = bool(development_ids.intersection(members))
        heldout = int(_hash(seed + ":" + family_id), 16) % 100 < 20
        split = "development" if forced or not heldout else "heldout"
        for task_id in members:
            result[task_id] = {"family_id": family_id, "split": split, "forced_development": forced}
    return result


def _load_contracts(path: Path | None) -> dict[tuple[str, str], dict]:
    if path is None:
        return {}
    # The ledger-only path is standard-library-only. Reviewed contracts use the
    # exact server schema, avoiding a second permissive schema implementation.
    from resources_servers.stem_judge.contracts import TaskContract

    result = {}
    for line in path.read_text().splitlines():
        entry = _object(json.loads(line))
        if "contract" in entry:
            if set(entry) - {"contract", "rubric"}:
                raise ValueError("reviewed entries allow only contract and rubric")
            contract = TaskContract.model_validate(entry["contract"])
            rubric = entry.get("rubric")
        else:
            contract = TaskContract.model_validate(entry)
            rubric = None
        if rubric is not None:
            if contract.task_type != "rubric" or not isinstance(rubric, list) or not rubric:
                raise ValueError("rubric criteria require a rubric contract and nonempty list")
            ids = set()
            positive_ids = set()
            absolute_total = 0.0
            for criterion in rubric:
                criterion = _object(criterion)
                if set(criterion) != {"id", "criterion", "points"}:
                    raise ValueError("rubric criterion keys must be id, criterion, points")
                identifier = _text(criterion, "id")
                _text(criterion, "criterion")
                points = criterion["points"]
                if isinstance(points, bool) or not isinstance(points, (float, int)):
                    raise ValueError("rubric points must be numeric")
                if not math.isfinite(points) or points == 0 or identifier in ids:
                    raise ValueError("rubric points must be finite and nonzero; IDs must be unique")
                ids.add(identifier)
                absolute_total += abs(points)
                if points > 0:
                    positive_ids.add(identifier)
            if not math.isfinite(absolute_total):
                raise ValueError("rubric absolute point total must be finite")
            if not positive_ids or not set(contract.mandatory_criteria).issubset(positive_ids):
                raise ValueError("rubric must have positive points and every mandatory criterion must be positive")
        key = (contract.task_id, contract.task_version)
        if key in result:
            raise ValueError(f"duplicate reviewed contract: {key}")
        result[key] = {"contract": contract.model_dump(mode="json"), "rubric": rubric}
    return result


def prepare(
    dataset: Path,
    output_dir: Path,
    *,
    development_ids: set[str],
    contracts_path: Path | None = None,
    seed: str = SPLIT_SEED,
) -> dict:
    """Write a new ledger, family manifest, coverage receipt and separated Gym rows.

    Existing output directories are refused. Contract-ready means schema/reference
    checks passed; every row remains ``training_admitted: false`` until live validation.
    """
    if output_dir.exists():
        raise FileExistsError(output_dir)
    rows, hashes = [], []
    plain_hash = hashlib.sha256()
    with gzip.open(dataset, "rb") as handle:
        for line in handle:
            plain_hash.update(line)
            row = _object(json.loads(line))
            for field in ("id", "version_sha256", "problem", "version", "domain"):
                _text(row, field)
            if row["version"] not in {"original_valid_pair", "deepseek_repaired"}:
                raise ValueError(f"unsupported source version: {row['version']}")
            rows.append(row)
            hashes.append(hashlib.sha256(line).hexdigest())
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError("dataset contains duplicate task IDs")
    if development_ids - ids:
        raise ValueError(f"development IDs absent from dataset: {sorted(development_ids - ids)}")
    contracts = _load_contracts(contracts_path)
    unknown = contracts.keys() - {(r["id"], r["version_sha256"]) for r in rows}
    if unknown:
        raise ValueError(f"reviewed contracts do not match selected task versions: {sorted(unknown)}")
    splits = family_splits(rows, development_ids=development_ids, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    coverage = defaultdict(Counter)
    artifact_names = ["ledger.jsonl", "families.jsonl", "development.gym.jsonl", "heldout.gym.jsonl", "coverage.json"]
    with (
        (output_dir / "ledger.jsonl").open("x") as ledger,
        (output_dir / "development.gym.jsonl").open("x") as development,
        (output_dir / "heldout.gym.jsonl").open("x") as heldout,
    ):
        for line_no, (row, source_hash) in enumerate(zip(rows, hashes, strict=True), start=1):
            repaired = row["version"] == "deepseek_repaired"
            reference = row.get("expected_answer")
            reference_hash = _hash(reference) if isinstance(reference, str) else None
            review = contracts.get((row["id"], row["version_sha256"]))
            contract = review["contract"] if review else None
            kind = _object(row.get("reference") or {}).get("kind")
            state = "needs_contract"
            if kind in {"tests_only", "native_grader"} or (contract and contract["task_type"] == "native"):
                state = "native_assets_unavailable"
            elif contract:
                source_domain = DOMAIN_ALIASES.get(row["domain"].casefold(), row["domain"].casefold())
                contract_domain = DOMAIN_ALIASES.get(contract["domain"].casefold(), contract["domain"].casefold())
                if contract["repaired"] != repaired:
                    state = "repair_version_mismatch"
                elif contract["policy_input_revision"] != "original":
                    state = "unsupported_policy_input_revision"
                elif contract_domain not in {"unspecified", source_domain}:
                    state = "domain_mismatch"
                elif repaired and not contract["assets_verified"]:
                    state = "repaired_assets_unverified"
                elif contract["reference_sha256"] != reference_hash or reference_hash is None:
                    state = "reference_hash_mismatch"
                elif not reference.strip():
                    state = "missing_reference"
                elif contract["task_type"] == "rubric" and not review["rubric"]:
                    state = "missing_rubric"
                else:
                    state = "contract_ready"
            lineage = {
                key: row.get(key)
                for key in ("original_question_sha256", "qa_variant_ids", "source_memberships", "upstream")
            }
            item = {
                "task_id": row["id"],
                "task_version": row["version_sha256"],
                "version": row["version"],
                "domain": row["domain"],
                "repaired": repaired,
                "dataset_line": line_no,
                "source_row_sha256": source_hash,
                "problem_sha256": _hash(row["problem"]),
                "reference_sha256": reference_hash,
                "reference_kind": kind,
                "lineage": lineage,
                **splits[row["id"]],
                "adapter_revision": ADAPTER_REVISION,
                "contract_sha256": _hash(_json(review)) if review else None,
                "task_contract": contract,
                "state": state,
                "training_admitted": False,
                "unresolved_reason": None if state == "contract_ready" else state,
            }
            ledger.write(_json(item) + "\n")
            counts[state] += 1
            group = (row["domain"], "repaired" if repaired else "KEEP")
            coverage[group]["selected"] += 1
            coverage[group][state] += 1
            if state == "contract_ready":
                gym_row = {
                    "id": row["id"],
                    "uuid": row["id"],
                    "agent_ref": {"type": "responses_api_agents", "name": "stem_contract_simple_agent"},
                    "responses_create_params": {"input": [{"role": "user", "content": row["problem"]}]},
                    "expected_answer": reference,
                    "task_contract": contract,
                    "metadata": {"lineage": lineage, "source_row_sha256": source_hash, **splits[row["id"]]},
                }
                if review["rubric"]:
                    gym_row["rubric"] = review["rubric"]
                destination = heldout if item["split"] == "heldout" else development
                destination.write(_json(gym_row) + "\n")
    families = defaultdict(list)
    for task_id, split in splits.items():
        families[split["family_id"]].append(task_id)
    with (output_dir / "families.jsonl").open("x") as handle:
        for family_id, members in sorted(families.items()):
            handle.write(_json({**splits[members[0]], "task_ids": sorted(members)}) + "\n")
    coverage_rows = [
        {"domain": domain, "selection": selection, **dict(count), "training_admitted": 0, "live_evaluated": 0}
        for (domain, selection), count in sorted(coverage.items())
    ]
    (output_dir / "coverage.json").write_text(_json(coverage_rows) + "\n")
    receipt = {
        "adapter_revision": ADAPTER_REVISION,
        "dataset_path": str(dataset.resolve()),
        "compressed_sha256": _file_hash(dataset),
        "uncompressed_sha256": plain_hash.hexdigest(),
        "rows": len(rows),
        "families": len(families),
        "split_seed": seed,
        "forced_development_ids": sorted(development_ids),
        "contracts_sha256": _file_hash(contracts_path) if contracts_path else None,
        "counts": dict(counts),
        "training_admitted": 0,
        "limitations": [
            "Recorded lineage grouping cannot detect unidentified semantic duplicates.",
            "Contract-ready is not accuracy validation, runner validation, or training admission.",
            "Heldout candidate answers were not used by this preparation step.",
        ],
        "artifacts_sha256": {name: _file_hash(output_dir / name) for name in artifact_names},
    }
    (output_dir / "receipt.json").write_text(_json(receipt) + "\n")
    return receipt


def main() -> None:
    """Command-line entrypoint for a single immutable preparation directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-ids", type=Path, required=True, help="JSON list of known development task IDs")
    parser.add_argument("--contracts", type=Path, help="JSONL of TaskContract or {contract, rubric} reviewed records")
    parser.add_argument("--split-seed", default=SPLIT_SEED)
    args = parser.parse_args()
    development_ids = json.loads(args.development_ids.read_text())
    if not isinstance(development_ids, list) or not all(isinstance(item, str) for item in development_ids):
        parser.error("--development-ids must contain a JSON list of strings")
    receipt = prepare(
        args.dataset,
        args.output_dir,
        development_ids=set(development_ids),
        contracts_path=args.contracts,
        seed=args.split_seed,
    )
    print(json.dumps({key: receipt[key] for key in ("rows", "families", "counts", "training_admitted")}, indent=2))


if __name__ == "__main__":
    main()

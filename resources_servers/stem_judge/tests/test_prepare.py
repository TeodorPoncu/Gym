# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preparation regressions use invented tasks, never private question-bank data."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from resources_servers.stem_judge.prepare import family_splits, prepare


def _row(task_id: str, **overrides) -> dict:
    return {
        "id": task_id,
        "version_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
        "original_question_sha256": hashlib.sha256((task_id + " original").encode()).hexdigest(),
        "version": "original_valid_pair",
        "domain": "Mathematics",
        "problem": "What is 1 + 1?",
        "expected_answer": "2",
        "reference": {"kind": "qa"},
        "upstream": {},
        **overrides,
    }


def _dataset(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "questions.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _review(row: dict, **overrides) -> dict:
    return {
        "task_id": row["id"],
        "task_version": row["version_sha256"],
        "task_type": "answer",
        "reference_sha256": hashlib.sha256(row["expected_answer"].encode()).hexdigest(),
        **overrides,
    }


def _reviews(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "contracts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_family_transitive_closure_forces_all_variants_into_development() -> None:
    rows = [
        _row("a", upstream={"dataset": "source", "original_record_id": 0}),
        _row("b", upstream={"dataset": "source", "original_record_id": 0}, qa_variant_ids=["variant"]),
        _row("c", qa_variant_ids=["variant"]),
    ]
    first = family_splits(rows, development_ids={"c"})
    second = family_splits(list(reversed(rows)), development_ids={"c"})
    assert first == second
    assert len({item["family_id"] for item in first.values()}) == 1
    assert all(item["split"] == "development" and item["forced_development"] for item in first.values())


def test_source_identifiers_are_namespaced_and_missing_ids_do_not_merge() -> None:
    rows = [
        _row("a", upstream={"dataset": "source1", "original_record_id": 3}),
        _row("b", upstream={"dataset": "source2", "original_record_id": 3}),
        _row("c", upstream={"dataset": "source2", "original_record_id": None}),
        _row("d", upstream={"dataset": "source2", "original_record_id": None}),
    ]
    splits = family_splits(rows, development_ids=set())
    assert len({item["family_id"] for item in splits.values()}) == 4


def test_ledger_retains_every_row_without_automatic_admission(tmp_path: Path) -> None:
    rows = [_row("a"), _row("b", reference={"kind": "native_grader"}, expected_answer=None)]
    source = _dataset(tmp_path, rows)
    output = tmp_path / "prepared"
    receipt = prepare(source, output, development_ids={"a"})
    ledger = _lines(output / "ledger.jsonl")
    assert len(ledger) == receipt["rows"] == 2
    assert receipt["counts"] == {"needs_contract": 1, "native_assets_unavailable": 1}
    assert not any(row["training_admitted"] for row in ledger)
    assert (output / "development.gym.jsonl").read_text() == ""
    assert (output / "heldout.gym.jsonl").read_text() == ""
    assert receipt["uncompressed_sha256"] == hashlib.sha256(gzip.decompress(source.read_bytes())).hexdigest()
    for name, digest in receipt["artifacts_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        prepare(source, output, development_ids={"a"})


def test_reviewed_contract_outputs_policy_without_reference_or_labels(tmp_path: Path) -> None:
    from resources_servers.stem_judge.app import StemJudgeRunRequest
    from responses_api_agents.simple_agent.app import SimpleAgentRunRequest

    row = _row("a", kimi_profile={"n_correct": 6}, rl_suitability={"decision": "RL_SUITABLE"})
    output = tmp_path / "prepared"
    receipt = prepare(
        _dataset(tmp_path, [row]),
        output,
        development_ids={"a"},
        contracts_path=_reviews(tmp_path, [_review(row)]),
    )
    assert receipt["counts"] == {"contract_ready": 1}
    gym_row = _lines(output / "development.gym.jsonl")[0]
    assert gym_row["id"] == gym_row["uuid"] == row["id"]
    assert gym_row["agent_ref"] == {"type": "responses_api_agents", "name": "stem_contract_simple_agent"}
    assert gym_row["responses_create_params"] == {"input": [{"role": "user", "content": row["problem"]}]}
    assert gym_row["expected_answer"] == "2"
    assert gym_row["task_contract"]["task_id"] == "a"
    assert "kimi_profile" not in json.dumps(gym_row)
    assert "RL_SUITABLE" not in json.dumps(gym_row)
    assert receipt["training_admitted"] == 0
    forwarded = SimpleAgentRunRequest.model_validate(gym_row).model_dump()
    verifier_row = StemJudgeRunRequest.model_validate(forwarded)
    assert verifier_row.task_contract.task_version == row["version_sha256"]
    assert verifier_row.expected_answer == row["expected_answer"]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"repaired": False}, "repair_version_mismatch"),
        ({"repaired": True}, "repaired_assets_unverified"),
        ({"repaired": True, "assets_verified": True, "reference_sha256": "0" * 64}, "reference_hash_mismatch"),
        ({"repaired": True, "assets_verified": True}, "contract_ready"),
    ],
)
def test_repaired_contract_must_match_reviewed_version_and_reference(
    tmp_path: Path, overrides: dict, expected: str
) -> None:
    row = _row("a", version="deepseek_repaired")
    receipt = prepare(
        _dataset(tmp_path, [row]),
        tmp_path / "prepared",
        development_ids={"a"},
        contracts_path=_reviews(tmp_path, [_review(row, **overrides)]),
    )
    assert receipt["counts"] == {expected: 1}


def test_unmatched_contract_version_and_development_ids_fail_before_output(tmp_path: Path) -> None:
    row = _row("a")
    dataset = _dataset(tmp_path, [row])
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="absent from dataset"):
        prepare(dataset, output, development_ids={"missing"})
    with pytest.raises(ValueError, match="selected task versions"):
        prepare(
            dataset,
            output,
            development_ids=set(),
            contracts_path=_reviews(tmp_path, [_review(row, task_version="other")]),
        )
    assert not output.exists()


def test_rubric_requires_reviewed_criteria_and_preserves_them(tmp_path: Path) -> None:
    row = _row("a")
    review = {
        "contract": _review(row, task_type="rubric", mandatory_criteria=["method"]),
        "rubric": [{"id": "method", "criterion": "Shows the requested method", "points": 1}],
    }
    source = _dataset(tmp_path, [row])
    output = tmp_path / "prepared"
    prepare(source, output, development_ids={"a"}, contracts_path=_reviews(tmp_path, [review]))
    assert _lines(output / "development.gym.jsonl")[0]["rubric"] == review["rubric"]
    review["rubric"][0]["points"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        prepare(source, tmp_path / "invalid", development_ids={"a"}, contracts_path=_reviews(tmp_path, [review]))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"policy_input_revision": "new-prompt"}, "unsupported_policy_input_revision"),
        ({"domain": "biology"}, "domain_mismatch"),
        ({"domain": "Mathematics"}, "contract_ready"),
    ],
)
def test_contract_cannot_claim_a_different_prompt_or_domain(tmp_path: Path, overrides: dict, expected: str) -> None:
    row = _row("a")
    receipt = prepare(
        _dataset(tmp_path, [row]),
        tmp_path / "prepared",
        development_ids={"a"},
        contracts_path=_reviews(tmp_path, [_review(row, **overrides)]),
    )
    assert receipt["counts"] == {expected: 1}


def test_unknown_source_version_is_not_treated_as_keep(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported source version"):
        prepare(
            _dataset(tmp_path, [_row("a", version="unreviewed_draft")]),
            tmp_path / "prepared",
            development_ids={"a"},
        )
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize(
    ("criteria", "mandatory", "message"),
    [
        (
            [
                {"id": "good", "criterion": "Good work", "points": 1},
                {"id": "penalty", "criterion": "Wrong work", "points": -1},
            ],
            ["penalty"],
            "must be positive",
        ),
        (
            [
                {"id": "a", "criterion": "First part", "points": 1e308},
                {"id": "b", "criterion": "Second part", "points": 1e308},
            ],
            [],
            "total must be finite",
        ),
    ],
)
def test_rubric_admission_agrees_with_server_constraints(
    tmp_path: Path, criteria: list[dict], mandatory: list[str], message: str
) -> None:
    row = _row("a")
    review = {
        "contract": _review(row, task_type="rubric", mandatory_criteria=mandatory),
        "rubric": criteria,
    }
    with pytest.raises(ValueError, match=message):
        prepare(
            _dataset(tmp_path, [row]),
            tmp_path / "prepared",
            development_ids={"a"},
            contracts_path=_reviews(tmp_path, [review]),
        )

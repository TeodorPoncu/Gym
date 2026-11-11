# Task-aware grading

The question-bank workflow now uses [shared grading profiles](shared-grading.md)
and automatically prepared metadata. This document describes the optional
explicit-contract mode; the dataset adapter no longer requires or consumes it.

This opt-in implementation extends `jiachengx/stem-judge` at
`78001c5ebc047d8d149f7bed74a5f9be5cb3b796`. The original YAML, prompts,
parsers and no-contract scoring path remain available for fixed-answer baseline
comparisons. Use `configs/stem_contract.yaml` to require contracts on every row.
Its `judge_model` must be a separately configured fixed model service; set
`judge_model_revision` to the exact deployed checkpoint or API model revision.
The server records that assertion, but cannot attest a remote deployment's weights.

`configs/dsv41_flash.yaml` composes the DSV4.1 Flash NVCF model service with
the contract configuration. Supply `STEM_JUDGE_BASE_URL`, `STEM_JUDGE_API_KEY`,
and `STEM_JUDGE_MODEL_REVISION` through the process environment. The revision
must come from the service's deployment identity; its routing alias alone is
not an immutable weight revision. Credentials are never checked into the branch.
For a different gateway such as NVIDIA Inference Hub, set `STEM_JUDGE_MODEL` to
the DSV4.1 Flash identifier advertised by that endpoint's authenticated model
catalog. The default is the private NVCF alias; aliases are not interchangeable
across gateways. Endpoint access and the actual returned model identity still
need live verification.

## Contract boundary

Pass `task_contract` beside `expected_answer` in a Gym row. `TaskContract` in
`contracts.py` is the versioned schema. Contracts are reviewed metadata, not
instructions supplied by the candidate. Required fields are `task_type`,
`task_id`, and `task_version`. Bind answer references with `reference_sha256`
(SHA256 of the exact UTF-8 reference). Review repaired references against the
selected repaired task before setting `assets_verified`.

For example, an answer-only scalar contract can declare `deterministic: numeric`,
`unit: m/s`, and an explicit `absolute_tolerance`. No tolerance is inferred from
a reference's decimal places. Numeric checks accept single decimal/scientific
scalars and identical literal units; unit conversion, expressions, time notation,
collections, symbolic identities and chemistry require semantic judgment or a
future specialized verifier. Choice checks require an explicit label-to-text map.
Deterministic mismatches defer to the judge rather than declaring incorrectness.

`method_required` and `required_components` prohibit deterministic shortcuts.
For proof, derivation, or mechanism tasks, the judge must inspect the required
work in the submitted answer even when the final value matches. A rubric contract
specifies `mandatory_criteria` and `rubric_pass_fraction`. Fractional
`rubric_score` and binary `correctness` are separate fields; positive partial
credit does not imply task success.

Contracts default to `answer_format: free`. Only require boxes when the candidate
actually received that instruction. Preserve historical policy input when
replaying saved answers. References, rubrics, hidden tests and historical labels
belong in verifier metadata and must never be appended to policy input.

## Unresolved grades

The contract path requires a completed judge response containing exactly one JSON
object with `verdict`, `reason_code`, and `evidence`. Rubric judgments additionally
contain `criteria_met` with exactly the requested IDs and JSON boolean values.
Duplicate keys, extra objects, prose, code fences, contradictory rubric results,
and incomplete responses cannot produce a resolved grade.

Unresolved responses carry `verdict: unresolved`, `correctness: null`,
`evaluation_completed: false`, `mask_sample: true`, and a failure reason.
The numeric reward field remains required by Gym's wire schema; consumers must
honor the status fields. This branch does **not** prove NeMoRL exclusion from
losses, group baselines or normalization. Keep it out of optimization until the
selected NeMoRL/Gym pair demonstrates whole-group exclusion for grader failures.

Native code contracts are deliberately unresolved until pinned test assets and
an actual runner are integrated on the designated sandbox cluster. No LLM
substitution supplies native test coverage. Symbolic checking, specialized
chemical validation, reviewed source/domain override registries, caching, and
full fixed-answer replay remain follow-up work.

## Preparation and validation

`prepare.py` prepares membership, split and candidate evaluation artifacts from
the immutable question-bank format using shared profiles. Its coverage report is
an inventory, not an accuracy or training readiness claim. Keep the output and
private data outside public source commits.
Group source duplicates and original/repaired variants before tuning; reserve
the supplied conflict families for development. Unknown semantic duplicates may
still require manual family review.

Run the focused suite from the repository root:

```bash
python -m pytest resources_servers/stem_judge/tests --no-cov
```

Tests use synthetic examples and mocked model responses. They establish parser,
contract, routing and failure behavior only. Before production use, compare
unchanged baseline, contract judge, and deterministic routes on the same saved
answers; independently adjudicate positives and negatives; measure false accepts,
false rejects, unresolved coverage, repeated-input disagreement and cost. The
conflict-selected historical examples are development regressions, not gold or
a population estimate. Run real Gym rollouts and inspect verifier traces before
the small NeMoRL integration run. No live validation result is implied here.

# Shared grading for question-bank data

The question-bank adapter emits ordinary Gym rows with a question, reference,
agent route and provenance. It does not require a contract file or a custom
grading specification for each question.

Compose `configs/question_bank.yaml`, `configs/dsv41_flash.yaml`, and the chosen
policy-model configuration. Supply `STEM_JUDGE_BASE_URL`, `STEM_JUDGE_API_KEY`,
`STEM_JUDGE_MODEL_REVISION`, and optionally `STEM_JUDGE_MODEL` through the runtime
environment. The model alias defaults to the private DSV4.1 Flash NVCF route;
other gateways require the identifier from their own authenticated model catalog.

## Routes and final-answer selection

- `question_bank_math_simple_agent` uses the existing `math_with_judge` server's
  Math-Verify implementation, including its concurrency limit and terminable
  subprocess. A library match resolves correctness; a nonmatch, parse failure or
  library timeout goes to the structured DSV4.1 Flash judge.
- `question_bank_semantic_simple_agent` sends the final answer to the structured
  DSV4.1 Flash judge, using shared equivalence rules.

Both routes select only the **last** `\boxed{...}` in the final assistant
message. The library and the judge receive that same selected answer. Nested
braces and escaped literal braces are handled, and whitespace after `\boxed`
is accepted. A missing, empty or malformed last box fails the required output
format; it never recovers an earlier box. Policy truncation is distinct from a
truncated judge response, which is an unresolved grade.

The existing `math_with_judge` server also now isolates the last box before
calling Math-Verify. Previously, its library path could parse the whole response
even though its judge path received the final box.

Unresolved grades carry `mask_sample: true`, `evaluation_completed: false`,
`correctness: null`, and failure metadata. This server behavior does not establish
that a particular NeMoRL version excludes failures from all losses and group
statistics; that integration still requires a real rollout/training test.

## Generation and grading instructions

The dataset adapter adds this exact instruction once:

> Solve the following problem step by step. Put your final answer, and only your final answer, in \boxed{}.

Gym's `simple_agent` forwards prepared input; there is no universal solver-prompt
append in that agent. Dataset preparation is the explicit insertion point here.
The original question is retained separately for provenance and verification.
The changed prompt is for new generations, and must not be retroactively applied
as an instruction that historical answers received.

`--prompt-overrides` accepts a JSON object with `domains` and `sources` maps of
names to additional generation instructions. For example:

```json
{"domains":{"physics":"Include the units requested by the question."},"sources":{}}
```

These are shared instructions, not per-row annotations. Canonical domain/source
keys come from the exported metadata. References and hidden tests must never be
included in generation instructions. Judge customization is separate: configure
`judge_domain_instructions` and `judge_source_instructions` in the resources
server YAML. Domain additions are applied before source additions. A Math-Verify
match still goes to the judge when generation or grading overrides apply, so a
scalar-equivalence result cannot bypass added obligations.

## Preparing the selected pool

Run from the repository root, using a new output directory:

```bash
python -m resources_servers.stem_judge.prepare \
  --dataset /path/to/questions.jsonl.gz \
  --inventory /path/to/row-grading-inventory.csv.gz \
  --development-ids /path/to/development-ids.json \
  --output-dir /path/to/new-output
```

The inventory is validated against the selected task IDs, versions and row
hashes. Its rubric/construction queue is excluded. Native-test declarations are
also excluded from the text routes, including coding tasks under other domain
labels. Known native/rubric metadata is checked again by the resources server
before any library acceptance. Coding requires its real tests and sandbox runner.

The immutable original pool and family split are preserved. The output contains
separate development/heldout Gym files, a ledger for every selected row, coverage
counts and checksums. Repaired rows retain their selected repaired question and
reference instead of restoring an original reference. Hash binding establishes
which data was used; it does not establish reference correctness.

These are **candidate evaluation inputs**, not a claim of training admission or
judge accuracy. The retained historical contract mode is optional and is not
used by this adapter. Run representative live DSV4.1 judgments and the selected
NeMoRL integration before admitting these routes to optimization.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit grading contracts and conservative, bounded answer checks.

These checks only establish equivalence for a complete, declared answer-only
contract. An unsupported representation or a mismatch defers to the judge; it
does not establish that a submitted answer is incorrect.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, DecimalException, localcontext
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, model_validator


NonEmptyString = Annotated[str, Field(min_length=1, pattern=r"\S")]
NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class TaskContract(BaseModel):
    """Reviewed task requirements passed only to verification, never the policy.

    ``task_version`` identifies the exact task revision. ``assets_verified`` records
    review of references/tests against that revision, including repaired questions;
    it is not a claim that a runner or training integration has been exercised.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1"] = "1"
    task_type: Literal["answer", "rubric", "native"]
    task_id: NonEmptyString
    task_version: NonEmptyString
    reference_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    repaired: bool = False
    assets_verified: bool = False
    domain: NonEmptyString = "unspecified"
    source: NonEmptyString = "unspecified"
    answer_format: Literal["free", "boxed"] = "free"
    required_components: list[NonEmptyString] = Field(default_factory=list)
    method_required: bool = False
    deterministic: Literal["none", "exact", "numeric", "choice"] = "none"
    absolute_tolerance: NonNegativeFinite | None = None
    relative_tolerance: NonNegativeFinite | None = None
    unit: NonEmptyString | None = None
    choice_map: dict[NonEmptyString, NonEmptyString] = Field(default_factory=dict)
    rubric_pass_fraction: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    mandatory_criteria: list[NonEmptyString] = Field(default_factory=list)
    policy_input_revision: NonEmptyString = "original"
    judge_input_revision: NonEmptyString = "contract-v1"

    @model_validator(mode="after")
    def _consistent_requirements(self) -> TaskContract:
        if self.deterministic != "none" and (
            self.task_type != "answer" or self.method_required or self.required_components
        ):
            raise ValueError("deterministic checks require an answer-only contract without method/components")
        if self.deterministic == "choice" and not self.choice_map:
            raise ValueError("choice checking requires an explicit choice_map")
        if self.deterministic in {"exact", "choice"} and (
            self.absolute_tolerance is not None or self.relative_tolerance is not None or self.unit is not None
        ):
            raise ValueError("units and tolerances require numeric checking or semantic judgment")
        if len(self.mandatory_criteria) != len(set(self.mandatory_criteria)):
            raise ValueError("mandatory_criteria must be unique")
        if self.mandatory_criteria and self.task_type != "rubric":
            raise ValueError("mandatory_criteria require a rubric contract")
        if self.unit is not None and self.unit != self.unit.strip():
            raise ValueError("unit must not contain leading or trailing whitespace")
        for label, value in self.choice_map.items():
            if label != label.strip() or value != value.strip():
                raise ValueError("choice labels and values must not contain surrounding whitespace")
            if value in self.choice_map and self.choice_map[value] != value:
                raise ValueError("choice text must not be an ambiguous label for another choice")
        return self


class Judgment(BaseModel):
    """A complete structured judgment; evidence is ordinary text, not executable."""

    model_config = ConfigDict(extra="forbid", strict=True)

    verdict: Literal["correct", "incorrect", "unresolved"]
    reason_code: NonEmptyString
    evidence: NonEmptyString
    criteria_met: dict[NonEmptyString, StrictBool] | None = None


def hash_text(text: str) -> str:
    """Hash exact UTF-8 text without normalizing task content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def parse_judgment(text: str, criteria_ids: list[str] | None = None) -> Judgment | None:
    """Parse exactly one JSON judgment, rejecting ambiguous or incomplete output.

    No extraction from prose/fences is attempted. A rubric verdict must contain all
    and only the requested criterion IDs with real JSON booleans. In answer mode,
    criterion results are disallowed. Completion status is the caller's obligation.
    """
    try:
        raw = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        judgment = Judgment.model_validate(raw)
    except (ValueError, ValidationError, RecursionError):
        return None
    if criteria_ids is None:
        return judgment if judgment.criteria_met is None else None
    if len(criteria_ids) != len(set(criteria_ids)) or not criteria_ids:
        return None
    if judgment.criteria_met is None or set(judgment.criteria_met) != set(criteria_ids):
        return None
    return judgment


_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _scalar(text: str, unit: str | None) -> Decimal | None:
    # Bound parsing/arithmetic even when the server's answer-length cap is off.
    if len(text) > 4096:
        return None
    if unit is not None:
        if not text.endswith(unit):
            return None
        text = text[: -len(unit)].rstrip()
    if _NUMBER.fullmatch(text) is None:
        return None
    try:
        number = Decimal(text)
    except DecimalException:
        return None
    if not number.is_finite() or abs(number.as_tuple().exponent) > 1000:
        return None
    return number


def deterministic_check(contract: TaskContract, candidate: str, reference: str) -> bool | None:
    """Return True only for proven whole-contract equivalence, otherwise None.

    Numeric checks accept one decimal/scientific scalar with the declared literal
    unit. They do not infer units, convert scales, evaluate expressions, normalize
    clock notation, or inspect intermediate values. Tolerances default to zero and
    use ``abs(candidate-reference) <= max(atol, rtol*abs(reference))``.
    """
    if contract.task_type != "answer" or contract.method_required or contract.required_components:
        return None
    if contract.reference_sha256 is not None and hash_text(reference) != contract.reference_sha256:
        return None
    candidate, reference = candidate.strip(), reference.strip()
    if not candidate or not reference:
        return None
    if contract.deterministic == "exact":
        return True if candidate == reference else None
    if contract.deterministic == "choice":
        values = set(contract.choice_map.values())
        lhs = contract.choice_map.get(candidate, candidate if candidate in values else None)
        rhs = contract.choice_map.get(reference, reference if reference in values else None)
        return True if lhs is not None and lhs == rhs else None
    if contract.deterministic != "numeric":
        return None
    lhs, rhs = _scalar(candidate, contract.unit), _scalar(reference, contract.unit)
    if lhs is None or rhs is None:
        return None
    atol = Decimal(str(contract.absolute_tolerance or 0))
    rtol = Decimal(str(contract.relative_tolerance or 0))
    # Enough precision for exact subtraction and tolerance multiplication of all
    # accepted inputs; avoid accepting just-outside-boundary values by rounding.
    with localcontext() as context:
        context.prec = 2 * (4096 + 1000) + 64
        try:
            equivalent = abs(lhs - rhs) <= max(atol, rtol * abs(rhs))
        except DecimalException:
            return None
    return True if equivalent else None

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract grading excludes unusable measurements and keeps binary success explicit."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError
from pytest import mark, raises

from nemo_gym.client import ServerClient
from nemo_gym.verification import JudgeError
from resources_servers.stem_judge.app import RubricCriterion, StemJudgeConfig, StemJudgeResourcesServer
from resources_servers.stem_judge.contracts import TaskContract, hash_text
from resources_servers.stem_judge.tests.test_app import _msg, _request, _response, _split


SERVER_DIR = Path(__file__).resolve().parents[1]


def server(**overrides) -> StemJudgeResourcesServer:
    config = {
        "host": "127.0.0.1",
        "port": 8080,
        "entrypoint": "app.py",
        "judge_model_server": {"name": "fixed_judge", "type": "responses_api_models"},
        "judge_model_revision": "test-model-revision",
        "judge_responses_create_params": {"input": [], "temperature": 0},
        "answer_prompt_template_fpath": str(SERVER_DIR / "prompt_templates/stem_v11.txt"),
        "rubric_prompt_template_fpath": str(SERVER_DIR / "prompt_templates/stem_rubric.txt"),
        **overrides,
    }
    return StemJudgeResourcesServer(config=StemJudgeConfig(**config), server_client=MagicMock(spec=ServerClient))


def contract(**overrides) -> TaskContract:
    return TaskContract(
        **{
            "task_id": "q1",
            "task_version": "v1",
            "task_type": "answer",
            "reference_sha256": hash_text("2"),
            **overrides,
        }
    )


def request(answer="2", *, task_contract=None, **overrides):
    response = _split(answer)
    response.status = "completed"
    return _request(response, task_contract=task_contract or contract(), **overrides)


def judge(monkeypatch, text=None, *, verdict="correct", criteria_met=None, status="completed"):
    payload = {"verdict": verdict, "reason_code": "reviewed", "evidence": "Submitted work meets the obligation."}
    if criteria_met is not None:
        payload["criteria_met"] = criteria_met
    response = _response(_msg(text if text is not None else json.dumps(payload)))
    response.status = status
    mock = AsyncMock(return_value=response)
    monkeypatch.setattr("resources_servers.stem_judge.app.call_judge", mock)
    return mock, response


@mark.asyncio
async def test_exact_answer_is_proven_without_judge(monkeypatch):
    call, _ = judge(monkeypatch)
    result = await server().verify(request(task_contract=contract(deterministic="exact")))
    assert (result.verdict, result.correctness, result.reward) == ("correct", True, 1)
    assert result.evaluation_completed and not result.mask_sample
    assert result.grading_provenance["route"] == "exact"
    call.assert_not_awaited()


@mark.asyncio
async def test_method_obligation_is_not_bypassed_by_reference_match(monkeypatch):
    call, _ = judge(monkeypatch, verdict="incorrect")
    result = await server().verify(
        request(task_contract=contract(method_required=True, required_components=["proof"]))
    )
    assert result.verdict == "incorrect" and result.reward == 0 and not result.mask_sample
    payload = json.loads(call.await_args.kwargs["json"].input[-1].content)
    assert payload["candidate"] == payload["reference"] == "2"
    assert payload["contract"]["required_components"] == ["proof"]
    assert result.grading_provenance["judge_request"]["input"][0]["role"] == "system"


@mark.asyncio
@mark.parametrize(
    "text",
    [
        "Judgement: yesterday",
        '{"verdict":"correct","verdict":"incorrect","reason_code":"x","evidence":"x"}',
        '{"verdict":"correct","reason_code":"x","evidence":"x"}\n{"verdict":',
        '{"verdict":"correct","reason_code":"x","evidence":"x","extra":true}',
    ],
)
async def test_malformed_verdict_is_unresolved_and_sidecar_routed(monkeypatch, text):
    judge(monkeypatch, text)
    result = await server().verify(request())
    assert result.verdict == "unresolved" and result.correctness is None
    assert not result.evaluation_completed and result.mask_sample
    payload = result.model_dump()
    assert payload["instance_config"]["mask_sample"]
    assert payload["_ng_failure_class"] == "judge_failed"
    assert result.grading_status == "judge_unparsed"
    assert len(result.judge_evaluations) == 1


@mark.asyncio
@mark.parametrize("status", [None, "incomplete", "failed", "in_progress"])
async def test_judge_must_explicitly_complete(monkeypatch, status):
    judge(monkeypatch, status=status)
    result = await server().verify(request())
    assert result.grading_status == "judge_incomplete" and result.mask_sample


@mark.asyncio
async def test_judge_incomplete_details_override_completed_status(monkeypatch):
    _, response = judge(monkeypatch)
    response.incomplete_details = _split("2", truncated=True).incomplete_details
    result = await server().verify(request())
    assert result.grading_status == "judge_incomplete" and result.mask_sample


@mark.asyncio
async def test_transport_failure_keeps_contract_outcome(monkeypatch):
    call, _ = judge(monkeypatch)
    call.side_effect = JudgeError("endpoint unavailable")
    result = await server().verify(request())
    assert result.grading_status == "judge_transport_failed" and result.mask_sample
    assert result.verdict == "unresolved" and result.correctness is None
    assert result.failure_kind == "judge_failed"
    assert call.await_args.kwargs["max_attempts"] == 2
    assert call.await_args.kwargs["timeout"].total == 300


@mark.asyncio
@mark.parametrize(
    "row, expected",
    [
        ({"task_contract": contract(task_type="native")}, "native_runner_unavailable"),
        ({"task_contract": contract(repaired=True)}, "repaired_assets_unverified"),
        ({"task_contract": contract(reference_sha256=hash_text("3"))}, "reference_hash_mismatch"),
        ({"expected": ""}, "missing_reference"),
        ({"task_contract": contract(task_type="rubric")}, "rubric_contract_mismatch"),
        ({"answer_format": "boxed"}, "conflicting_answer_format"),
    ],
)
async def test_unsupported_task_is_never_a_numeric_failure(monkeypatch, row, expected):
    call, _ = judge(monkeypatch)
    result = await server().verify(request(**row))
    assert result.grading_status == expected and result.mask_sample
    assert result.verdict == "unresolved" and not result.evaluation_completed
    call.assert_not_awaited()


@mark.asyncio
async def test_required_contract_missing(monkeypatch):
    call, _ = judge(monkeypatch)
    result = await server(require_task_contract=True).verify(_request(_split("2")))
    assert result.grading_status == "missing_task_contract" and result.mask_sample
    call.assert_not_awaited()


@mark.asyncio
async def test_unfixed_judge_is_not_called(monkeypatch):
    call, _ = judge(monkeypatch)
    result = await server(judge_model_revision=None).verify(request())
    assert result.grading_status == "fixed_judge_not_configured" and result.mask_sample
    call.assert_not_awaited()


@mark.asyncio
async def test_rubric_partial_credit_is_separate_from_correctness(monkeypatch):
    judge(monkeypatch, verdict="incorrect", criteria_met={"answer": True, "proof": False})
    result = await server().verify(
        request(
            task_contract=contract(task_type="rubric", rubric_pass_fraction=0.5, mandatory_criteria=["proof"]),
            rubric=[
                {"id": "answer", "points": 1, "criterion": "Gives the final value"},
                {"id": "proof", "points": 1, "criterion": "Proves the value"},
            ],
        )
    )
    assert result.rubric_score == result.reward == 0.5
    assert result.verdict == "incorrect" and result.correctness is False
    assert result.evaluation_completed and not result.mask_sample


@mark.asyncio
async def test_rubric_verdict_must_agree_with_pass_predicate(monkeypatch):
    judge(monkeypatch, verdict="correct", criteria_met={"proof": False})
    result = await server().verify(
        request(
            task_contract=contract(task_type="rubric", mandatory_criteria=["proof"]),
            rubric=[{"id": "proof", "points": 1, "criterion": "Proves the value"}],
        )
    )
    assert result.grading_status == "judge_conflicting" and result.mask_sample
    assert result.rubric_score is None


@mark.asyncio
async def test_list_form_question_and_split_judge_content(monkeypatch):
    call, response = judge(monkeypatch)
    text = response.output[0].content[0].text
    first, second = text.split(",", 1)
    response.output[0].content = [_msg(first + ",").content[0], _msg(second).content[0]]
    body = request()
    body.responses_create_params = body.responses_create_params.model_validate(
        {"input": [{"role": "user", "content": [{"type": "input_text", "text": "Compute 1+1"}]}]}
    )
    result = await server().verify(body)
    assert result.correctness is True
    assert json.loads(call.await_args.kwargs["json"].input[-1].content)["question"] == "Compute 1+1"


@mark.asyncio
async def test_policy_and_judge_truncation_are_distinct(monkeypatch):
    call, _ = judge(monkeypatch)
    body = request()
    body.response.incomplete_details = _split("2", truncated=True).incomplete_details
    result = await server().verify(body)
    assert result.grading_status == "policy_truncated" and result.correctness is False
    assert not result.mask_sample and result.evaluation_completed
    call.assert_not_awaited()


@mark.parametrize("points", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_rubric_weights_are_rejected(points):
    with raises(ValidationError):
        RubricCriterion(id="r1", criterion="work", points=points)


def test_nonpositive_concurrency_is_rejected():
    with raises(ValidationError):
        server(judge_endpoint_max_concurrency=0)


@mark.asyncio
async def test_conflicting_judgment_messages_are_unresolved(monkeypatch):
    _, response = judge(monkeypatch)
    response.output.append(_msg('{"verdict":"incorrect","reason_code":"wrong","evidence":"Wrong value"}'))
    result = await server().verify(request())
    assert result.grading_status == "judge_unparsed" and result.mask_sample


@mark.asyncio
async def test_reverification_clears_stale_failure_routing(monkeypatch):
    judge(monkeypatch)
    body = request()
    body = type(body).model_validate(
        {
            **body.model_dump(),
            "instance_config": {"mask_sample": True, "other": 3},
            "_ng_failure_class": "judge_failed",
            "_ng_failure_judge_error": "previous failure",
            "failure_kind": "judge_failed",
        }
    )
    result = await server().verify(body)
    assert result.correctness and result.evaluation_completed and not result.mask_sample
    assert "_ng_failure_class" not in result.model_dump()
    assert result.model_dump()["instance_config"] == {"other": 3}
    assert result.failure_kind is None


def test_overflowing_rubric_total_is_rejected():
    with raises(ValidationError, match="absolute point total"):
        request(rubric=[{"id": str(i), "points": 1e308, "criterion": "work"} for i in range(2)])


@mark.asyncio
async def test_contract_call_uses_shared_http_response_validation():
    instance = server()
    response = _response(_msg('{"verdict":"incorrect","reason_code":"wrong_value","evidence":"Value differs"}'))
    response.status = "completed"
    http_response = MagicMock(status=200)
    http_response.read = AsyncMock(return_value=response.model_dump_json().encode())
    instance.server_client.post = AsyncMock(return_value=http_response)
    result = await instance.verify(request())
    assert result.verdict == "incorrect" and not result.mask_sample
    assert instance.server_client.post.await_count == 1
    assert len(result.judge_evaluations) == 1


@mark.asyncio
async def test_contract_call_invalid_response_schema_is_unresolved():
    instance = server()
    http_response = MagicMock(status=200)
    http_response.read = AsyncMock(return_value=b'{"unexpected":"response"}')
    instance.server_client.post = AsyncMock(return_value=http_response)
    result = await instance.verify(request())
    assert result.verdict == "unresolved" and result.mask_sample
    assert result.grading_status == "judge_transport_failed"
    assert instance.server_client.post.await_count == 1


@mark.asyncio
@mark.parametrize(
    "kwargs,status",
    [
        ({"task_contract": contract(reference_sha256=None)}, "missing_reference_hash"),
        ({"uuid": "different-task"}, "task_id_mismatch"),
    ],
)
async def test_contract_identity_binding_is_required(monkeypatch, kwargs, status):
    call, _ = judge(monkeypatch)
    result = await server().verify(request(**kwargs))
    assert result.mask_sample and result.grading_status == status
    call.assert_not_awaited()

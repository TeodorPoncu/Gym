# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.math_with_judge.answer_extraction import extract_last_boxed_answer
from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r"Earlier \boxed{2}; final \boxed{3}", "3"),
        (r"Final \boxed {\frac{1}{2}}", r"\frac{1}{2}"),
        ("Final \\boxed\n{3}", "3"),
        (r"\boxed{\{1,2\}}", r"\{1,2\}"),
        (r"\boxed{\frac{a+{b}}{c}}", r"\frac{a+{b}}{c}"),
        (r"\boxed{2} then \boxed{3", None),
        (r"\boxed{2} then \boxed {\frac{3}{4}", None),
        (r"\boxed{2} then \boxed{}", None),
        (r"\boxed{2} then \boxed{ }", None),
        (r"\boxed{2} then \boxed", None),
        (r"\boxed{2} then \boxed no argument", None),
        (r"\boxed{2} then \\boxed{3}", "2"),
        (r"\\boxed{3}", None),
        (r"\boxedvalue{3}", None),
        ("Nothing boxed", None),
    ],
)
def test_last_box_selection(response: str, expected: str | None) -> None:
    assert extract_last_boxed_answer(response) == expected


def make_server(*, tolerant: bool = False) -> LibraryJudgeMathResourcesServer:
    return LibraryJudgeMathResourcesServer(
        config=LibraryJudgeMathResourcesServerConfig(
            host="127.0.0.1",
            port=1,
            name="test_math",
            entrypoint="",
            should_use_judge=False,
            format_tolerant_answer_extraction=tolerant,
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.parametrize("tolerant", [False, True])
async def test_library_receives_only_last_box(tolerant: bool) -> None:
    server = make_server(tolerant=tolerant)
    check = AsyncMock(return_value=(0.0, "3"))
    server._verify_answer_with_library_async = check
    result = await server._verify_answer("question", "2", r"Earlier \boxed{2}, final \boxed {3}")
    check.assert_awaited_once_with("2", r"\boxed{3}")
    assert result == (0.0, "3", 0.0, None)


@pytest.mark.parametrize("tolerant", [False, True])
async def test_malformed_last_box_never_recovers_earlier_box(tolerant: bool) -> None:
    server = make_server(tolerant=tolerant)
    check = AsyncMock(return_value=(1.0, "2"))
    server._verify_answer_with_library_async = check
    assert await server._verify_answer("question", "2", r"Earlier \boxed{2}, final \boxed{{3") == (
        0.0,
        None,
        0.0,
        None,
    )
    check.assert_not_awaited()


@pytest.mark.parametrize(("expected", "reward"), [("2", 0.0), ("3", 1.0)])
async def test_actual_math_library_does_not_accept_earlier_correct_answer(expected: str, reward: float) -> None:
    server = make_server()
    actual, extracted, library, judge = await server._verify_answer(
        "question", expected, r"An intermediate value is $2$. Earlier \boxed{2}, final \boxed{3}."
    )
    assert actual == library == reward
    assert extracted == "3"
    assert judge is None

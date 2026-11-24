"""Task for the checkpoint branching end-to-end tests (``test_branch_e2e.py``).

It lives in its own file because a branch reloads the task, by name, from
the log it branches.

The task makes a real model's turns predictable without scripting them: each
``advance`` call returns the token the next call needs, so the model takes
one turn per step and cannot batch or reorder them. Every step writes a file
under ``/work``, the one sandbox path the task checkpoints. The checkpoint
hooks record a listing of ``/work`` at each checkpoint and at each resume, so
a test can tell which checkpoint's files a run started from without relying
on what the model says.
"""

from __future__ import annotations

import os
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import TaskState
from inspect_ai.tool import Tool, ToolError, tool
from inspect_ai.util import CheckpointConfig, TurnInterval, sandbox

WORK_DIR = "/work"

# TOKENS[k] admits step k + 1; the last one is the answer to submit
TOKENS = ["amber", "birch", "cedar", "dune", "ember", "fjord"]
STEPS = len(TOKENS) - 1

FILES_AT_CHECKPOINTS = "files_at_checkpoints"
"""Store key: the ``/work`` listing at each checkpoint, in checkpoint order."""

FILES_AT_RESUME = "files_at_resume"
"""Store key: the ``/work`` listing found by the latest resume."""

FAIL_AT_STEP_ENV = "INSPECT_TEST_BRANCH_FAIL_AT_STEP"
"""Step at which ``advance`` raises, failing the sample (unset: never)."""


async def work_listing() -> str:
    """Name and content hash of every file under ``/work``."""
    result = await sandbox().exec(
        ["sh", "-c", f"cd {WORK_DIR} && sha256sum * 2>/dev/null | sort -k2"]
    )
    return result.stdout


@tool
def advance() -> Tool:
    async def execute(token: str) -> str:
        """Record the next step.

        Args:
            token: Token for the next step.
        """
        if token not in TOKENS[:STEPS]:
            raise ToolError(f"{token!r} is not the token for the next step.")
        step = TOKENS.index(token) + 1
        if os.environ.get(FAIL_AT_STEP_ENV) == str(step):
            raise RuntimeError(f"step {step} failed")
        await sandbox().write_file(f"{WORK_DIR}/step-{step}.txt", token)
        following = TOKENS[step]
        if step == STEPS:
            return f"Step {step} recorded. All steps are done: submit {following!r}."
        return f"Step {step} recorded. The token for the next step is {following!r}."

    return execute


async def _record_files_at_checkpoint(state: TaskState) -> None:
    listings: list[str] = state.store.get(FILES_AT_CHECKPOINTS, [])
    state.store.set(FILES_AT_CHECKPOINTS, [*listings, await work_listing()])


async def _record_files_at_resume(
    state: TaskState, attempt: Literal["initial", "resume", "resume_for_scoring"]
) -> None:
    state.store.set(FILES_AT_RESUME, await work_listing())


@task
def branch_steps() -> Task:
    return Task(
        dataset=[
            Sample(
                id="steps",
                input=(
                    f"Record {STEPS} steps with the advance tool, one call at a "
                    f"time. The token for the first step is {TOKENS[0]!r}; each "
                    "call returns the token for the next. When all steps are "
                    "recorded, submit the word you are told to submit."
                ),
                target=TOKENS[STEPS],
            )
        ],
        solver=react(tools=[advance()]),
        scorer=includes(),
        sandbox=("docker", "branch_compose.yaml"),
        checkpoint=CheckpointConfig(
            trigger=TurnInterval(every=1),
            sandbox_paths={"default": [WORK_DIR]},
        ),
        on_checkpoint=_record_files_at_checkpoint,
        on_resume=_record_files_at_resume,
        message_limit=40,
    )

"""How a branch of an eval is resolved into a task run.

These tests resolve the task without running it. ``test_branch_e2e.py``
covers a branch end to end, and ``test_resume_copy.py`` its checkpoint copy.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from inspect_ai import Task
from inspect_ai._eval.loader import resolve_previous_task
from inspect_ai._eval.task import PreviousTask
from inspect_ai._eval.task.resolved import ResolvedTask
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.dataset import Sample
from inspect_ai.log import (
    EvalBranch,
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalSpec,
    EvalStats,
)
from inspect_ai.model import Model, ModelUsage, get_model
from inspect_ai.util import CheckpointConfig

MODEL = "openrouter/anthropic/claude-sonnet-5"
OTHER_MODEL = "openrouter/deepseek/deepseek-v3.2"
USAGE = {MODEL: ModelUsage(input_tokens=7, output_tokens=3, total_tokens=10)}
BRANCH = EvalBranch(
    log="parent.eval",
    eval_id="parent-eval-id",
    sample_id="s1",
    epoch=1,
    checkpoint_id=2,
)


@pytest.fixture(autouse=True)
def _model_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The models are only constructed, never called."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-key")


def _parent_log(tmp_path: Path, branch: EvalBranch | None = None) -> EvalLog:
    return EvalLog(
        location=str(tmp_path / "parent.eval"),
        eval=EvalSpec(
            created=datetime.now(timezone.utc).isoformat(),
            task="task",
            model=MODEL,
            dataset=EvalDataset(),
            config=EvalConfig(),
            branch=branch,
        ),
        stats=EvalStats(model_usage=USAGE),
    )


def _resolve(
    log: EvalLog,
    *,
    task: Task | None = None,
    branch: EvalBranch | None = None,
    model: Model | None = None,
) -> ResolvedTask:
    task = task or Task(dataset=[Sample(id="s1", input="input")])
    return resolve_previous_task(
        task,
        {},
        get_model(MODEL),
        None,
        PreviousTask(
            id="task-id",
            task=task,
            task_args={},
            model=model,
            model_roles=None,
            sandbox=None,
            log=log,
            log_info=None,
            branch=branch,
        ),
        0,
        CheckpointConfig(),
    )


async def test_branch_is_a_new_run_of_the_branched_sample(tmp_path: Path) -> None:
    """It reuses no sample and carries no usage of the eval it was branched from."""
    log = _parent_log(tmp_path)

    resolved = _resolve(log, branch=BRANCH)
    source = resolved.sample_source

    assert resolved.branch == BRANCH
    assert resolved.initial_model_usage is None
    assert resolved.initial_role_usage is None
    assert source is not None
    assert source.branch == BRANCH
    assert source.seed is None
    assert await source.lookup("s1", 1) is None
    assert source.prior_checkpoints_dir == str(tmp_path / "parent.checkpoints")


def test_retry_carries_the_usage_of_the_eval_it_retries(tmp_path: Path) -> None:
    resolved = _resolve(_parent_log(tmp_path))

    assert resolved.branch is None
    assert resolved.initial_model_usage == USAGE
    assert resolved.sample_source is not None
    assert resolved.sample_source.branch is None


def test_retry_of_a_branch_is_still_a_branch(tmp_path: Path) -> None:
    """The retry's log keeps the pointer, and the retry starts no new branch."""
    resolved = _resolve(_parent_log(tmp_path, branch=BRANCH))

    assert resolved.branch == BRANCH
    assert resolved.sample_source is not None
    assert resolved.sample_source.branch is None


def test_branch_of_a_task_that_disables_checkpointing_is_refused(
    tmp_path: Path,
) -> None:
    task = Task(dataset=[Sample(id="s1", input="input")], checkpoint=False)

    with pytest.raises(PrerequisiteError, match="disables checkpointing"):
        _resolve(_parent_log(tmp_path), task=task, branch=BRANCH)


def test_branch_model_outranks_the_task_model(tmp_path: Path) -> None:
    task = Task(dataset=[Sample(id="s1", input="input")], model=MODEL)
    other = get_model(OTHER_MODEL)

    resolved = _resolve(_parent_log(tmp_path), task=task, branch=BRANCH, model=other)

    assert resolved.model is other

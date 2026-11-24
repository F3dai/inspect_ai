"""End-to-end tests for branching an eval from a checkpoint (``eval_branch``).

Every eval here runs a real model (through OpenRouter) in a Docker sandbox.
The task (``branch_task.py``) makes the model take one tool call per turn and
records what the sandbox held at each checkpoint and at each resume, so the
assertions compare recorded state rather than model output: a branch from
checkpoint N must find the files the eval it branched from had at N, and
its conversation must begin with the one that eval's next model call was
given.

One eval of two epochs is shared by the tests that branch from it.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest
from click.testing import CliRunner
from test_helpers.utils import skip_if_no_docker, skip_if_no_openrouter

from checkpoint.branch_task import (
    FAIL_AT_STEP_ENV,
    FILES_AT_CHECKPOINTS,
    FILES_AT_RESUME,
)
from inspect_ai import eval, eval_branch, eval_retry
from inspect_ai._cli.eval import eval_branch_command
from inspect_ai._util.constants import EVAL_SET_ID_FILE
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.event import InfoEvent, ModelEvent, SpanBeginEvent
from inspect_ai.event._checkpoint import CheckpointEvent
from inspect_ai.log import EvalLog, EvalSample, read_eval_log
from inspect_ai.model import ChatMessage
from inspect_ai.util._checkpoint._layout import (
    eval_checkpoints_dir,
    sample_checkpoints_dir,
)

TASK = f"{Path(__file__).parent / 'branch_task.py'}@branch_steps"
SAMPLE_ID = "steps"
MODEL = "openrouter/anthropic/claude-sonnet-5"
OTHER_MODEL = "openrouter/deepseek/deepseek-v3.2"
MESSAGE = "Before the next step, say in one sentence which steps are recorded."


class Run(NamedTuple):
    """A finished eval and its samples by epoch (attachments resolved)."""

    log: EvalLog
    samples: dict[int, EvalSample]

    @property
    def sample(self) -> EvalSample:
        [sample] = self.samples.values()
        return sample


def _run(log: EvalLog) -> Run:
    samples = read_eval_log(log.location, resolve_attachments=True).samples or []
    return Run(log, {sample.epoch: sample for sample in samples})


def _branch(
    source: Run,
    checkpoint_id: int,
    log_dir: Path,
    *,
    epoch: int = 1,
    model: str | None = None,
    message: str | None = None,
) -> Run:
    return _run(
        eval_branch(
            source.log.location,
            sample_id=SAMPLE_ID,
            checkpoint_id=checkpoint_id,
            epoch=epoch,
            model=model,
            message=message,
            log_dir=str(log_dir),
            display="none",
        )
    )


def _files_at_checkpoint(sample: EvalSample, checkpoint_id: int) -> str:
    listings: list[str] = sample.store[FILES_AT_CHECKPOINTS]
    return listings[checkpoint_id - 1]


def _conversation_at_checkpoint(
    sample: EvalSample, checkpoint_id: int
) -> list[ChatMessage]:
    """The conversation a checkpoint holds: the input of the model call after it."""
    calls = [event for event in sample.events if isinstance(event, ModelEvent)]
    return calls[checkpoint_id].input


def _checkpoint_ids(sample: EvalSample) -> list[int]:
    return [e.checkpoint_id for e in sample.events if isinstance(e, CheckpointEvent)]


def _events_since_last_resume(sample: EvalSample) -> list[object]:
    resumes = [
        index
        for index, event in enumerate(sample.events)
        if isinstance(event, InfoEvent)
        and event.source == "checkpoint"
        and isinstance(event.data, dict)
        and event.data.get("event") == "resume"
    ]
    return list(sample.events[resumes[-1] :])


def _model_calls_since_last_resume(sample: EvalSample) -> list[ModelEvent]:
    return [e for e in _events_since_last_resume(sample) if isinstance(e, ModelEvent)]


def _ids(messages: list[ChatMessage]) -> list[str | None]:
    return [message.id for message in messages]


def _checkpoint_files(run: Run, epoch: int = 1) -> dict[str, int]:
    """Name and size of every file in a sample's checkpoints dir, bar its working dir."""
    root = Path(
        sample_checkpoints_dir(
            eval_checkpoints_dir(run.log.location, None), SAMPLE_ID, epoch
        )
    )
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).parts[0] != "context"
    }


@pytest.fixture(scope="module")
def parent(tmp_path_factory: pytest.TempPathFactory) -> Run:
    [log] = eval(
        TASK,
        model=MODEL,
        epochs=2,
        log_dir=str(tmp_path_factory.mktemp("parent")),
        display="none",
    )
    assert log.status == "success", log.error
    return _run(log)


@pytest.fixture(scope="module")
def branch(parent: Run, tmp_path_factory: pytest.TempPathFactory) -> Run:
    """A branch of the parent's first epoch at checkpoint 2, on the same model."""
    files_before = _checkpoint_files(parent)
    run = _branch(parent, 2, tmp_path_factory.mktemp("branch"))
    assert run.log.status == "success", run.log.error
    assert _checkpoint_files(parent) == files_before
    return run


@skip_if_no_openrouter
@skip_if_no_docker
@pytest.mark.slow
def test_branch_starts_from_the_checkpoint(parent: Run, branch: Run) -> None:
    source, sample = parent.samples[1], branch.sample

    # the sandbox holds the files of checkpoint 2, not of a later one
    assert sample.store[FILES_AT_RESUME] == _files_at_checkpoint(source, 2)
    assert sample.store[FILES_AT_RESUME] != _files_at_checkpoint(source, 3)

    # the conversation continues the one checkpoint 2 holds
    conversation = _conversation_at_checkpoint(source, 2)
    assert _ids(sample.messages[: len(conversation)]) == _ids(conversation)
    assert len(sample.messages) > len(conversation)

    # the branch's history is the parent's up to the checkpoint, then its own
    assert _checkpoint_ids(sample) == list(range(1, len(_checkpoint_ids(sample)) + 1))
    assert len(_checkpoint_ids(sample)) > 2
    [restore] = [
        e
        for e in sample.events
        if isinstance(e, SpanBeginEvent) and e.type == "prior_run"
    ]
    assert restore.name == "checkpoint restore 1"
    assert all(call.model == MODEL for call in _model_calls_since_last_resume(sample))


@skip_if_no_openrouter
@skip_if_no_docker
@pytest.mark.slow
def test_branch_log_records_what_it_was_branched_from(parent: Run, branch: Run) -> None:
    pointer = branch.log.eval.branch
    assert pointer is not None
    assert pointer.eval_id == parent.log.eval.eval_id
    assert Path(pointer.log) == Path(parent.log.location).resolve()
    assert (pointer.sample_id, pointer.epoch, pointer.checkpoint_id) == (
        SAMPLE_ID,
        1,
        2,
    )
    assert pointer.message is None

    # a branch is a new task run, not another attempt of the parent's
    assert branch.log.eval.task_id != parent.log.eval.task_id
    assert branch.sample.uuid != parent.samples[1].uuid


@skip_if_no_openrouter
@skip_if_no_docker
@pytest.mark.slow
def test_branch_continues_with_another_model(parent: Run, tmp_path: Path) -> None:
    """The second epoch is branched, so the source dir is not the branch's own name."""
    run = _branch(parent, 3, tmp_path, epoch=2, model=OTHER_MODEL)
    source, sample = parent.samples[2], run.sample

    assert run.log.status == "success", run.log.error
    assert run.log.eval.model == OTHER_MODEL
    assert sample.store[FILES_AT_RESUME] == _files_at_checkpoint(source, 3)
    conversation = _conversation_at_checkpoint(source, 3)
    assert _ids(sample.messages[: len(conversation)]) == _ids(conversation)

    calls = _model_calls_since_last_resume(sample)
    assert calls and all(call.model == OTHER_MODEL for call in calls)
    assert set(sample.model_usage) == {MODEL, OTHER_MODEL}


@skip_if_no_openrouter
@skip_if_no_docker
@pytest.mark.slow
def test_branch_message_joins_the_conversation(parent: Run, tmp_path: Path) -> None:
    run = _branch(parent, 2, tmp_path, message=MESSAGE)
    sample = run.sample

    assert run.log.status == "success", run.log.error
    assert run.log.eval.branch is not None
    assert run.log.eval.branch.message == MESSAGE

    # the first model call sees checkpoint 2's conversation, then the message
    conversation = _conversation_at_checkpoint(parent.samples[1], 2)
    first_input = _model_calls_since_last_resume(sample)[0].input
    assert _ids(first_input[:-1]) == _ids(conversation)
    assert first_input[-1].role == "user"
    assert first_input[-1].text == MESSAGE
    assert first_input[-1].source == "operator"
    assert [m.text for m in sample.messages].count(MESSAGE) == 1


@skip_if_no_openrouter
@skip_if_no_docker
@pytest.mark.slow
@pytest.mark.parametrize(
    "steps_before_failure",
    [
        pytest.param(0, id="before-the-branch-commits-a-checkpoint"),
        pytest.param(1, id="after-the-branch-commits-a-checkpoint"),
    ],
)
def test_retry_of_a_failed_branch_holds_its_message_once(
    parent: Run,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps_before_failure: int,
) -> None:
    """A retry adds the message again only if no checkpoint of the branch holds it.

    The branch's first checkpoint holds the message. A branch that failed
    before committing it is retried from the branch checkpoint, which does
    not; one that failed later is retried from a checkpoint that does.
    """
    steps_at_branch = len(_files_at_checkpoint(parent.samples[1], 2).splitlines())
    monkeypatch.setenv(
        FAIL_AT_STEP_ENV, str(steps_at_branch + 1 + steps_before_failure)
    )
    failed = _branch(parent, 2, tmp_path, message=MESSAGE)
    assert failed.log.status == "error"
    assert len(_checkpoint_ids(failed.sample)) == 2 + steps_before_failure

    monkeypatch.delenv(FAIL_AT_STEP_ENV)
    [retried_log] = eval_retry(
        failed.log.location, log_dir=str(tmp_path), display="none"
    )
    retried = _run(retried_log)

    assert retried.log.status == "success", retried.log.error
    assert retried.log.eval.branch == failed.log.eval.branch
    assert [m.text for m in retried.sample.messages].count(MESSAGE) == 1


@skip_if_no_openrouter
@skip_if_no_docker
@pytest.mark.slow
def test_branch_of_a_branch(parent: Run, branch: Run, tmp_path: Path) -> None:
    """A branch is branched from a checkpoint it committed itself."""
    run = _branch(branch, 4, tmp_path)
    sample = run.sample

    assert run.log.status == "success", run.log.error
    assert run.log.eval.branch is not None
    assert run.log.eval.branch.eval_id == branch.log.eval.eval_id
    assert run.log.eval.branch.eval_id != parent.log.eval.eval_id
    assert sample.store[FILES_AT_RESUME] == _files_at_checkpoint(branch.sample, 4)
    assert _checkpoint_ids(sample) == list(range(1, len(_checkpoint_ids(sample)) + 1))
    assert [
        e.name
        for e in sample.events
        if isinstance(e, SpanBeginEvent) and e.type == "prior_run"
    ] == ["checkpoint restore 1", "checkpoint restore 2"]


@skip_if_no_openrouter
@skip_if_no_docker
@pytest.mark.slow
def test_branch_refuses_a_checkpoint_it_cannot_start_from(
    parent: Run, tmp_path: Path
) -> None:
    """The last checkpoint is taken after the agent completes: nothing is left to run."""
    completed = _checkpoint_ids(parent.samples[1])[-1]

    with pytest.raises(PrerequisiteError, match="Checkpoints a branch can start from"):
        _branch(parent, completed, tmp_path)

    assert not list(tmp_path.iterdir())


@skip_if_no_openrouter
@skip_if_no_docker
@pytest.mark.slow
def test_branch_refuses_an_eval_set_log_dir(parent: Run, tmp_path: Path) -> None:
    (tmp_path / EVAL_SET_ID_FILE).write_text("eval-set-id")

    with pytest.raises(PrerequisiteError, match="eval set"):
        _branch(parent, 2, tmp_path)


@skip_if_no_openrouter
@skip_if_no_docker
@pytest.mark.slow
def test_cli_lists_the_checkpoints_a_branch_can_start_from(parent: Run) -> None:
    result = CliRunner().invoke(eval_branch_command, [parent.log.location])

    assert result.exit_code == 0, result.output
    rows = [line for line in result.output.splitlines() if SAMPLE_ID in line]
    branchable = sum(len(_checkpoint_ids(s)) - 1 for s in parent.samples.values())
    assert len(rows) == branchable

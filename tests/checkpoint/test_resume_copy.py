"""Tests for the retry startup copy (``_resume_copy``).

Local-fs coverage of ``copy_resume_payloads``: whole-attempt
replication and failure semantics. Interrupt safety lives one level
up — the copy runs before the destination log's first write, so a
failed copy means no log and the next retry falls back — which is why
these tests only assert copy behavior. The s3 legs of the underlying
copy helpers are covered in ``test_fs_copy_s3.py``.

Also covers the branch copy (``copy_branch_payload``) and the check of
which checkpoints a branch can start from (``branchable_checkpoint``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

import inspect_ai.util._checkpoint._resume_copy as resume_copy
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.log import EvalBranch
from inspect_ai.util._checkpoint._layout.sample_checkpoints_dir import (
    scan_latest_committed_id,
)
from inspect_ai.util._checkpoint._layout.schemas import Checkpoint, SnapshotDetails
from inspect_ai.util._checkpoint._resume_copy import (
    branchable_checkpoint,
    branchable_checkpoints,
    copy_branch_payload,
    copy_resume_payloads,
)
from inspect_ai.util._checkpoint._triggers import CheckpointTriggerKind
from inspect_ai.util._checkpoint.resume import resolve_resume_checkpoint


def _checkpoint_json(
    checkpoint_id: int, trigger: CheckpointTriggerKind = "turn"
) -> str:
    return Checkpoint(
        checkpoint_id=checkpoint_id,
        trigger=trigger,
        turn=checkpoint_id,
        created_at=datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc),
        duration_ms=10,
        size_bytes=100 + checkpoint_id,
        host=SnapshotDetails(
            snapshot_id=f"snap-{checkpoint_id}",
            size_bytes=100 + checkpoint_id,
            duration_ms=10,
        ),
        sandboxes={},
    ).model_dump_json()


def _build_payload(sample_dir: Path, *, checkpoint_ids: list[int]) -> None:
    """Materialize a complete sample payload.

    Host repo, a restic sandbox repo, an archive-strategy sandbox
    storage area, the restic config, the strategy pin, and the
    checkpoint files.
    """
    (sample_dir / "restic" / "host" / "data" / "ab").mkdir(parents=True)
    (sample_dir / "restic" / "host" / "config").write_text("host-config")
    (sample_dir / "restic" / "host" / "data" / "ab" / "cd").write_text("pack")
    (sample_dir / "restic" / "sandboxes" / "default").mkdir(parents=True)
    (sample_dir / "restic" / "sandboxes" / "default" / "config").write_text("sb-config")
    (sample_dir / "sandboxes" / "bulk" / "archive").mkdir(parents=True)
    (sample_dir / "sandboxes" / "bulk" / "archive" / "ckpt-00001.tar.gz").write_text(
        "tarball"
    )
    (sample_dir / "restic" / "restic-config.json").write_text(
        '{"restic_password":"pw"}'
    )
    (sample_dir / "restic" / "snapshot-strategies.json").write_text(
        '{"strategies":{"default":"restic-incremental","bulk":"archive"}}'
    )
    # the live host-context working dir: not payload (its committed
    # contents are in the host repo)
    (sample_dir / "context").mkdir()
    (sample_dir / "context" / "events.json").write_text("[]")
    for n in checkpoint_ids:
        (sample_dir / f"ckpt-{n:05d}.json").write_text(_checkpoint_json(n))


def _assert_payload_copied(dest: Path, *, checkpoint_ids: list[int]) -> None:
    assert (dest / "restic" / "host" / "config").read_text() == "host-config"
    assert (dest / "restic" / "host" / "data" / "ab" / "cd").read_text() == "pack"
    assert (
        dest / "restic" / "sandboxes" / "default" / "config"
    ).read_text() == "sb-config"
    assert (
        dest / "sandboxes" / "bulk" / "archive" / "ckpt-00001.tar.gz"
    ).read_text() == "tarball"
    assert (
        dest / "restic" / "restic-config.json"
    ).read_text() == '{"restic_password":"pw"}'
    assert (
        dest / "restic" / "snapshot-strategies.json"
    ).read_text() == '{"strategies":{"default":"restic-incremental","bulk":"archive"}}'
    for n in checkpoint_ids:
        assert (dest / f"ckpt-{n:05d}.json").read_text() == _checkpoint_json(n)
    assert not (dest / "context").exists()


async def test_copy_resume_payloads_replicates_every_sample_dir(
    tmp_path: Path,
) -> None:
    """Every sample dir the source has is copied — a complete archive.

    Completed samples' dirs are copied too: checkpoints are a permanent
    record (future work branches from arbitrary checkpoints), so each
    attempt with a log is a complete, self-contained archive.
    """
    source_eval = tmp_path / "old.checkpoints"
    _build_payload(source_eval / "s1__1", checkpoint_ids=[1, 2])
    _build_payload(source_eval / "s2__1", checkpoint_ids=[1])
    dest_eval = tmp_path / "new.checkpoints"

    await copy_resume_payloads(
        source_eval_dir=str(source_eval),
        destination_eval_dir=str(dest_eval),
    )

    _assert_payload_copied(dest_eval / "s1__1", checkpoint_ids=[1, 2])
    _assert_payload_copied(dest_eval / "s2__1", checkpoint_ids=[1])


async def test_copy_resume_payloads_missing_source_eval_dir(tmp_path: Path) -> None:
    """A prior attempt that never checkpointed leaves no eval dir at all."""
    dest_eval = tmp_path / "new.checkpoints"

    await copy_resume_payloads(
        source_eval_dir=str(tmp_path / "never-existed.checkpoints"),
        destination_eval_dir=str(dest_eval),
    )

    # nothing to copy
    assert not any(dest_eval.iterdir())


async def test_copy_resume_payloads_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any copy failure propagates.

    The caller runs before the destination log's first write, so the
    raise means the attempt dies without a log and the next retry falls
    back to the newest log that exists.
    """
    source_eval = tmp_path / "old.checkpoints"
    _build_payload(source_eval / "s1__1", checkpoint_ids=[1])
    dest_eval = tmp_path / "new.checkpoints"

    async def failing_copy(
        source_dir: str, destination_dir: str, rels: list[str]
    ) -> list[str]:
        raise OSError("simulated transient failure")

    monkeypatch.setattr(resume_copy, "_copy_payload_data", failing_copy)

    with pytest.raises(OSError):
        await copy_resume_payloads(
            source_eval_dir=str(source_eval),
            destination_eval_dir=str(dest_eval),
        )


async def test_copy_resume_payloads_copies_incomplete_sample_dirs_verbatim(
    tmp_path: Path,
) -> None:
    """A sample dir is copied as whatever it holds; nothing is required.

    The copy has no opinion about a sample dir's shape: a dir whose
    provisioning died before `restic init` finished, or a storage area
    that is an empty directory, copies as-is (an empty dir contributes
    no files). Neither holds a committed checkpoint, so resume detection
    runs such a sample fresh — and neither can poison the source
    attempt.
    """
    source_eval = tmp_path / "old.checkpoints"
    _build_payload(source_eval / "s1__1", checkpoint_ids=[1])
    (source_eval / "s1__1" / "restic" / "sandboxes" / "torn").mkdir()
    (source_eval / "s2__1" / "restic").mkdir(parents=True)  # no restic/host
    (source_eval / "s2__1" / "restic" / "restic-config.json").write_text("{}")
    dest_eval = tmp_path / "new.checkpoints"

    await copy_resume_payloads(
        source_eval_dir=str(source_eval),
        destination_eval_dir=str(dest_eval),
    )

    _assert_payload_copied(dest_eval / "s1__1", checkpoint_ids=[1])
    assert not (dest_eval / "s1__1" / "restic" / "sandboxes" / "torn").exists()
    assert (dest_eval / "s2__1" / "restic" / "restic-config.json").read_text() == "{}"
    assert not list((dest_eval / "s2__1").glob("ckpt-*.json"))


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


async def test_copy_branch_payload_stops_at_the_branch_checkpoint(
    tmp_path: Path,
) -> None:
    """The branch's latest committed checkpoint is the one it starts from.

    Everything else is copied as in a retry, so resume's own orphan
    handling discards the later snapshots. The destination dir is named
    for the branch run, whose epoch need not be the source's.
    """
    source = tmp_path / "parent.checkpoints" / "s1__2"
    _build_payload(source, checkpoint_ids=[1, 2, 3, 4])
    # a storage area's file is payload whatever its name
    (source / "restic" / "host" / "ckpt-00004.json").write_text("pack")
    before = _files(source)
    dest = tmp_path / "branch.checkpoints" / "s1__1"

    await copy_branch_payload(
        source_sample_dir=str(source),
        destination_sample_dir=str(dest),
        checkpoint_id=2,
    )

    _assert_payload_copied(dest, checkpoint_ids=[1, 2])
    assert sorted(path.name for path in dest.glob("ckpt-*.json")) == [
        "ckpt-00001.json",
        "ckpt-00002.json",
    ]
    assert (dest / "restic" / "host" / "ckpt-00004.json").read_text() == "pack"
    assert await scan_latest_committed_id(str(dest)) == 2
    assert _files(source) == before


async def test_copy_branch_payload_copies_nothing_from_an_unbranchable_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "parent.checkpoints" / "s1__1"
    _build_payload(source, checkpoint_ids=[1, 2])
    dest = tmp_path / "branch.checkpoints" / "s1__1"

    with pytest.raises(PrerequisiteError):
        await copy_branch_payload(
            source_sample_dir=str(source),
            destination_sample_dir=str(dest),
            checkpoint_id=3,
        )

    assert not dest.exists()


async def test_branchable_checkpoint_returns_the_committed_checkpoint(
    tmp_path: Path,
) -> None:
    _build_payload(tmp_path, checkpoint_ids=[1, 2, 3])

    assert (await branchable_checkpoint(str(tmp_path), 2)).checkpoint_id == 2


@pytest.mark.parametrize(
    "checkpoint_id",
    [
        pytest.param(9, id="unknown"),
        pytest.param(2, id="torn-file"),
        pytest.param(4, id="agent-complete"),
    ],
)
async def test_branchable_checkpoint_names_the_checkpoints_a_branch_can_start_from(
    tmp_path: Path, checkpoint_id: int
) -> None:
    _build_payload(tmp_path, checkpoint_ids=[1, 2, 3])
    (tmp_path / "ckpt-00002.json").write_text('{"checkpoint_id":')
    (tmp_path / "ckpt-00004.json").write_text(_checkpoint_json(4, "agent_complete"))

    with pytest.raises(PrerequisiteError) as excinfo:
        await branchable_checkpoint(str(tmp_path), checkpoint_id)

    assert [c.checkpoint_id for c in await branchable_checkpoints(str(tmp_path))] == [
        1,
        3,
    ]
    assert str(excinfo.value).endswith("Checkpoints a branch can start from: 1, 3.")


async def test_branchable_checkpoint_missing_sample_dir(tmp_path: Path) -> None:
    with pytest.raises(PrerequisiteError, match="can start from: none"):
        await branchable_checkpoint(str(tmp_path / "never-existed__1"), 1)


@pytest.mark.parametrize(
    ("latest_checkpoint", "message"),
    [
        pytest.param(2, "Try another approach.", id="at-the-branch-checkpoint"),
        pytest.param(3, None, id="after-it"),
    ],
)
async def test_resume_adds_a_branch_message_only_at_the_branch_checkpoint(
    tmp_path: Path, latest_checkpoint: int, message: str | None
) -> None:
    """A retry of a branch does not repeat its message.

    The message joins the conversation when the branch starts, so every
    checkpoint the branch then commits already holds it.
    """
    _build_payload(
        tmp_path / "s1__1", checkpoint_ids=list(range(1, latest_checkpoint + 1))
    )
    branch = EvalBranch(
        log="parent.eval",
        eval_id="parent-eval-id",
        sample_id="s1",
        epoch=1,
        checkpoint_id=2,
        message="Try another approach.",
    )

    resume = await resolve_resume_checkpoint(str(tmp_path), "s1", 1, branch)

    assert resume is not None
    assert resume.message == message

# Checkpointing: branching an eval from a checkpoint

> **Status.** Prototype. Implemented: `eval_branch()` / `inspect eval-branch`,
> the `EvalSpec.branch` log field, the branch copy, and delivery of a
> message at the branch point. Not implemented: any viewer support, and
> everything under [Not covered](#not-covered). The decisions under
> [Open questions](#open-questions) are provisional.

## Overview

Checkpointing resumes a sample from its latest committed checkpoint when an
eval is retried. Every committed checkpoint is kept, and each one has all
the data it references (`checkpoint-snapshot-strategy.md` §4.1), so any of
them can be resumed from. A branch does that deliberately: it runs one
sample again as a new eval that starts at a chosen checkpoint, optionally
with a different model or with a user message added at that point.

``` bash
inspect eval ctf.py --checkpoint=turn:1
inspect eval-branch logs/<log>.eval                       # list checkpoints
inspect eval-branch logs/<log>.eval --sample-id web-1 --checkpoint-id 6 \
    --model openai/gpt-6-luna --message "Try a different approach."
```

The uses are comparisons from a shared state: what another model does from
the same point, how repeatable a continuation is, how the agent responds to
a changed instruction. They are not measurements of a model's performance
on the task.

## What a branch is

A branch is a new task run of one sample, for one epoch, whose checkpoint
history is the history of the sample it was branched from up to the branch
checkpoint. It is not another attempt of the eval it was branched from: it
has its own `task_id`, reuses none of that eval's samples, and carries none
of its eval-level usage.

Its log is an ordinary log with one addition, `eval.branch`:

| Field | |
|---|---|
| `log` | Location of the log branched from, when the branch was created. |
| `eval_id` | Id of the eval branched from. Stable if the log moves. |
| `sample_id`, `epoch` | The sample branched. |
| `checkpoint_id` | The checkpoint the branch started from. |
| `message` | The user message added at the branch point, if any. |

Each branch names its immediate parent, so a tree of branches is recovered
by following `eval.branch.eval_id` from log to log. A retry of a branch
(`eval-retry`) keeps the branch's pointer.

## Mechanism

Branching adds no restore code. It arranges for the existing resume to find
the branch checkpoint as the sample's latest.

1. **Copy.** Before the branch's log is started, `copy_branch_payload`
   (`_resume_copy.py`) copies the branched sample's checkpoints dir to the
   branch's own, leaving out the `ckpt-NNNNN.json` files after the branch
   checkpoint. It is the retry startup copy for one sample, with one filter.
   Storage areas are copied verbatim; the copy reads nothing beneath them.
   The source is only read.
2. **Resume.** A checkpoint file is the commit point, so the branch
   checkpoint is now the dir's latest committed checkpoint. Resume detection
   (`resolve_resume_checkpoint`) finds it, and hydration restores it as it
   would for a retry: snapshots the copy carried from later checkpoints are
   orphans, and the orphan handling that already exists discards them. The
   next checkpoint is numbered one past the branch checkpoint.

Because the branch's dir holds its whole history, a branch is branched, or
retried, like any other eval.

The entry point reuses `eval-retry`'s reconstruction of the task from the
log (`_eval_previous_async` in `eval.py`). A branch differs from a retry in
four places: a fresh task id, the sample and epoch selection, an optional
model override, and a sample source (`EvalSampleSource.branch`) that reuses
no prior sample and asks for the branch copy.

A branch cannot start from the `agent_complete` checkpoint: a resume from it
only re-scores.

## The message

A message given to a branch is delivered as an operator message. Once the
agent has entered the checkpointer, which is when it restores its
conversation, the first agent channel it opens is posted the message, and
the loop's `before_turn` drain appends it. `react()` needed no change.

The message is delivered when the checkpoint being resumed is the branch
checkpoint, and only then. The branch's first checkpoint is taken after the
turn that received the message, so it holds the message. A branch that
fails before committing that checkpoint is retried from the branch
checkpoint and is given the message again; one that fails later is retried
from a checkpoint that already has it.

## A different model

Branching does nothing to the conversation for a different model: the new
model is given what the earlier one produced, and providers handle another
model's content as they do in any multi-model conversation
(`reasoning_history` controls what reasoning is replayed).

A branch keeps the logged eval-level generate config. The logged model's
own config, base URL and arguments describe that model instance and are not
applied to a model the caller names.

## What a checkpoint costs

A branch can only start where a checkpoint was taken, so branching invites
`TurnInterval(every=1)`. Measured on one machine, with a sandbox capturing
five small files:

| Sandbox strategy | Time per checkpoint |
|---|---|
| `ResticSnapshots` (the default) | about 3.1 s |
| `ArchiveSnapshots` | 0.94 s |

The data is negligible in both. With restic the time is a fixed cost: each
restic command takes 0.5 to 0.7 s to open its repository (key derivation; a
repository created with `--insecure-no-password` opens no faster), and a
checkpoint runs about five of them in sequence for a sandbox. They are the
backup in the sandbox and, on the host, the snapshot listings before and
after the copy-out and the two commands of `list_changed_files`. The host
backup runs in parallel with these, and is what remains for the archive
strategy.

In the restic runs the checkpoints took about twice as long as the model
calls. Options, none of which this prototype implements:

- Use the archive strategy when the captured paths are small.
- Take the changed-file list from the backup's own output, or make it
  optional. It is two of the five commands.
- Merge the two snapshot listings around the copy-out.
- Take the snapshot while the model generates its next reply. The sandbox
  does not change during a generate, so the snapshot only has to finish
  before the next tool call.

A large sandbox will weigh these differently, and will make the branch copy
and the restore matter, which they do not at this size.

## Validation

Run with real models through OpenRouter, Docker sandboxes and the default
(restic) snapshot strategy. `tests/checkpoint/test_branch_e2e.py` covers:

- A branch finds the files the branched sample had at the checkpoint and
  continues its conversation; the sample branched from is unchanged.
- A branch from Claude Sonnet 5 to DeepSeek V3.2.
- A message is added once, including across a retry of a failed branch.
- A branch of a branch.

Replaying one model's reasoning to another was checked by hand, and only in
one direction: a DeepSeek run that produced reasoning was branched to
Claude, which accepted it. The Claude runs produced no reasoning, with
reasoning requested, so the other direction was not exercised.

The archive strategy was also checked by hand: an eval that used it was
branched to another model, and the branch found the files of its
checkpoint.

Not exercised: remote checkpoint locations, direct provider APIs, and
bridged agents.

## Not covered {#not-covered}

- **Viewer.** Nothing renders `eval.branch`. A branch's transcript shows the
  inherited history under the existing `prior_run` span.
- **More than one sample**, or more than one epoch, per branch.
- **Changing limits.** A branch inherits the sample's limits and its usage
  up to the checkpoint, as a retry does.
- **Tasks with a dynamic sample source.**
- **A checkpoint before the first turn.** The first checkpoint follows the
  first turn.

## Open questions {#open-questions}

1. **How branches relate, and how the viewer shows it.** This prototype
   records a parent pointer on the eval. A per-sample pointer would allow
   branching several samples into one eval. For the viewer, three places
   could use the pointer:
   - *The transcript.* The viewer has a branch point control for
     trajectories that fork within a sample (`timeline_branch()`): it lists
     the trajectories that start at a point and switches between them. The
     same control at a checkpoint could list the evals branched from it.
   - *The log page.* The header the viewer loads already carries
     `eval.branch`; it could show the eval and checkpoint branched from,
     with a link.
   - *The log list.* It is built from `LogOverview`, which has no pointer,
     so the list cannot show or group branches without one.

   Eval dataframes (`inspect_ai.analysis`) have no branch columns either;
   with them, evals could be grouped by the checkpoint they share.
2. **Retention.** `retention="delete"` is documented as the default but not
   yet enforced. Once it is, a successful eval can be branched only if its
   checkpoints were retained.
3. **The checkpoint configuration is not in the log.** As for `eval-retry`,
   a configuration given to `eval()` must be given again. Recording it would
   remove the step for both.
4. **Later snapshots travel with the copy.** For the restic strategy the
   branch's repository holds the later checkpoints' data, unreferenced, and
   ingress places the repository in the branch's sandbox. Pruning after the
   orphans are forgotten would remove it, at a cost per branch. The archive
   strategy does not have this: resume deletes the later archives.
5. **Eval sets.** A branch's log in an eval set's log directory would be
   read as a run of the task, so `eval_branch()` refuses that directory.
   Eval sets could instead ignore logs with `eval.branch`.
6. **Message delivery.** The agent channel, as here, or the resume report
   (`cp.restored`), which is documented for agents but which `react()` does
   not yet read.
7. **Listing checkpoints.** `eval-branch` without `--checkpoint-id` lists
   them. A separate command may be clearer.
8. **`eval-retry` of several logs.** Sharing the reconstruction made each
   log resolve its own settings; before, a setting left unset carried over
   from the log retried before it.

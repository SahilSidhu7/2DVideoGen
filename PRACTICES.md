# PRACTICES.md - how this project was run

Twenty-four attempts, roughly half of them failures. These are the working
rules that survived, each one written because something went wrong without it.

---

## 1. Measure the thing you actually want, never the loss

`eval_loss` across the four shipped checkpoints went **0.9240 -> 0.9470 ->
1.0042 -> 0.9468**, while out-of-distribution prompt-match went **25.0% ->
62.5% -> 70.8% -> 79.2%**. For three consecutive checkpoints, loss got *worse*
as the model got *better*, because each new dataset was higher-entropy than the
last. Anything decided on loss in this project would have been decided wrong.

`model/scene_eval.py` scores parse / valid / render / prompt-match / staging on
prompts the model never saw, and reports in-distribution and out-of-distribution
**separately, because the gap between them is the result**.

## 2. Predict before you measure, then score the prediction

Every attempt from 23 onward states a prediction per technique *before* running
it, then scores it honestly in a table - including the ones that were wrong.
Attempt 23 falsified two of its own five predictions; Attempt 24 falsified one
and understated another. Those rows are the most transferable content in
`ATTEMPTS.md`.

## 3. Distrust the evaluator, four times over

Four separate times a metric or a decoding choice produced a **confident wrong
answer**:

1. Perceptual similarity scored known-bad blobs *above* a correct control.
2. A single stochastic detection compared against a different crop basis cost a
   clip its verdict - twice.
3. Pose estimators return *no person* on every stick figure this project
   produces, including the known-good ones.
4. Beam search reported the mode of a fine distribution and was recorded as a
   model defect (staging 0.188 -> 0.862 on identical weights, sampling instead).

Each was caught by disbelieving a number that looked too clean. A metric that
has never surprised you has probably never been checked.

## 4. Change one variable

Attempt 24 trained a 3.7x larger model on **byte-identical data** (val
`md5 a8f6c5c7`), the same effective batch, the same lr, the same epochs, the
same sequence caps - so the delta is attributable to capacity and nothing else.
Where a second variable was unavoidable (GPU instead of CPU, gradient
checkpointing), it is named in the write-up rather than buried.

## 5. Never overwrite a checkpoint

`model/checkpoints/<version>/` holds every shipped model, the data that
produced it, its summary, and `run_provenance.json` (base model, params, data
paths, effective batch, device). `model/verify_checkpoints.py` reloads all of
them and reproduces a published string byte-for-byte. New runs go to a **new**
directory; nothing writes into `checkpoints/`.

## 6. Respect the hardware, in code

The development machine's GPU fan is damaged, so **every** GPU loop runs through
`tools/gpuguard.py`: 50% duty cycle, 70 degC ceiling, 60 degC resume, half-clock
lock where an elevated shell allows it. HF `Trainer` owns its own loop, so the
guard enters it as `GuardCallback(TrainerCallback)` on `on_step_end`. CPU is the
explicit default in `model/train.py` (`use_cpu=not args.gpu`) precisely because
`Trainer` will otherwise claim a GPU silently - which it did, and OOM'd.

Every run reports its duty honestly, including what was *not* protected:
`busy 594s / idle 594s = 50% duty, 0 cooldowns, clock lock off`.

## 7. Assume the run will be interrupted

Long runs were killed three times mid-training. The first kill landed
mid-checkpoint-save and, with `save_total_limit=1`, would have destroyed the
only good checkpoint. Now: `save_total_limit=2`, a `--resume` flag, and long
jobs launched **detached from the harness** so nothing reaps them.

## 8. Log the failures with their real causes

`ATTEMPTS.md` is 24 attempts including retractions, falsified predictions and a
route closed by measurement rather than opinion. Roughly half failed. The
failures are the reason the file is the primary source and the README is the
summary.

## 9. Reproducibility hygiene

- `USE_TF=0` before anything importing `transformers` (4.57 otherwise tries a
  Keras 3 backend and dies).
- Every eval cell carries a `--tag`, because two cells once overwrote each
  other's output.
- Fixed `--seed`, and the exact command in the write-up next to its result.
- `md5` recorded for weights and data whenever a claim depends on them being
  the same bytes.

## 10. Where AI assistance was used

Claude (via Claude Code) was used throughout as a pair-programmer: writing and
refactoring the training/eval harnesses, diagnosing the CUDA OOM and the
silent GPU claim, building the comparison figures, and drafting documentation.
Every number in this repository comes from a script in this repository that was
actually executed - the model wrote code and prose, it did not supply results.

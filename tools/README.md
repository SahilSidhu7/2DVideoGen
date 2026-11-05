# tools/

## `gpuguard.py`: thermal and power guard

The GPU fan on the development machine (an RTX 4060 Laptop with 8 GB) is damaged.
Unthrottled GPU work overheats it, so every GPU loop in this project runs
through `tools.gpuguard.Guard`. The guard holds the card to about 50% of
wall-clock time and backs off when it gets hot. The project depended on it:
every GPU number in [`ATTEMPTS.md`](../ATTEMPTS.md) and
[`paper/PAPER.md`](../paper/PAPER.md) was measured under it. The rule itself is
written down in [`PRACTICES.md`](../PRACTICES.md) §6.

The guard is a **library you call from inside your loop**, not a wrapper around
a command. It cannot see a process's GPU work from outside. The loop has to tell
it when each unit of work ends.

### What it does

The guard combines two mechanisms:

1. **Clock lock.** On construction it runs `nvidia-smi -lgc 0,1500`, which pins the
   SM clock to about half of the card's 3105 MHz maximum. Power rises faster than
   clock speed, so half clock produces well under half the heat. This needs an
   elevated shell. If it fails, the guard prints
   `[gpuguard] clock lock unavailable (...); duty cycling only.` and carries on.
   An `atexit` hook restores default clocks (`nvidia-smi -rgc`).
2. **Duty cycle.** `Guard.step()` measures how long it has been since the last
   call and sleeps for `busy × (1 − duty) / duty`. At the default `duty = 0.5`,
   that is an equal span. After sleeping it reads the die temperature
   (`nvidia-smi --query-gpu=temperature.gpu`). At or above the ceiling, it
   blocks in 5-second sleeps until the die is back at or below the resume
   temperature.

### Parameters

| parameter | default | where to change it |
|---|---|---|
| `duty` | `0.5` | Environment variable `GPU_DUTY`, or `Guard(duty=...)`. |
| `temp_ceiling` | `70` °C | `Guard(temp_ceiling=...)` (module constant `TEMP_CEILING`). |
| `temp_resume` | `60` °C | `Guard(temp_resume=...)` (`TEMP_RESUME`). |
| `clock_mhz` | `1500` | `Guard(clock_mhz=...)` (`HALF_CLOCK_MHZ`). |

### Usage

In a plain for-loop:

```python
from tools.gpuguard import Guard

g = Guard()                 # tries the clock lock
for batch in loader:
    train_step(batch)
    g.step()                # blocks until the duty budget allows more work
print(g.report())           # e.g. "[gpuguard] busy 594s / idle 594s = 50% duty, 0 cooldowns, clock lock off"
g.release()                 # restores default clocks
```

HF `Trainer` owns its own loop, so [`model/train.py`](../model/train.py) enters
the guard as a callback (`GuardCallback`, which calls `guard.step()` on
`on_step_end`). The callback is attached only when `--gpu` is passed. CPU is the
explicit default, so `Trainer` cannot claim the GPU without the guard. This is
the command that trained the 220M v4 checkpoint (Attempt 24, from
[`model/MODELS.md`](../model/MODELS.md)):

```bash
USE_TF=0 python model/train.py --base t5-base --gpu --grad-checkpoint \
    --train model/checkpoints/v3-staged/data/scene3_train.jsonl \
    --val model/checkpoints/v3-staged/data/scene3_val.jsonl \
    --out model/scene_ckpt4 --epochs 6 --batch 4 --grad-accum 4 \
    --lr 3e-4 --max-in 160 --max-out 256 --val-cap 200
```

That run reported `busy 594s / idle 594s = 50% duty, 0 cooldowns, clock lock off`.
The shell was not elevated, so duty cycling and the temperature ceiling did all
the work.

Other callers:

- [`critic/evaluate.py`](../critic/evaluate.py) creates a guard whenever a YOLO
  model is loaded. `--no-gpu-guard` skips it, and is meant for quick local tests only.
- [`svg/animate.py`](../svg/animate.py) and [`svg/render_style.py`](../svg/render_style.py)
  call `guard.step()` once per diffusion step (`animate.py`) or once per rendered frame (`render_style.py`).

Running the module directly prints the current temperature and an empty report.
Use it as a quick check that `nvidia-smi` is reachable:

```bash
python -m tools.gpuguard
```

### On a machine without a GPU

The guard does not fail. If `nvidia-smi` is missing or errors:

- the clock lock is reported as unavailable;
- `temperature()` returns `None`, so the cooldown check never blocks;
- **the duty-cycle sleep still runs**, so a loop that calls `step()` takes about
  twice as long.

Only construct a guard around real GPU work. `model/train.py` and
`critic/evaluate.py` already do this: they attach the guard only on `--gpu` or
when a GPU model is loaded. The project's CPU paths (`scenescript.py`,
`model/scene_infer.py`, and CPU training) never touch it.

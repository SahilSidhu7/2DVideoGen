# tools/

One file: [`gpuguard.py`](gpuguard.py), the thermal and power guard that holds
this project's GPU work to about 50%.

## Why it exists

The fan on the development machine is damaged, so every GPU run has to be
throttled. The guard applies two mechanisms together:

1. **Clock lock.** `nvidia-smi -lgc 0,1500` pins the SM clock to about half of
   the RTX 4060 Laptop's maximum (`HALF_CLOCK_MHZ = 1500`). This needs an
   elevated shell. If it fails, the guard prints a notice and carries on with
   the duty cycle alone.
2. **Duty cycle.** `Guard.step()` times the work since the last call and sleeps
   for `busy * (1 - duty) / duty`. At the default duty of 0.5 that is an equal
   span, so the GPU is busy at most half the wall-clock time.

On top of both, if the die reaches `TEMP_CEILING` (70 °C) the guard blocks,
polling every 5 s, until the temperature is back at or under `TEMP_RESUME`
(60 °C). If `nvidia-smi` cannot report a temperature, the temperature check is
skipped.

## API

| name | what it does |
|---|---|
| `Guard(duty, temp_ceiling, temp_resume, clock_mhz)` | Locks the clock if it can and registers `release` with `atexit`. Defaults come from the module constants. |
| `Guard.step()` | Call after each unit of GPU work. Sleeps to hold the duty budget, then runs `cool()`. |
| `Guard.cool()` | Blocks while the die is over the ceiling. |
| `Guard.report()` | One-line summary: busy and idle seconds, duty percentage, cooldown count, whether the clock lock is on. |
| `Guard.release()` | Restores default clocks with `nvidia-smi -rgc`. Safe to call twice. |
| `temperature()` | Current GPU temperature in °C, or `None` if `nvidia-smi` is unavailable. |

## Configuration

| setting | default | how to change it |
|---|---|---|
| duty | `0.5` | env var `GPU_DUTY`, or the `duty=` argument |
| temperature ceiling | `70` | `temp_ceiling=` argument |
| temperature resume | `60` | `temp_resume=` argument |
| locked clock | `1500` MHz | `clock_mhz=` argument |

`GPU_DUTY` is read once when the module is imported.

## Usage

```python
from tools.gpuguard import Guard

g = Guard()                  # locks clocks if it can
for batch in loader:
    train_step(batch)
    g.step()                 # blocks until the budget allows more work
print(g.report())
g.release()                  # restores default clocks
```

Callers in this repository import it the same way:

- `critic/evaluate.py` creates a `Guard` unless run with `--no-gpu-guard`, and
  puts `guard.report()` in the report as `gpu_guard`.
- `svg/animate.py` and `svg/render_style.py` create a `Guard` for their
  diffusion runs.
- `critic/metrics.py` refers to the guard for model-backed metrics.

Run from the repository root so that `tools` is importable.

To check the guard on its own:

```bash
python tools/gpuguard.py     # prints the temperature and a duty report, then releases
```

## Related

`tools/Soup/` is git-ignored vendored third-party source; clone it separately
if you need it.

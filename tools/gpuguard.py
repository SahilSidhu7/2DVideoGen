"""Thermal/power guard: keep this project at ~50% of the GPU.

The fan on this machine is damaged, so every training or inference run must be
throttled. Two mechanisms, applied together:

1. Clock lock. `nvidia-smi -lgc` pins the SM clock to half of max. Power scales
   super-linearly with clock, so half clock is well under half the heat. Needs
   an elevated shell; if it fails we fall back on (2) alone.
2. Duty cycle. `Guard.step()` is called once per training step. It measures how
   long the step took and sleeps for an equal span, so the GPU is busy at most
   50% of wall-clock time. If the die passes `temp_ceiling` it sleeps longer
   until the temperature comes back under `temp_resume`.

Usage:
    from tools.gpuguard import Guard
    g = Guard()                      # locks clocks if it can
    for batch in loader:
        train_step(batch)
        g.step()                     # blocks until the budget allows more work
    g.release()                      # restores default clocks
"""

import atexit
import os
import subprocess
import time

HALF_CLOCK_MHZ = 1500      # ~half of the 4060 Laptop's 3105 MHz max
TEMP_CEILING = 70          # degC: stop issuing work above this
TEMP_RESUME = 60           # degC: resume once back under this
DUTY = float(os.environ.get("GPU_DUTY", "0.5"))


def _smi(*args):
    try:
        out = subprocess.run(["nvidia-smi", *args], capture_output=True,
                             text=True, timeout=15)
        return out.returncode, (out.stdout + out.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def temperature():
    rc, out = _smi("--query-gpu=temperature.gpu", "--format=csv,noheader")
    if rc != 0:
        return None
    try:
        return int(out.splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


class Guard:
    def __init__(self, duty=DUTY, temp_ceiling=TEMP_CEILING,
                 temp_resume=TEMP_RESUME, clock_mhz=HALF_CLOCK_MHZ):
        self.duty = duty
        self.temp_ceiling = temp_ceiling
        self.temp_resume = temp_resume
        self.locked = False
        self.slept = 0.0
        self.worked = 0.0
        self.cooldowns = 0
        rc, out = _smi("-lgc", f"0,{clock_mhz}")
        if rc == 0:
            self.locked = True
            atexit.register(self.release)
        else:
            print(f"[gpuguard] clock lock unavailable ({out.splitlines()[0] if out else 'no output'}); "
                  "duty cycling only. Run the shell as Administrator for the lock.")
        self._mark = time.monotonic()

    def step(self):
        """Call after each unit of GPU work. Blocks to hold the duty budget."""
        now = time.monotonic()
        busy = now - self._mark
        self.worked += busy
        idle = busy * (1.0 - self.duty) / self.duty
        if idle > 0:
            time.sleep(idle)
            self.slept += idle
        self.cool()
        self._mark = time.monotonic()

    def cool(self):
        """Block while the die is over the ceiling."""
        t = temperature()
        if t is None or t < self.temp_ceiling:
            return
        self.cooldowns += 1
        print(f"[gpuguard] {t}C >= {self.temp_ceiling}C, holding until {self.temp_resume}C")
        while True:
            time.sleep(5)
            self.slept += 5
            t = temperature()
            if t is None or t <= self.temp_resume:
                return

    def report(self):
        total = self.worked + self.slept
        pct = 100.0 * self.worked / total if total else 0.0
        return (f"[gpuguard] busy {self.worked:.0f}s / idle {self.slept:.0f}s "
                f"= {pct:.0f}% duty, {self.cooldowns} cooldowns, "
                f"clock lock {'on' if self.locked else 'off'}")

    def release(self):
        if self.locked:
            _smi("-rgc")
            self.locked = False


if __name__ == "__main__":
    g = Guard()
    print("temp:", temperature())
    print(g.report())
    g.release()

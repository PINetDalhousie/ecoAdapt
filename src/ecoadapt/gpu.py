"""GPU-frequency actuation and power telemetry through NVML."""
import threading
import time

import pynvml

pynvml.nvmlInit()


def _handle(idx):
    return pynvml.nvmlDeviceGetHandleByIndex(idx)


def sm_clock(idx):
    return int(pynvml.nvmlDeviceGetClockInfo(_handle(idx), pynvml.NVML_CLOCK_SM))


class FrequencyActuator:
    """Locks the SM clock. A new command is refused until the previous transition
    has settled and the dwell interval has passed. Needs root."""

    def __init__(self, gpu_index, dwell_ms=250.0, tol_mhz=60, settle_timeout_s=0.75):
        self.idx = gpu_index
        self.dwell = dwell_ms / 1000.0
        self.tol = tol_mhz
        self.timeout = settle_timeout_s
        self.current = None
        self.works = None
        self.history = []          # (t, mhz) of every accepted command
        self._last = 0.0
        self._lock = threading.Lock()

    def _settled(self):
        waited = time.monotonic() - self._last
        if self.current is None or not self.works or waited >= self.timeout:
            return True
        return abs(sm_clock(self.idx) - self.current) <= self.tol

    def can_set(self):
        return time.monotonic() - self._last >= self.dwell and self._settled()

    def set(self, mhz, force=False):
        with self._lock:
            if mhz == self.current:
                return True
            if not force and not self.can_set():
                return False
            try:
                pynvml.nvmlDeviceSetGpuLockedClocks(_handle(self.idx), mhz, mhz)
                ok = True
            except pynvml.NVMLError as e:
                if self.works is None:
                    print(f"[ecoadapt] cannot lock GPU clocks ({e}); run with sudo -E")
                ok = False
            self.works = ok if self.works is None else self.works
            self.current = mhz
            self._last = time.monotonic()
            self.history.append((self._last, mhz))
            return ok

    def reset(self):
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(_handle(self.idx))
        except pynvml.NVMLError:
            pass
        self.current = None


class PowerMonitor:
    """Samples GPU power every `interval_ms` on a background thread."""

    def __init__(self, gpu_index, interval_ms=50):
        self.idx = gpu_index
        self.dt = interval_ms / 1000.0
        self.samples = []          # (t, watts, sm_mhz)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        h = _handle(self.idx)
        while not self._stop.is_set():
            try:
                w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                c = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                self.samples.append((time.monotonic(), w, c))
            except pynvml.NVMLError:
                pass
            time.sleep(self.dt)

    def _window(self, t0, t1):
        return [s for s in self.samples
                if (t0 is None or s[0] >= t0) and (t1 is None or s[0] <= t1)]

    def energy_j(self, t0=None, t1=None):
        s = self._window(t0, t1)
        return sum(0.5 * (a[1] + b[1]) * (b[0] - a[0]) for a, b in zip(s, s[1:]))

    def mean_power(self, t0=None, t1=None):
        s = self._window(t0, t1)
        return sum(x[1] for x in s) / len(s) if s else None

    def mean_clock(self, t0=None, t1=None):
        s = self._window(t0, t1)
        return sum(x[2] for x in s) / len(s) if s else None

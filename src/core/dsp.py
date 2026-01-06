import numpy as np


class FractionalDelayLine:
    """Real-time friendly fractional delay line for block processing.

    - Keeps a ring buffer of past samples.
    - Supports fractional delays via linear interpolation.
    - `process(x, delay_samples)` returns x delayed by `delay_samples`.

    Notes:
    - If delay is 0, the output equals the input (no latency).
    - The buffer auto-resizes to accommodate larger delays or block sizes.
    """

    def __init__(self, initial_capacity: int = 2048):
        cap = int(initial_capacity)
        cap = max(cap, 8)
        self._buf = np.zeros(cap, dtype=np.float32)
        self._w = 0

    def reset(self):
        self._buf.fill(0.0)
        self._w = 0

    @property
    def capacity(self) -> int:
        return int(self._buf.size)

    def _ensure_capacity(self, delay_samples: float, block_size: int):
        # Need enough slack so that writing a block never overwrites history that
        # could be read for any sample in the block.
        d = float(delay_samples)
        d = 0.0 if not np.isfinite(d) else max(d, 0.0)
        need = int(np.ceil(d)) + int(block_size) + 4
        if need <= self._buf.size:
            return

        # Grow to next power-of-two-ish size for fewer reallocations.
        new_cap = self._buf.size
        while new_cap < need:
            new_cap *= 2

        new_buf = np.zeros(new_cap, dtype=np.float32)

        # Preserve recent history by copying current buffer content in order.
        # Old buffer logical order: [w..end) then [0..w)
        old = self._buf
        w = int(self._w)
        tail = old[w:]
        head = old[:w]
        ordered = np.concatenate([tail, head], dtype=np.float32)
        # Place ordered history at start of new buffer and set write pointer to 0.
        new_buf[: ordered.size] = ordered
        self._buf = new_buf
        self._w = 0

    def process(self, x: np.ndarray, delay_samples: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n = int(x.size)
        if n == 0:
            return x

        d = float(delay_samples)
        if not np.isfinite(d):
            d = 0.0
        if d < 0.0:
            d = 0.0

        self._ensure_capacity(d, n)
        buf = self._buf
        N = int(buf.size)
        w0 = int(self._w)

        # Write input block into ring buffer.
        end = w0 + n
        if end <= N:
            buf[w0:end] = x
        else:
            k = N - w0
            buf[w0:] = x[:k]
            buf[: end - N] = x[k:]

        # Read delayed samples (linear interpolation).
        base = (w0 + np.arange(n, dtype=np.float32))
        pos = base - np.float32(d)
        pos_floor = np.floor(pos)
        frac = pos - pos_floor

        idx0 = (pos_floor.astype(np.int64) % N)
        idx1 = (idx0 + 1) % N

        y0 = buf[idx0]
        y1 = buf[idx1]
        y = (1.0 - frac) * y0 + frac * y1

        # Advance write pointer.
        self._w = int((w0 + n) % N)
        return y.astype(np.float32, copy=False)

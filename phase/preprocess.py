"""The one thing done to a trace before the network sees it.

Per-window z-score, computed in float64 and returned as float32. It is deliberately the only
preprocessing there is: at inference the device has no corpus statistics, so anything a training
run normalised by would not exist where the model has to run.

One definition, used by the dataset, by `evaluate.py`, by `train.py`'s test pass and by the edge
bundle - so the host and the device cannot drift apart on a mean.
"""

from __future__ import annotations

import numpy as np

WINDOW = "window"
"""Per-window z-score. Any other value is a pass-through."""

FLAT = 1e-8
"""Below this standard deviation a window is flat and is left alone - an apnoea or a lost bin is
a real thing, and dividing it by its own noise amplifies that noise into what looks like
breathing."""


def normalise_window(values, mode: str = WINDOW) -> np.ndarray:
    """Centre and scale one window on its own statistics. float64 inside, float32 out."""
    values = np.asarray(values, dtype=np.float64)
    if mode != WINDOW:
        return values.astype(np.float32)
    centred = values - values.mean()
    scale = centred.std()
    scaled = centred / scale if scale > FLAT else centred
    return scaled.astype(np.float32)

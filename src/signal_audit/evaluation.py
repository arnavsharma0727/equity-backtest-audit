from __future__ import annotations

import numpy as np


def purged_walk_forward(signal_time: list[str], label_end: list[str], splits: int, embargo_days: int):
    if len(signal_time) != len(label_end) or len(signal_time) < splits * 2:
        raise ValueError("need matching dates and at least two rows per requested split")
    if splits < 2 or embargo_days < 0:
        raise ValueError("splits must be >=2 and embargo_days nonnegative")
    order = np.argsort(np.asarray(signal_time))
    times = np.asarray(signal_time, dtype="datetime64[D]")[order]
    ends = np.asarray(label_end, dtype="datetime64[D]")[order]
    blocks = np.array_split(np.arange(len(times)), splits + 1)[1:]
    folds = []
    for test_indices in blocks:
        if not len(test_indices):
            continue
        test_start = times[test_indices[0]]
        cutoff = test_start - np.timedelta64(embargo_days, "D")
        train_indices = np.flatnonzero((times < test_start) & (ends < cutoff))
        if len(train_indices):
            folds.append({"train": order[train_indices], "test": order[test_indices], "test_start": str(times[test_indices[0]]), "test_end": str(times[test_indices[-1]])})
    if not folds:
        raise ValueError("no non-empty training fold remains after purging; reduce splits/embargo or add history")
    return folds


def circular_block_sharpe_interval(returns: list[float], block_length: int, draws: int, seed: int = 20261008) -> dict[str, float]:
    values = np.asarray(returns, dtype=float)
    if len(values) < 20 or not np.isfinite(values).all():
        raise ValueError("at least 20 finite out-of-sample returns are required")
    if not 1 <= block_length <= len(values) or draws < 100:
        raise ValueError("block_length must be within sample size and draws must be >=100")
    rng = np.random.default_rng(seed)
    n = len(values)
    sharpe = lambda sample: float(np.mean(sample) / np.std(sample, ddof=1) * np.sqrt(252)) if np.std(sample, ddof=1) else 0.0
    stats = np.empty(draws)
    blocks_needed = int(np.ceil(n / block_length))
    starts = np.arange(n)
    for i in range(draws):
        block_starts = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate([(start + np.arange(block_length)) % n for start in block_starts])[:n]
        stats[i] = sharpe(values[indices])
    observed = sharpe(values)
    low, high = np.quantile(stats, [0.025, 0.975])
    return {"annualized_sharpe": observed, "ci_2_5pct": float(low), "ci_97_5pct": float(high), "block_length": float(block_length), "bootstrap_draws": float(draws)}

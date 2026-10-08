from signal_audit.evaluation import circular_block_sharpe_interval, purged_walk_forward


def test_purge_excludes_overlapping_labels():
    dates = [f"2024-01-{day:02d}" for day in range(1, 13)]
    ends = [f"2024-01-{min(day + 2, 31):02d}" for day in range(1, 13)]
    folds = purged_walk_forward(dates, ends, splits=2, embargo_days=1)
    for fold in folds:
        test_start = fold["test_start"]
        assert all(ends[i] < test_start for i in fold["train"])


def test_block_bootstrap_returns_interval():
    values = [((i % 7) - 3) / 1000 for i in range(100)]
    result = circular_block_sharpe_interval(values, block_length=5, draws=100)
    assert result["ci_2_5pct"] <= result["ci_97_5pct"]

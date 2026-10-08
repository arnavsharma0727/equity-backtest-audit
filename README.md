# Equity signal validation audit

A small audit CLI for researchers who already have point-in-time signals and realized forward returns. It builds strictly chronological folds, purges any training label whose outcome window overlaps the test window, applies an embargo, and estimates an out-of-sample Sharpe confidence interval with a circular block bootstrap.

## Skill demonstrated

- Leakage-resistant purged walk-forward validation for overlapping financial labels.
- Block-bootstrap uncertainty estimation for serially dependent returns.

## Input contract

CSV columns: `signal_time`, `label_end`, `net_return`. Each row must represent one daily observation; `net_return` must be the realized daily return for the row's signal, after the submitter's documented transaction costs. Rows must not be duplicated by signal time. Sharpe annualization uses 252 daily periods.

```bash
pip install -e '.[test]'
signal-audit --csv predictions_and_returns.csv --splits 5 --embargo-days 2 --bootstrap 2000 --out output
```

The report contains fold boundaries, training/test row counts, the fraction of test rows, and an aggregate annualized Sharpe interval. The tool never creates market observations, model predictions, or fees; it audits the supplied real results. Bootstrap draws resample observed returns in blocks and are labeled as inferential resamples, not market history.

## Validation and limitations

The input's point-in-time feature construction cannot be proved from these three columns. The researcher must retain feature timestamps and document costs/benchmark definitions before treating a result as credible. The bootstrap interval assumes the chosen block length is reasonable; sensitivity to that choice should be reported.

```bash
pytest
```

# Data

Put one folder per item type here. Only `good/` is required.

```
data/<item>/good/*.jpg     # normal items: the anomaly model learns from these
data/<item>/defect/*.jpg   # optional: a few defective examples to calibrate the threshold
```

Images are git-ignored. Share them through a drive link.
`uv run python scripts/make_synthetic.py` creates a toy dataset for testing.

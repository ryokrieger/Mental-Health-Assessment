"""Checks that the numpy model in api/bindu.py reproduces the original scikit-learn pipeline.

Run from the project root, in the environment that trained the model (needs scikit-learn, joblib, pandas):
    python tests/test_model.py path/to/rf_grid.pkl path/to/train_raw.csv path/to/test_raw.csv
"""
import os, sys, importlib.util
import joblib
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("bindu", os.path.join(root, "api", "bindu.py"))
bindu = importlib.util.module_from_spec(spec); spec.loader.exec_module(bindu)

pkl, train_csv, test_csv = sys.argv[1:4]
pipe = joblib.load(pkl)
real = pd.concat([pd.read_csv(train_csv), pd.read_csv(test_csv)])[bindu.FEATURE_ORDER].values.astype(float)
rng = np.random.default_rng(0)
hi = np.array([bindu.MAX_ANSWER[f] for f in bindu.FEATURE_ORDER])
synthetic = rng.integers(0, hi + 1, size=(20000, len(hi))).astype(float)
extremes = np.array([np.zeros(len(hi)), hi.astype(float)])

worst = 0.0
for name, X in (("real students", real), ("random answer patterns", synthetic), ("all-minimum / all-maximum", extremes)):
    ref = pipe.predict_proba(X)
    got = np.array([bindu.predict_proba(row) for row in X])
    diff = np.abs(ref - got).max()
    same = (ref.argmax(1) == got.argmax(1)).mean()
    worst = max(worst, diff)
    print(f"{name:<28} n={len(X):>6}  max |probability difference| = {diff:.2e}   same predicted class: {same * 100:.2f}%")
    assert diff < 1e-5 and same == 1.0
print("\nPASS: the exported model reproduces the scikit-learn pipeline")
"""Export the tuned RF + ETC (grid) pipeline from Notebook 04 to a small numpy file.

Usage (from the project root, in the same environment that trained the model):
    python tools/export_model.py models/Optimised/etc/rf_grid.pkl models/bindu_rf_etc.npz

The deployed model takes the RAW questionnaire answers (integers), applies the StandardScaler in numpy
and walks the 300 trees; it needs nothing but numpy. SMOTE is a training-time step and is not needed
for prediction. Thresholds are kept in scaled space, exactly as trained, so the output is identical
to the scikit-learn pipeline (see tests/test_model.py).
"""
import sys, json
import joblib
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
pipe = joblib.load(src)
scaler, forest = pipe.named_steps['scaler'], pipe.named_steps['clf']
assert list(forest.classes_) == [0, 1, 2]

feat, thr, left, right, val, roots = [], [], [], [], [], []
offset = 0
for est in forest.estimators_:
    t = est.tree_
    n = t.node_count
    is_leaf = t.children_left == -1
    f = np.where(is_leaf, 0, t.feature)
    thr_scaled = np.where(is_leaf, 0.0, t.threshold)   # thresholds stay in scaled space, exactly as trained
    idx = np.arange(n) + offset
    left.append(np.where(is_leaf, idx, t.children_left + offset))      # leaves point to themselves
    right.append(np.where(is_leaf, idx, t.children_right + offset))
    feat.append(f); thr.append(thr_scaled)
    v = t.value[:, 0, :]
    val.append(v / v.sum(axis=1, keepdims=True))
    roots.append(offset)
    offset += n

np.savez_compressed(dst,
    feature=np.concatenate(feat).astype(np.int16), threshold=np.concatenate(thr).astype(np.float64),
    left=np.concatenate(left).astype(np.int32), right=np.concatenate(right).astype(np.int32),
    value=np.concatenate(val).astype(np.float32), roots=np.array(roots, dtype=np.int32),
    mean=scaler.mean_.astype(np.float64), scale=scaler.scale_.astype(np.float64))
print(f'wrote {dst}: {len(roots)} trees, {offset} nodes')
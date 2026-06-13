import time, csv, json
import numpy as np
import feat_lib as fl

rows = []
with open("results/manifest.csv") as f:
    for r in csv.DictReader(f):
        rows.append(r)

test = rows[:6]
t0 = time.time()
for r in test:
    ts = time.time()
    out, diag = fl.extract_one(r["wav"], r["phn"])
    dt = time.time() - ts
    nmeas = sum(1 for k, v in out.items() if isinstance(v, float) and np.isfinite(v))
    print(f"{r['utt_id']}  {dt:.2f}s  finite={nmeas}/40  diag={diag}")
    if r is test[0]:
        for k in fl.FEATURES_40:
            print(f"    {k:18s} {out[k]}")
print(f"avg {(time.time()-t0)/len(test):.2f}s/utt over {len(test)} files")

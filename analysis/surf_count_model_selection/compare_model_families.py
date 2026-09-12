"""
compare_model_families.py
-------------------------
Answers, with numbers rather than opinion: is the surfer-count model's error
a hyperparameter problem, a model-family problem, a target-scale problem, or
an information problem?

No tuning had ever been done on this model — no grid search, no
cross-validation, no sweep anywhere in the repo (`git log -S"GridSearch"`
finds nothing). The production hyperparameters were hand-picked once. This
script checks whether that left anything on the table.

Selection is done by 5-fold CV on the TRAIN split only; the held-out test
rows are scored once at the end and never used to choose anything.

Runtime is a few minutes (the 72-combination grid dominates).

Usage:
    python analysis/surf_count_model_selection/compare_model_families.py
"""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'code'))
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.model_selection import train_test_split, KFold
from fit_surfer_count_model import load_and_prepare, standardize

X,y,df,num = load_and_prepare()
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
Xtr,Xte = standardize(Xtr,Xte,num)
ytr_v, yte_v = ytr.to_numpy(float), yte.to_numpy(float)
kf = KFold(n_splits=5, shuffle=True, random_state=0)

def cv_mae(make, transform=None, inverse=None):
    maes=[]
    for tr,va in kf.split(Xtr):
        Xa,Xb = Xtr.iloc[tr], Xtr.iloc[va]
        ya,yb = ytr_v[tr], ytr_v[va]
        t = transform(ya) if transform else ya
        m = make().fit(Xa,t)
        p = m.predict(Xb)
        if inverse: p = inverse(p)
        maes.append(np.mean(np.abs(np.clip(p,0,None)-yb)))
    return float(np.mean(maes))

results=[]
t0=time.time()

# --- baseline (current production hyperparameters) ---
BASE = dict(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=42)
results.append(("GBT poisson (current)", cv_mae(lambda: HistGradientBoostingRegressor(loss="poisson",**BASE)), BASE))

# --- hyperparameter grid on the same loss ---
best=None
grid = list(itertools.product([3,4,6,None],[0.03,0.05,0.1],[300,600],[0.0,1.0,10.0]))
for depth,lr,it,l2 in grid:
    kw = dict(max_iter=it, learning_rate=lr, max_depth=depth, l2_regularization=l2, random_state=42)
    s = cv_mae(lambda kw=kw: HistGradientBoostingRegressor(loss="poisson",**kw))
    if best is None or s<best[0]: best=(s,kw)
results.append((f"GBT poisson (best of {len(grid)} grid)", best[0], best[1]))
print(f"grid done in {time.time()-t0:.0f}s", flush=True)

# --- alternative losses / target transforms ---
results.append(("GBT squared_error", cv_mae(lambda: HistGradientBoostingRegressor(loss="squared_error",**BASE)), BASE))
results.append(("GBT absolute_error", cv_mae(lambda: HistGradientBoostingRegressor(loss="absolute_error",**BASE)), BASE))
results.append(("GBT on log1p(y)", cv_mae(lambda: HistGradientBoostingRegressor(loss="squared_error",**BASE),
                                          np.log1p, np.expm1), BASE))
results.append(("GBT on sqrt(y)", cv_mae(lambda: HistGradientBoostingRegressor(loss="squared_error",**BASE),
                                         np.sqrt, lambda p: np.clip(p,0,None)**2), BASE))

# --- other model classes ---
results.append(("RandomForest(400)", cv_mae(lambda: RandomForestRegressor(n_estimators=400,random_state=42,n_jobs=-1)), {}))
results.append(("ExtraTrees(400)", cv_mae(lambda: ExtraTreesRegressor(n_estimators=400,random_state=42,n_jobs=-1)), {}))
results.append(("PoissonRegressor (linear)", cv_mae(lambda: PoissonRegressor(alpha=1.0,max_iter=2000)), {}))
results.append(("Ridge (linear)", cv_mae(lambda: Ridge(alpha=1.0)), {}))
results.append(("Predict train mean", cv_mae(lambda: __import__("sklearn.dummy",fromlist=["DummyRegressor"]).DummyRegressor(strategy="mean")), {}))

print(f"\n{'model':<34} {'CV MAE (train, 5-fold)':>24}")
for name,s,_ in sorted(results,key=lambda r:r[1]):
    print(f"  {name:<32} {s:>22.3f}")

# --- held-out check: current vs best grid ---
print("\n--- held-out test (never used for selection) ---")
def report(name, model, transform=None, inverse=None):
    t = transform(ytr_v) if transform else ytr_v
    m = model.fit(Xtr,t); p = m.predict(Xte)
    if inverse: p = inverse(p)
    p = np.clip(p,0,None)
    e = p-yte_v
    line = f"{name:<30} MAE {np.mean(np.abs(e)):.3f}  bias {np.mean(e):+.3f}"
    for lo,hi,lab in [(0,5,'0-4'),(30,10**6,'30+')]:
        k=(yte_v>=lo)&(yte_v<hi); line += f"  | {lab} bias {np.mean(e[k]):+6.2f}"
    print(line)
report("GBT poisson (current)", HistGradientBoostingRegressor(loss="poisson",**BASE))
report("GBT poisson (best grid)", HistGradientBoostingRegressor(loss="poisson",**best[1]))
report("GBT on log1p(y)", HistGradientBoostingRegressor(loss="squared_error",**BASE), np.log1p, np.expm1)
report("GBT absolute_error", HistGradientBoostingRegressor(loss="absolute_error",**BASE))
report("RandomForest(400)", RandomForestRegressor(n_estimators=400,random_state=42,n_jobs=-1))
print(f"\nbest grid params: {best[1]}")
print(f"total {time.time()-t0:.0f}s")

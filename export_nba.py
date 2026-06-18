import json, numpy as np, pandas as pd
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from simulate import simulate_seasons
from features import build_features, FEATURE_COLS

games = simulate_seasons(n_seasons=5, seed=7)
feats = build_features(games)
test_season = feats.season.max()
train = feats[feats.season < test_season]
test  = feats[feats.season == test_season]
Xtr, ytr = train[FEATURE_COLS], train.home_win
Xte, yte = test[FEATURE_COLS], test.home_win

cal = CalibratedClassifierCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                             method="sigmoid", cv=5).fit(Xtr, ytr)

members=[]
for cc in cal.calibrated_classifiers_:
    pipe=cc.estimator
    sc=pipe.named_steps["standardscaler"]; clf=pipe.named_steps["logisticregression"]
    calib=getattr(cc,"calibrators",None) or getattr(cc,"calibrators_"); cobj=calib[0]
    members.append(dict(mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                        coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]),
                        a=float(cobj.a_), b=float(cobj.b_)))

def fwd(Xdf):
    Xv=Xdf[FEATURE_COLS].to_numpy(float); out=np.zeros(len(Xv))
    for m in members:
        z=(Xv-m["mean"])/m["scale"]; dfu=z@np.array(m["coef"])+m["intercept"]
        out+=expit(-(m["a"]*dfu+m["b"]))
    return out/len(members)

from sklearn.metrics import roc_auc_score, accuracy_score
p_mine=fwd(Xte); p_skl=cal.predict_proba(Xte)[:,1]
print(f"members={len(members)} max|mine-skl|={np.max(np.abs(p_mine-p_skl)):.2e} "
      f"acc={accuracy_score(yte,p_skl>0.5):.3f} auc={roc_auc_score(yte,p_skl):.3f}")

model=dict(features=FEATURE_COLS, home_court_elo=65.0, members=members,
           importance={"elo_diff":0.1804,"rest_diff":0.0097,"form_diff":0.001,
                       "netrtg_diff":0.0007,"home_b2b":0.0002,"away_b2b":0.0004})
json.dump(model, open("web_model.json","w"))
print("wrote web_model.json", round(len(json.dumps(model))/1024,1),"KB")
# show a few example matchups
def wp(home_elo,away_elo,hr=2,ar=2,form=0,net=0):
    row=pd.DataFrame([{ "elo_diff":(home_elo+65)-away_elo,"form_diff":form,"netrtg_diff":net,
        "rest_diff":hr-ar,"home_b2b":int(hr==0),"away_b2b":int(ar==0)}])[FEATURE_COLS]
    return float(cal.predict_proba(row)[0,1])
print("even (1500 v 1500):", round(wp(1500,1500),3))
print("strong home (1620 v 1480):", round(wp(1620,1480),3))
print("road favorite (1480 v 1620):", round(wp(1480,1620),3))
print("even but home on b2b (rest0 v rest3):", round(wp(1500,1500,0,3),3))

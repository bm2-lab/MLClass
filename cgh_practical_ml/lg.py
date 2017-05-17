from sklearn.externals import joblib
from train.build_model import *
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

xtr, ytr, xte, yte = joblib.load('data/clsdata.pkl')


cls = build_logit(xtr, ytr)

y_hat_prob = model_predict(xte, cls)

fpr, tpr, thr = roc_curve(yte, y_hat_prob)
auc = roc_auc_score(yte, y_hat_prob)
print(auc)

plt.plot(fpr, tpr)
plt.show()
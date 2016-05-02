import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LinearRegression

x, y = load_svmlight_file('reg_big.data')


fold = 5
kf = KFold(len(y), fold)
lxtr, lytr, lxte, lyte = zip(*[(x[idx_tr], y[idx_tr], x[idx_te], y[idx_te]) for idx_tr, idx_te in kf])

m = LinearRegression()
m.fit(x, y)
r2_train = r2_score(y, m.predict(x))

lmodel = map(lambda (xtr, ytr): LinearRegression().fit(xtr, ytr), zip(lxtr, lytr))
lr2_test = map(lambda (model, xte, yte): r2_score(yte, model.predict(xte)), zip(lmodel, lxte, lyte))
r2_test = np.average(lr2_test)

print('Traing R2 Score: {0}'.format(np.round(r2_train, 5)))
print('Testing R2 Score: {0}'.format(np.round(r2_test, 5)))

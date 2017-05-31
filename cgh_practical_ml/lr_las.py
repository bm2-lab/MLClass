import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import Lasso
from train.build_model import *

np.random.seed(1337)


x, y = load_svmlight_file('data/reg_big.data')
x = np.asarray(x.todense())

tri, tei = split_testing_data_r(y)
xtr = x[tri]
ytr = y[tri]
xte = x[tei]
yte = y[tei]

alp = 0.1

m = Lasso(alpha=alp)
m.fit(xtr, ytr)
r2_train = r2_score(ytr, m.predict(xtr))
r2_test = r2_score(yte, m.predict(xte))

print('Traing R2 Score: {0}'.format(np.round(r2_train, 5)))
print('Testing R2 Score: {0}'.format(np.round(r2_test, 5)))


import numpy as np
from sklearn.datasets import load_svmlight_file

x, y = load_svmlight_file('data/reg_big.data')
x = np.asarray(x.todense())








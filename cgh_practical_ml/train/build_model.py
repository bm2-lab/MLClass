import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit, KFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score


def split_testing_data_r(y):
    ss = ShuffleSplit(n_splits=1, test_size=0.2)
    tri = None
    tei = None
    for itr, ite in ss.split(y):
        tri = itr
        tei = ite
    return tri, tei


def split_kfold_r(y):
    skf = KFold(5)
    ilst = []
    for tri, tei in skf.split(y):
        ilst.append((tri, tei))
    return ilst


def split_testing_data_c(y):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    tri = None
    tei = None
    for itr, ite in sss.split(np.zeros(len(y)), y):
        tri = itr
        tei = ite
    return tri, tei


def split_kfold_c(y):
    skf = StratifiedKFold(5)
    ilst = []
    for tri, tei in skf.split(np.zeros(len(y)), y):
        ilst.append((tri, tei))
    return ilst


def get_cutting_point(ytr_true, ytr_pred):
    fpr, tpr, thr = roc_curve(ytr_true, ytr_pred)
    thres = thr[np.argmax(tpr - fpr)]
    return thres


def feature_selection_logit(xtr, ytr):
    model = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5)
    model.fit(xtr, ytr)
    columns = np.arange(xtr.shape[1])[~np.isclose(model.coef_.ravel(), 0)]
    return columns


def build_logit_with_fs(xtr_raw, ytr, columns):
    cls = LogisticRegression()
    xtr = xtr_raw[:, columns]
    cls.fit(xtr, ytr)
    return cls


def build_logit(xtr, ytr):
    cls = LogisticRegression()
    cls.fit(xtr, ytr)
    return cls


def model_predict_with_fs(x_raw, cls, columns):
    x = x_raw[:, columns]
    y_hat_prob = cls.predict_proba(x)[:, 1]
    return y_hat_prob.ravel()


def model_predict(x, cls):
    y_hat_prob = cls.predict_proba(x)[:, 1]
    return y_hat_prob.ravel()


def eval_model(y, y_hat_prob, of_log=None):
    thres = 0.5
    y_hat = np.where(y_hat_prob >= thres, 1, 0)
    print('Accuracy: {0}'.format(accuracy_score(y, y_hat)))
    print('AUC: {0}'.format(roc_auc_score(y, y_hat_prob)))
    if of_log is not None:
        of_log.write('Accuracy: {0}'.format(accuracy_score(y, y_hat)) + '\n')
        of_log.write('AUC: {0}'.format(roc_auc_score(y, y_hat_prob)) + '\n')

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


class PLSDA:

    def __init__(self):
        self.plsda = None
        pass

    def train(self, X_train, y_train, ncomp=4):
        plsda = PLSRegression(n_components=ncomp)
        # Fit the training set
        plsda.fit(X_train, y_train)

        self.plsda = plsda

    def predict(self, X_predict):
        try:
            return (self.plsda.predict(X_predict)[:, 0] > 0.5).astype(int)
        except AttributeError:
            print("PLSDA isn't initiated")
            return -1


def get_X_and_y():
    data = pd.read_csv(r'data\StudentsTrain_4sem.csv').to_numpy()

    y = data[:, :1]
    y = (y == 'Выпускник').astype(int)

    X = data[:, 4:-1]
    X = (X - X.mean()) / X.std()
    return X, y


def one_run(n_test=1):
    X, y = get_X_and_y()

    X_train = X[:-n_test]
    y_train = y[:-n_test]
    X_test = X[-n_test:]
    y_test = y[-n_test:]

    pls = PLSDA()
    pls.train(X_train, y_train)
    res = pls.predict(X_test)
    print(res)
    print(accuracy_score(y_test, res))


def cross_validation_run(n=10):
    X, y = get_X_and_y()

    accuracy = []
    cval = KFold(n_splits=n, shuffle=False)
    for train, test in cval.split(X):
        pls = PLSDA()
        pls.train(X[train, :], y[train])
        y_pred = pls.predict(X[test, :])
        accuracy.append(accuracy_score(y[test], y_pred))
    print(f"Average accuracy on {n} splits: ", np.array(accuracy).mean())
    print(accuracy)


one_run()
cross_validation_run()

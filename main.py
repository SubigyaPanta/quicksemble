import pandas
from sklearn.datasets import load_iris
from xgboost import XGBClassifier

if __name__ == '__main__':
    data = load_iris()
    X = data['data']
    Y = data['target']
    y_name = data['target_names']

    print(X, Y, y_name)

    pd = pandas.DataFrame(data=X)
    print(pd[0:2])
    print(X[0:2])
    xg = XGBClassifier()
    # xg.predict(X)

    print(type(xg.__class__.__name__))
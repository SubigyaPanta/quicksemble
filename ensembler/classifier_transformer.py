import numpy
from sklearn.base import TransformerMixin


class ClassifierTransformer(TransformerMixin):
    """
    This class is used only for saved models, or models that are already fitted.
    """

    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        pred = self.clf.predict(X)
        return pred[:, numpy.newaxis]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
import numpy
from sklearn.base import TransformerMixin


def get_transformer(kind='hard', clf=None):
    """
    Transformer factory
    :param kind: Type of transformer requested
    :param clf: Classifier to transform
    :return:
    """
    if kind == 'hard':
        return HardClassifierTransformer(clf)
    elif kind == 'soft':
        return SoftClassifierTransformer(clf)
    else:
        return ValueError('Only "hard" and "soft" values are allowed.')


class HardClassifierTransformer(TransformerMixin):
    """
    This class is used only for saved models, or models that are already fitted.
    It doesn't use probabilities, just uses predicted labels.
    'fit' method is not implemented because we do not want to re-fit a model that is already
    trained and saved.
    """

    def __init__(self, clf=None):
        self.clf = clf

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        assert self.clf is not None
        pred = self.clf.predict(X)
        return pred[:, numpy.newaxis]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)


class SoftClassifierTransformer(TransformerMixin):
    """
    This class is used only for saved models, or models that are already fitted.
    It uses probabilities.
    'fit' method is not implemented because we do not want to re-fit a model that is already
    trained and saved.
    """

    def __init__(self, clf=None):
        self.clf = clf

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        assert self.clf is not None
        pred = self.clf.predict_proba(X)
        return pred

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

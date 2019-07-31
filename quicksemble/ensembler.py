from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from xgboost.core import XGBoostError

from quicksemble.classifier_transformer import get_transformer
from quicksemble.utils import load_object


class Ensembler():
    """
    A Simple class to create ensemble of models with only two layers.
    First layer is for collection of models. Second layer is for the
    meta model.
    """

    def __init__(self, models:list, modelpaths:list=None, merge_models=False, meta_model=LogisticRegression(), voting='hard'):
        """

        :param models: List of trained/untrained Models
        :param modelpath: List of path of trained and saved Models
        :param merge_models: Merge array of saved and not saved models
        :param meta_model: Model for the second layer. Default is Logistic Regression
        :param voting: 'hard' use prediction values of base layer or 'soft' use predicted probabilities of base layer
        """
        if merge_models:
            assert modelpaths is not None
            assert models is not None
            assert isinstance(models, list)
            model_from_paths = [load_object(paths) for paths in modelpaths]
            self.models = models + model_from_paths
        else:
            # no need to merge. Giving more priority to already trained models.
            if modelpaths is None:
                self.models = models
            else:
                self.models = [load_object(paths) for paths in modelpaths]

        self.meta_model = meta_model
        self.ensemble = None
        self.voting = voting

    def fit_base(self, X, y, mode='unfitted'):
        """
        To fit the ensemble model
        :param X: features
        :param y: labels
        :param mode: unfitted -> to fit only those models that are not trained
                     all -> to fit all models
        :return:
        """

        if mode == 'unfitted':
            for m in self.models:
                try:
                    m.predict(X[0:2]) # just using two rows to make it fast
                except (NotFittedError, XGBoostError) as  nfe:
                    m.fit(X, y)
        elif mode == 'all':
            for m in self.models:
                m.fit(X, y)
        else:
            raise ValueError('Mode can only be "unfitted" or "all". Unknown value passed.')

        return self.models

    def compile(self, n_jobs=2) -> Pipeline:
        """
        To Build the ensemble. This method should be called only if all models in base
        are already fitted.
        :return:
        """
        meta_features = []
        for i, mdl in enumerate(self.models):
            meta_features.append((mdl.__class__.__name__+str(i), get_transformer(kind=self.voting, clf=mdl)))

        self.ensemble = Pipeline(steps=[
            ('base_layer', FeatureUnion(meta_features, n_jobs=n_jobs)),
            ('final_layer', self.meta_model)
        ])

        return self.ensemble

    def fit(self, X, y, mode='unfitted', n_jobs=2):
        self.fit_base(X, y, mode)
        self.compile(n_jobs)
        return self.ensemble.fit(X, y)

    def predict(self, X):
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        """
        Note that this method will only work if meta model has predict_proba
        :param X:
        :return: numpy.ndarray
        """
        return self.ensemble.predict_proba(X)



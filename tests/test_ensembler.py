from unittest import TestCase

import numpy
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from quicksemble.ensembler import Ensembler


class TestEnsembler(TestCase):

    def test_object_constructor(self):
        models = [RandomForestClassifier(), XGBClassifier()]
        ensembler1 = Ensembler(models)
        self.assertListEqual(ensembler1.models, models)

        with self.assertRaises(AssertionError) as ae:
            ensembler2 = Ensembler(models, None, True)

    def test_fit_base(self):
        data = load_iris()
        X = data['data']
        Y = data['target']
        models = [RandomForestClassifier(), XGBClassifier()]
        ensembler1 = Ensembler(models)
        ensembler1.fit_base(X, Y)

        p1 = ensembler1.models[0].predict(X)
        self.assertIsInstance(p1, numpy.ndarray)
        p2 = ensembler1.models[1].predict(X)
        self.assertIsInstance(p2, numpy.ndarray)

    def test_compile(self):
        data = load_iris()
        X = data['data']
        Y = data['target']
        models = [RandomForestClassifier(), XGBClassifier()]
        ensembler1 = Ensembler(models)
        ensembler1.fit_base(X, Y)
        ensemble = ensembler1.compile()
        print(ensemble)

    def test_fit_predict(self):
        data = load_iris()
        X = data['data']
        Y = data['target']
        models = [RandomForestClassifier(), XGBClassifier()]
        ensembler1 = Ensembler(models)
        ensembler1.fit(X, Y)
        pred = ensembler1.predict(X)

        acs = accuracy_score(Y, pred)
        print(acs)

from unittest import TestCase

import numpy
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from quicksemble.ensembler import Ensembler
from quicksemble.utils import save_object


class TestEnsembler(TestCase):

    def setUp(self):
        data = load_iris()
        self.X = data['data']
        self.Y = data['target']

    def test_object_constructor(self):
        models = [RandomForestClassifier(), XGBClassifier()]
        ensembler1 = Ensembler(models)
        self.assertListEqual(ensembler1.models, models)

        with self.assertRaises(AssertionError) as ae:
            Ensembler(models, None, True)

        with self.assertRaises(AssertionError) as ae:
            Ensembler()

    def test_saved_constructor(self):
        rf = RandomForestClassifier()
        rf.fit(self.X, self.Y)
        save_object('rf.pkl', rf)
        xg = XGBClassifier()
        xg.fit(self.X, self.Y)
        save_object('xg.pkl', xg)

        es = Ensembler(modelpaths=[
            'rf.pkl',
            'xg.pkl'
        ])
        es.fit(self.X, self.Y)
        preds = es.predict(self.X)
        self.assertIsInstance(preds, numpy.ndarray)

    def test_fit_base(self):
        models = [RandomForestClassifier(), XGBClassifier()]
        ensembler1 = Ensembler(models)
        ensembler1.fit_base(self.X, self.Y)

        p1 = ensembler1.models[0].predict(self.X)
        self.assertIsInstance(p1, numpy.ndarray)
        p2 = ensembler1.models[1].predict(self.X)
        self.assertIsInstance(p2, numpy.ndarray)

    def test_compile(self):
        models = [RandomForestClassifier(), XGBClassifier()]
        ensembler1 = Ensembler(models)
        ensembler1.fit_base(self.X, self.Y)
        ensemble = ensembler1.compile()
        print(ensemble)

    def test_fit_predict(self):
        models = [RandomForestClassifier(), XGBClassifier()]
        ensembler1 = Ensembler(models)
        ensembler1.fit(self.X, self.Y)
        pred = ensembler1.predict(self.X)

        acs = accuracy_score(self.Y, pred)
        print('Accuracy Score: ', acs)

    def test_fit_predictproba(self):
        models = [RandomForestClassifier(), XGBClassifier()]
        ensembler1 = Ensembler(models, voting='soft')
        ensembler1.fit(self.X, self.Y)
        pred = ensembler1.predict(self.X)

        acs = accuracy_score(self.Y, pred)
        print('Accuracy Score: ', acs)

    def test_intermediary_state_features(self):
        models = [RandomForestClassifier(), XGBClassifier()]
        ensembler1 = Ensembler(models, voting='hard')
        ensembler1.fit(self.X, self.Y)
        meta_features1 = ensembler1.ensemble.named_steps['base_layer'].transform(self.X)
        print(meta_features1.shape)
        self.assertEqual(meta_features1.shape[0], self.X.shape[0])
        self.assertEqual(meta_features1.shape[1], 2)

        ensembler2 = Ensembler(models, voting='soft')
        ensembler2.fit(self.X, self.Y)
        meta_features2 = ensembler2.ensemble.named_steps['base_layer'].transform(self.X)
        print(meta_features2.shape)
        self.assertEqual(meta_features2.shape[0], self.X.shape[0])
        self.assertEqual(meta_features2.shape[1], 6)

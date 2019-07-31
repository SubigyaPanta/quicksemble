from unittest import TestCase

import numpy
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from quicksemble.classifier_transformer import get_transformer, HardClassifierTransformer, SoftClassifierTransformer


class TestHardClassifierTransformer(TestCase):

    def setUp(self):
        data = load_iris()
        self.X = data['data']
        self.Y = data['target']

    def test_get(self):
        tx = get_transformer('hard')
        self.assertIsInstance(tx, HardClassifierTransformer)

    def test_transform(self):
        # When there is no classifier assigned
        tx = get_transformer('hard')
        with self.assertRaises(AssertionError):
            tx.transform(self.X)

        # When there is a classifier assigned
        clf = RandomForestClassifier()
        clf.fit(self.X, self.Y)
        tx = get_transformer('hard', clf)
        transformed = tx.transform(self.X)
        self.assertIsInstance(transformed, numpy.ndarray)
        self.assertEqual(transformed.shape[1], 1)
        self.assertEqual(transformed.shape[0], self.X.shape[0])
        self.assertEqual(2, len(transformed.shape))


class TestSoftClassifierTransformer(TestCase):

    def setUp(self):
        data = load_iris()
        self.X = data['data']
        self.Y = data['target']

    def test_get(self):
        tx = get_transformer('soft')
        self.assertIsInstance(tx, SoftClassifierTransformer)

    def test_transform(self):
        # When there is no classifier assigned
        tx = get_transformer('soft')
        with self.assertRaises(AssertionError):
            tx.transform(self.X)

        # When there is a classifier assigned
        clf = RandomForestClassifier()
        clf.fit(self.X, self.Y)
        tx = get_transformer('soft', clf)
        transformed = tx.transform(self.X)
        print(transformed)
        print(transformed.shape)
        self.assertIsInstance(transformed, numpy.ndarray)
        self.assertEqual(transformed.shape[1], 3)
        self.assertEqual(transformed.shape[0], self.X.shape[0])
        self.assertEqual(2, len(transformed.shape))

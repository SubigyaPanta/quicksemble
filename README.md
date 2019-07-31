# Quicksemble
A simple package to create an ensemble for quick experiments.

### Dependencies
1. Numpy `pip install numpy`
2. Scikit Learn `pip install scikit-learn`
3. Xgboost `pip install xgboost`

## Basic Usage
```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from quicksemble.ensembler import Ensembler

#
# Define train and test dataset here
#

models = [
    RandomForestClassifier(random_state=21),
    XGBClassifier(random_state=21)
]
# Default meta classifier is LogisticRegression. Hence it is weighted voting.
ensemble = Ensembler(models)
ensemble.fit(X_train, y_train)
ensemble.predict(X_test)

```

To change the default meta classifer:
```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from quicksemble.ensembler import Ensembler

#
# Define train and test dataset here
#

models = [
    RandomForestClassifier(random_state=21),
    XGBClassifier(random_state=21)
]

# Use Neural Network as meta classifier
ensemble = Ensembler(models, meta_model=MLPClassifier())
ensemble.fit(X_train, y_train)
ensemble.predict(X_test)
```

By default, Base models use "hard" voting, i.e., it outputs predictions of the 
base models. We can switch it to "soft" voting, i.e., it outputs probabilities
of each class by the base model.

To change voting style:
```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from quicksemble.ensembler import Ensembler

#
# Define train and test dataset here
#

models = [
    RandomForestClassifier(random_state=21),
    XGBClassifier(random_state=21)
]

# Use soft voting. 
ensemble = Ensembler(models, voting='soft')
ensemble.fit(X_train, y_train)
ensemble.predict(X_test)
```

To view output of intermediary state i.e., output of base layers (layer 1)
that is going into meta layer (layer 2). Internally, it uses Pipelines from
scikit-learn. So, feel free to read docs about pipelines.
```python
ensemble = Ensembler(models, voting='soft')
ensemble.fit(X_train, y_train)

# This line will output the values. Note that you need to fit it first.
ensemble.ensemble.named_steps['base_layer'].transform(X_train)
```

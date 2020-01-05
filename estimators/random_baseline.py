from sklearn.dummy import DummyClassifier

from estimators.base_estimator import BaseEstimator

from utils import flatten

class RandomBaseline(BaseEstimator):
    def __init__(self):
        self.params = {
            'description': 'Random (Baseline)',
            'random_params': {
                'strategy': 'uniform'
            }
        }

    def fit(self, debate_data, y):
        x_texts = flatten([x.texts for x in debate_data])
        y = flatten(y)

        self.model = DummyClassifier(**self.params['random_params'])

        self.model.fit(x_texts, y)

    def predict(self, debate_data, proba=False):
        x_texts = flatten([x.texts for x in debate_data])
        test_x = x_texts

        return self.model.predict_proba(test_x) if proba else self.model.predict(test_x)

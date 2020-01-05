from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from estimators.base_estimator import BaseEstimator
from utils import flatten

class SVMNgramBaseline(BaseEstimator):
    def __init__(self):
        self.params = {
            'description': 'SVM ngram (Baseline)',
            'tfidf_params': {
                'ngram_range': (1, 2)
            },
            'svm_params': {
                'random_state': 0,
                'C': 10,
                'gamma': 0.1,
                'kernel': 'rbf',
                'probability': True,
            }
        }

    def fit(self, debate_data, y):
        x_texts = flatten([x.texts for x in debate_data])
        y = flatten(y)

        self.model = Pipeline([
            ('ngrams', TfidfVectorizer(**self.params['tfidf_params'])),
            ('clf', SVC(**self.params['svm_params']))
        ])

        self.model.fit(x_texts, y)

    def predict(self, debate_data, proba=False):
        x_texts = flatten([x.texts for x in debate_data])

        return self.model.predict_proba(x_texts) if proba else self.model.predict(x_texts)

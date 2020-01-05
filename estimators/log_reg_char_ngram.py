from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import get_n_jobs, flatten, grid_search_log_reg_c
from estimators.base_estimator import BaseEstimator

class LogRegCharNGram(BaseEstimator):
    def __init__(self):
        self.params = {
            'description': 'Logistic Regression - Character NGram TF-IDF',
            'tfidf_params': {
                'ngram_range': (2, 7),
                'analyzer': 'char',
            },
            'log_reg': {
                'penalty': 'l2',
                'dual': False,
                'tol': 0.0001,
                'C': 1,
                'fit_intercept': True,
                'intercept_scaling': 1,
                'class_weight': 'balanced',
                'solver': 'sag',
                'max_iter': 20000,
                'multi_class': 'multinomial',
                'random_state': 0,
                'verbose': 0,
                'warm_start': False,
                'n_jobs': get_n_jobs()
            }
        }

    def fit(self, debate_data, y, with_grid_search=False):
        train_x = flatten([x.texts for x in debate_data])
        y = flatten(y)

        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(**self.params['tfidf_params'])),
            ('log_reg', LogisticRegression(**self.params['log_reg']))
        ])

        if with_grid_search:
            self.model = grid_search_log_reg_c(pipe, train_x, y, debate_data, 'macro_recall')
        else:
            self.model = pipe
            self.model.fit(train_x, y)

    def predict(self, debate_data, proba=False):
        test_x = flatten([x.texts for x in debate_data])

        return self.model.predict_proba(test_x) if proba else self.model.predict(test_x)

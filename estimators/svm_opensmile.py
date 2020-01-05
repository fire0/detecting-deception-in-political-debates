from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from estimators.base_estimator import BaseEstimator
from transformers import OpensmileTransformer

class SVMOpensmile(BaseEstimator):
    def __init__(self):
        self.params = {
            'description': 'SVM Opensmile (with parameter selection)',
            'scaler_params': {
                'feature_range': (-1, 1),
            },
            'svm_params': {
                'C': 0.03125,
                'kernel': 'rbf',
                'degree': 3,
                'gamma': 0.0078125,
                'coef0': 0.0,
                'shrinking': True,
                'probability': False,
                'tol': 0.001,
                'cache_size': 200,
                'class_weight': {'FALSE': 1, 'HALF-TRUE': 1, 'TRUE': 1},
                'verbose': True,
                'max_iter': -1,
                'decision_function_shape': 'ovr',
                'random_state': None
            }
        }

    def fit(self, train_x_texts, debates, train_y):
        opensmile = OpensmileTransformer(debates)
        train_x = opensmile.features
        self.model = Pipeline([
            ('scaler', MinMaxScaler(**self.params['scaler_params'])),
            ('clf', SVC(**self.params['svm_params']))
        ])

        self.model.fit(train_x, train_y)

    def predict(self, test_x_texts, debates, proba=False):
        opensmile = OpensmileTransformer(debates)
        test_x = opensmile.features

        return self.model.predict_proba(test_x) if proba else self.model.predict(test_x)

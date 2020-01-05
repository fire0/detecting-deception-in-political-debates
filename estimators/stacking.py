from __future__ import absolute_import, division, print_function

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import flatten, get_n_jobs, grid_search_log_reg_c
from estimators import BaseEstimator, LogRegOpensmile, LogRegLIWC, LogRegNGram, LogRegIVector, LogRegBertClsToken
from metrics import mae, macro_averaged_mae, accuracy, macro_f1, macro_recall

class Stacking(BaseEstimator):
    def __init__(self):
        self.params = {
            'description': 'Meta Stacking',
            'meta_params': {
                'penalty': 'l2',
                'dual': False,
                'tol': 0.0001,
                'C': 100,
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
            },
        }

        self.stack = [
            (LogRegIVector(), False),
            (LogRegOpensmile(), False),
            (LogRegBertClsToken(), False),
            (LogRegNGram(), False),
            (LogRegLIWC(), False),
        ]

        self.params['base_estimators'] = [predictor.params for predictor, _ in self.stack]

    def fit(self, debate_data, y, with_meta_grid_search=True):
        for predictor, grid_search in self.stack:
            if grid_search:
                predictor.fit(debate_data, y, True)

        debate_estimator_sample_probs = []
        debates_len = len(debate_data)
        for index, (test_debate_data, test_labels) in enumerate(zip(debate_data, y)):
            train_debate_data = debate_data[0:index] + debate_data[index + 1:debates_len + 1]
            train_labels = y[0:index] + y[index + 1:debates_len + 1]

            estimator_sample_probs = self.stack_fit_predict(train_debate_data, train_labels, [test_debate_data], [test_labels])
            debate_estimator_sample_probs.append(estimator_sample_probs)

        predictions_zero = list(zip(*debate_estimator_sample_probs))
        predictions_zero = [flatten(x) for x in predictions_zero]

        base_predictions = self.convert_to_meta_input(predictions_zero)

        y_meta = flatten(y)

        self.meta_learner = Pipeline([('log_reg', LogisticRegression(**self.params['meta_params']))])

        if with_meta_grid_search:
            self.meta_learner = grid_search_log_reg_c(self.meta_learner, base_predictions, y_meta, None, 'macro_recall')
        else:
            self.meta_learner.fit(base_predictions, y_meta)

        self.stack_fit_predict(debate_data, y)

    def predict(self, debate_data):
        predictions = []

        for predictor, _ in self.stack:
            prediction = predictor.predict_proba(debate_data).tolist()
            predictions.append(prediction)

        prediction_converted = self.convert_to_meta_input(predictions)

        return self.meta_learner.predict(prediction_converted)

    def stack_fit_predict(self, train_x, train_y, test_x = None, test_y = None):
        predictions = []

        for predictor, _ in self.stack:
            predictor.fit(train_x, train_y)

            if (test_x):
                prediction = predictor.predict_proba(test_x).tolist()
                predictions.append(prediction)

        if (test_x):
            return predictions

    def convert_to_meta_input(self, predictions):
        return list(map(lambda sample_predictions: flatten(sample_predictions), list(zip(*predictions))))

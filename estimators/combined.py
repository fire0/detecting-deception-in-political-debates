from estimators import BaseEstimator, LogRegOpensmile, LogRegLIWC, LogRegNGram, LogRegBertClsToken, LogRegIVector

class Combined(BaseEstimator):
    def __init__(self):
        self.classes = ['FALSE', 'HALF-TRUE', 'TRUE']
        self.params = {
            'description': 'Logistic Regression with LogRegIVector + LogRegOpensmile + LogRegBertClsToken + LogRegNGram + LogRegLIWC: averaged scores',
        }

    def fit(self, debate_data, y):
        self.estimators = [LogRegIVector(), LogRegOpensmile(), LogRegBertClsToken(), LogRegNGram(), LogRegLIWC()]

        for estimator in self.estimators:
            estimator.fit(debate_data, y)

    def predict(self, debate_data):
        estimator_probabilities = [estimator.predict_proba(debate_data) for estimator in self.estimators]

        predictions_final = []

        for predictions_per_sample in zip(*estimator_probabilities):
            avg_predictions = [sum(e)/len(e) for e in zip(*predictions_per_sample)]
            prediction_index = avg_predictions.index(max(avg_predictions))
            predictions_final.append(self.classes[prediction_index])
        return predictions_final

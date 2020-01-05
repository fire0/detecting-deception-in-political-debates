from abc import ABC, abstractmethod

from utils import update_dict

class BaseEstimator(ABC):
    @property
    def params(self):
        return self.__params

    @params.setter
    def params(self, params):
        self.__params = params

    @abstractmethod
    def fit(self, debate_data, y, val_debate_data, val_y): pass

    @abstractmethod
    def predict(self, debate_data, proba=False): pass

    def predict_proba(self, debate_data):
        return self.predict(debate_data, proba=True)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        for key, value in params.items():
            update_dict(self.params, key.split('__'), value)
        return self

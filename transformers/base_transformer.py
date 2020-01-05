from abc import ABC, abstractmethod, abstractproperty

class BaseTransformer(ABC):
    @property
    def names(self):
        return self.__names

    @property
    def features(self):
        return self.__features

    @names.setter
    def names(self, names):
        self.__names = names

    @features.setter
    def features(self, features):
        self.__features = features

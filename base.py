from abc import ABC, abstractmethod
import numpy as np

class Predictor(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

class Cost_function(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def gradient(self):
        pass




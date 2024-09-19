from abc import ABC, abstractmethod
import numpy as np

class Predictor(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass




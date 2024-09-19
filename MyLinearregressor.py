import numpy as np
from base import Predictor
from MyMSE import MyMSE

class MyLinearRegressor(Predictor):
    def __init__(self):
        super().__init__()

    def fit(self, x: np.ndarray, y: np.ndarray):
        self._num_rows = x.shape[0]
        self._vect_of_ones = np.ones((self._num_rows,1))
        self._X = np.concatenate((self._vect_of_ones, x), axis=1)
        self._coefficients = MyMSE().fit(x, y).predict()
        return self
        
    def predict(self, x: np.ndarray) -> np.array:
        self._X = np.concatenate((self._vect_of_ones, x), axis=1)
        return np.dot(self._X, self._coefficients)
    
    def get_coefficients(self):
        return self._coefficients

    def score(self, x: np.ndarray, y: np.ndarray):
        u = np.sum(((y - self.predict(x))**2))
        v = np.sum((y - y.mean())**2)
        return 1 - np.divide(u,v)


if __name__=="__main__":
    x= np.array([2.4,5.0,1.5,3.8,8.7,3.6,1.2,8.1,2.5,5,1.6,1.6,2.4,3.9,5.4]).reshape(-1, 1)
    y = np.array([2.1,4.7,1.7,3.6,8.7,3.2,1.0,8.0,2.4,6,1.1,1.3,2.4,3.9,4.8]).reshape(-1, 1)
    lr = MyLinearRegressor()
    lr.fit(x, y)
    print(lr.get_coefficients())
    print(lr.predict(x))
    print(lr.score(x,y))

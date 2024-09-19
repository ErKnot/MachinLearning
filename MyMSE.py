import numpy as np
from base import Predictor

STARTING_POINT = {
    "origin" : np.zeros,
    "ones" : np.ones,
    "random" : np.random.random
}

class MyMSE(Predictor):
    def __init__(self):
        super().__init__()
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        self._y = y
        self._num_rows = x.shape[0]
        self._num_cols = x.shape[1]
        self._X = np.concatenate((np.ones((self._num_rows,1)), x), axis=1)
        return self

    def mse(self, coefficients: np.ndarray) -> float:
        self._error = (np.dot(self._X, coefficients) - self._y)
        return np.mean( np.dot(self._error, self._error.T))
    
    def mse_gradient(self, coefficients: np.ndarray) -> float:
        self._error = (np.dot(self._X, coefficients) - self._y)
        return np.divide(2, self._num_rows) * np.dot(self._X.T, self._error)
        
    def predict(self, stochastic = False, learning_rate: float = 0.01, n_iterations: int = 1000, criterion: str = 'ones') -> np.array:
        self._predicted_coefficient = STARTING_POINT[criterion]((self._X.shape[1], 1))
        self._history = {
            "Predicted_coefficient": [self._predicted_coefficient],
            "cost_history": [self.mse(self._predicted_coefficient)]
        }
        
        if stochastic:
            self._predicted_coefficient = self._predicted_coefficient - learning_rate 
        
            return self._predicted_coefficient
            
        for iteration in np.arange(n_iterations):
            self._predicted_coefficient = self._predicted_coefficient - learning_rate * self.mse_gradient(self._predicted_coefficient)
            self._history["Predicted_coefficient"].append(self._predicted_coefficient)
            self._history["cost_history"].append(self.mse(self._predicted_coefficient))
        
        return self._predicted_coefficient
    
    def history(self):
        return self._history
            
# 

if __name__ == "__main__":
    x= np.array([2.4,5.0,1.5,3.8,8.7,3.6,1.2,8.1,2.5,5,1.6,1.6,2.4,3.9,5.4]).reshape(-1, 1)
    y = np.array([2.1,4.7,1.7,3.6,8.7,3.2,1.0,8.0,2.4,6,1.1,1.3,2.4,3.9,4.8]).reshape(-1, 1)
    mse = MyMSE()
    mse.fit(x, y)
    print(mse.mse(np.array([[1],[2]])))
    mse.predict(learning_rate=0.01,n_iterations= 1000, criterion='random')
    print(mse.history())
    










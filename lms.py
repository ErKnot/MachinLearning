import numpy as np
from base import Predictor

STARTING_POINT = {
    "origin" : np.zeros,
    "ones" : np.ones,
    "random": np.random.random
}


class Least_square(Predictor):
    def __init__(self):
        super().__init__()


    def fit(self, learning_features: np.ndarray, learning_targets: np.ndarray):
        # Find the theta that minimize the the LMS function, choose between gradient descend or stochastic gradient descent
        self._learning_features = learning_features
        self._learning_targets = learning_targets
        self._num_rows = learning_features.shape[0]
        self._num_cols = learning_features.shape[1]
        self._X = np.concatenate([np.ones((self._num_rows, 1)), learning_features], axis = 1)
        return self


    def gradient(self, coefficients: np.ndarray):
        self._error = (np.dot(self._X, coefficients) - self._learning_targets)
        return  np.dot(self._X.T, self._error)


    def predict(self, gradient_descent = True, learning_rate : float = 0.001, num_iterations: int = 10000, criterion: str = 'origin'):
        # Give the theta that minimize the LMS function

        self._predicted_coefficients = STARTING_POINT[criterion]((self._num_cols + 1, 1))
        self._history = {
            "Predicted_coefficient": [self._predicted_coefficients],
            "cost_history": [self.compute(self._predicted_coefficients)]
        }
        
        for iteration in range(num_iterations):
            self._predicted_coefficients += - learning_rate * self.gradient(self._predicted_coefficients)
            self._history["Predicted_coefficient"].append(self._predicted_coefficients)
            self._history["cost_history"].append(self.compute(self._predicted_coefficients))
        
        return  self._predicted_coefficients


    def history(self):
        return self._history


    def compute(self, coefficients: np.ndarray):
        # Give a point theta and return the resutl of the LMS function
        _error = (np.dot(self._X, coefficients) - self._learning_targets)
        return np.divide(np.dot(_error.T, _error),2).flat[0]




    
if __name__ == "__main__":
    x = np.array([2.4,5.0,1.5,3.8,8.7,3.6,1.2,8.1,2.5,5,1.6,1.6,2.4,3.9,5.4]).reshape(-1, 1)
    y = np.array([2.1,4.7,1.7,3.6,8.7,3.2,1.0,8.0,2.4,6,1.1,1.3,2.4,3.9,4.8]).reshape(-1, 1)
    ls = Least_square()
    ls.fit(x,y)
    print(ls.predict())


    



import numpy as np

STARTING_POINT = {
    "origin" : np.zeros,
    "ones" : np.ones,
    "random": np.random.random
}


class Least_square:
    def __init__(self):
        pass

    
    def gradient(self, theta: np.ndarray):
        self._gradient = np.dot(np.dot(self._X.T,self._X ), theta) - np.dot(self._X.T, self.learning_targets)
        return self._gradient

    def fit(self, learning_features: np.ndarray, learning_targets: np.ndarray):
        # Find the theta that minimize the the LMS function, choose between gradient descend or stochastic gradient descent
        self.learning_features = learning_features
        self.learning_targets = learning_targets
        self.num_rows = learning_features.shape[0]
        self.num_cols = learning_features.shape[1]
        self._X = np.concatenate([np.ones((self.num_rows, 1)), learning_features], axis = 1)
 

    
    def predict(self, gradient_descent = True, learning_rate : float = 0.01, num_iterations: int = 100):
        # Give the theta that minimize the LMS function
        self.step = []
        self._theta = np.random.rand(self.num_rows +1, 1)
        for iteration in range(num_iterations):
            self._gradient = np.dot(np.dot(self._X.T,self._X ), self._theta) - np.dot(self._X.T, self.learning_targets)
            slef._theta = self._theta - learning_rate * self._gradient
        
        return elf


    def compute(self, coefficients: np.ndarray):
        # Give a point theta and return the resutl of the LMS function
        _error = np.dot(self._X, coefficients) - self.learning_targets
        return np.divide(np.dot(_error.T, _error),2).flat[0]

    def predict(self):
        # Give the theta that minimize the LMS function
        pass


    
if __name__ == "__main__":
    lms = Least_square()
    lms.fit(np.array([[2, 3], [4, 3]]), np.array([[1], [2]]))
    print(lms.compute(np.array([1,1,1]).reshape((3,1))))
    print(lms.gradient(np.array([1,1,1]).reshape((3,1))))
    



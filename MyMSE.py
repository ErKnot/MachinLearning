import numpy as np
import matplotlib.pyplot as plt
from base import Predictor



class MyMSE(Predictor):
    def __init__(self):
        super().__init__()
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        self._y = y
        self._num_rows = x.shape[0]
        self._num_cols = x.shape[1]
        self._num_variables = self._num_cols + 1
        self._X = np.concatenate((np.ones((self._num_rows,1)), x), axis=1)
        return self

    def mse(self, coefficients: np.ndarray) -> float:
        self._error = (np.dot(self._X, coefficients) - self._y)
        return np.mean(self._error ** 2) * 2
    
    def gradient(self, coefficients: np.ndarray, x = None, y = None) -> float:
        _X = self._X if x is None else np.concatenate((np.ones((self._num_rows,1)), x), axis=1)
        _y = self._y if y is None else y
        _error = (np.dot(_X, coefficients) - _y)
        return np.divide(np.dot( _X.T, _error) * 2, self._num_rows)

        
    def predict(self, learning_rate: float = 0.01, batch_size: int = 1, n_iterations: int = 1000, dtype: str = "float64", start = None, random_state = None) -> np.array:
        
        # Setting up the data type for numpy array
        dtype_ = np.dtype(dtype)

        # Initializing the random number generator
        seed = None if random_state is None else int(random_state)
        rng = np.random.default_rng(seed = seed)

        # initializing the starting point of the gradient descent algo.
        self._predicted_coefficient = (
            rng.normal(size = self._num_variables).astype(dtype = dtype_).reshape(-1,1)
            if start is None else
            np.array(start, dtype=dtype_).reshape(-1,1)
        )

        # Initializing the dictionary that keep tracks of the the steps of the gradient descet and the cost-function values
        self._history = {
            "Predicted_coefficient": [self._predicted_coefficient],
            "cost_history": [self.mse(self._predicted_coefficient)]
        }


       
        for iteration in np.arange(n_iterations):


            self._predicted_coefficient += - learning_rate * self.gradient(self._predicted_coefficient)
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
    start = np.array([1,1]).reshape(-1,1)
    print(mse.mse(start))
    print("The gradent is:", mse.gradient(start))
#    print(mse.mse(np.array([[1],[2]])))
    b = mse.predict(learning_rate=0.01,n_iterations= 1000)
#    print(mse.history())
    y_pred =  b[0] + b[1] * x
    plt.scatter(x,y)
    plt.plot(x, y_pred)
    plt.show()

    










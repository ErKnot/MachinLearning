import numpy as np
from typing import Union, Self
from base import Cost_function

class MyMSE(Cost_function):
    def __init__(self):
        super().__init__()
    
    def fit(self, x: Union[np.ndarray, list], y: Union[np.ndarray, list], dtype: str = "float64") -> Self:
        # Setting up the data type for Numpy arrays
        self._dtype = np.dtype(dtype)

        # Converting x and y to numpy array
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            y = np.array(y, dtype=self._dtype).reshape(-1,1)
            x = np.array(x, dtype=self._dtype).reshape(y.shape[0],-1)

        # Checing dimension of x and y
        if x.shape[0] != y.shape[0]:
            raise ValueError("'x' and 'y' lengths do not match")

        # Initialize the variables
        self._y = y
        self._num_rows = x.shape[0]
        self._num_cols = x.shape[1]
        self._X = np.concatenate((np.ones((self._num_rows,1)), x), axis=1)
        return self

    def compute(self, arguments: Union[np.ndarray, list]) -> float:

        # Converting argument to a numpy array
        if not isinstance(arguments, np.ndarray):
            arguments = np.array(arguments, dtype=self._dtype).reshape(-1, 1)

        # Checing the dimension of arguments
        if arguments.shape[0] != self._num_cols + 1:
            raise ValueError("'arguments' lengths do not match")

        # Computing the mse function
        self._error = (np.dot(self._X, arguments) - self._y)
        return np.mean(self._error ** 2 )
    
    def gradient(self, arguments: Union[np.ndarray, list]) -> float:

        # Converting argument to a numpy array
        if not isinstance(arguments, np.ndarray):
            arguments = np.array(arguments, dtype=self._dtype).reshape(-1, 1)

        # Checing the dimension of arguments
        if arguments.shape[0] != self._num_cols + 1:
            raise ValueError("'arguments' lengths do not match")

        # Computing the gradient of the mse function
        self._error = (np.dot(self._X, arguments) - self._y)
        return np.divide(2, self._num_rows) * np.dot(self._X.T, self._error)
        
    # def predict(self, stochastic = False, learning_rate: float = 0.01, n_iterations: int = 1000, criterion: str = 'ones') -> np.array:
    #     self._predicted_coefficient = STARTING_POINT[criterion]((self._X.shape[1], 1))
    #     self._history = {
    #         "Predicted_coefficient": [self._predicted_coefficient],
    #         "cost_history": [self.mse(self._predicted_coefficient)]
    #     }
        
    #     if stochastic:
    #         self._predicted_coefficient = self._predicted_coefficient - learning_rate 
        
    #         return self._predicted_coefficient
            
    #     for iteration in np.arange(n_iterations):
    #         self._predicted_coefficient += - learning_rate * self.mse_gradient(self._predicted_coefficient)
    #         self._history["Predicted_coefficient"].append(self._predicted_coefficient)
    #         self._history["cost_history"].append(self.mse(self._predicted_coefficient))
        
    #     return self._predicted_coefficient
    
    # def history(self):
    #    return self._history
            
# 

if __name__ == "__main__":
    # Test with 1 dimensional features (row of x)
    # x= np.array([2.4,5.0,1.5,3.8,8.7,3.6,1.2,8.1,2.5,5,1.6,1.6,2.4,3.9,5.4]).reshape(-1, 1)
    # y = np.array([2.1,4.7,1.7,3.6,8.7,3.2,1.0,8.0,2.4,6,1.1,1.3,2.4,3.9,4.8]).reshape(-1, 1)
    # arguments = np.array([[1],[2]])

    # Test with  two dimensional features (row of x)
    # x= np.array([1,2,1,3]).reshape(-1, 2)
    # y = np.array([1,1]).reshape(-1, 1)
    # arguments = np.array([[1],[1], [1]])

    # Test with thre dimensional features (row of x)
    # x = np.random.randint(5, size=(2, 3))
    # y = np.random.randint(5, size=(2,1))
    # arguments = np.random.randint(5, size=(4, 1))
    # print("'x' is: ", x)
    # print("'y' is: ", y)
    # print("'arguments' is: ", arguments)

    # Test with y being a list:
    ## Dim 1
    # x= [2.4,5.0,1.5,3.8,8.7,3.6,1.2,8.1,2.5,5,1.6,1.6,2.4,3.9,5.4]
    # y = [2.1,4.7,1.7,3.6,8.7,3.2,1.0,8.0,2.4,6,1.1,1.3,2.4,3.9,4.8]
    # arguments = np.array([[1],[2]])

    ## Dim 2
    x= [1,2,1,3]
    y = [1,1]
    arguments = [1,1,1]

    # Test of the class
    mse = MyMSE()
    mse.fit(x, y)
    print("The value of the mse is: ", mse.compute(arguments))
    print("The gradient is: ", mse.gradient(arguments))



    










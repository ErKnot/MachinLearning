import numpy as np
from base import Predictor
from MyMSE import MyMSE
import matplotlib.pyplot as plt
import seaborn as sns

class MyLinearRegressor(Predictor):
    def __init__(self):
        super().__init__()

    def fit(self, x: np.ndarray, y: np.ndarray, cost_function = MyMSE, start=None, learn_rate=0.01, decay_rate=0.0, batch_size=None, n_iter=100, tolerance=1e-06, dtype: str = "float64", random_state=None):
        
        # Initializing the needed variables
        self._num_rows = x.shape[0]
        self._num_cols = x.shape[1]
        self._y = y
        self._Xy = np.c_[x, self._y]
        self._history = {
            "Predicted_coefficient": [],
            "cost_history": []
        }


        # Setting the batch size. If batch_size = None we set it to be equal the number of observations in order to get te batch gradient discent
        batch_size = self._num_rows if batch_size is None else batch_size

        # Checking the batch size
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("'batch_size' must be None, for stochastic gradient descent, or a postive integer")

        # Checing if the cost funcion is callable
        if not callable(cost_function):
            raise TypeError("'cost_function' must be callable")
        
        # Initializing the cost function, if the cost function is not defined the Mean square error function will be used
        cost_function_alg = cost_function()

        # Initializing the random generator
        seed = None if random_state is None else int(random_state)
        rng = np.random.default_rng(seed=seed)

        # Initializing the values of the starting vector of the gradient descent
        self.vector = (
            rng.normal(size=self._num_cols + 1).astype(dtype).reshape(-1, 1)
            if start is None else
            np.array(start, dtype=dtype)
        )


        # Initializing the cost funcion on all the data to use it to update the hystory dict
        cost_function_xy = cost_function().fit(x,y)
        self._history["Predicted_coefficient"].append(self.vector)
        self._history["cost_history"].append(cost_function_xy.compute(self.vector))

        # Initializing the diff in case the decay_rate > 0
        diff = 0

        # Performing the (stocastic) gradient descent
        for _ in range(n_iter):


            # shuffle x and y
            rng.shuffle(self._Xy)

            # Performing minibatch moves
            for start in range(0, self._num_rows, batch_size):
                stop = batch_size + start
                X_batch, y_batch = self._Xy[start:stop, :-1], self._Xy[start:stop, -1:]

                # Calculating the gradiente
                cost_function_gradient = cost_function_alg.fit(X_batch, y_batch).gradient(self.vector)
                diff =decay_rate * diff - learn_rate * cost_function_gradient

                # Checking the absolute value difference is smal enough
                if np.all(np.abs(diff) <= tolerance):
                    break

                # Updating the value of the variable
                self.vector += diff
            # Update the hystry dictionary
            self._history["Predicted_coefficient"].append(self.vector.copy())
            self._history["cost_history"].append(cost_function_xy.compute(self.vector))

        
        return self.vector if self.vector.shape else self.vector.item()

        
    def predict(self, x: np.array) -> np.array:
        x = np.concatenate((np.ones((self._num_rows,1)), x), axis=1)
        return np.dot(x, self.vector)
    
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
    print(lr.fit(x, y, MyMSE, learn_rate=0.01, n_iter = 1000, batch_size=None))
    y_pred = lr.predict(x)
    print(lr._history)
    plt.scatter(x,y)
    plt.plot(x, y_pred, c='r')
    plt.show()
    


import numpy as np
import matplotlib.pyplot as plt 

def gradient_descent(gradient, x, y, start, learn_rate: float, n_iter: int = 50, tolerance: float = 1e-06, dtype=float):
    
    # checking if the gradient is callable
    if not callable(gradient):
        raise TypeError("'gradient' must be callable")
    
    # Setting up the data type for Numpy arrays
    dtype_ = np.dtype(dtype)
    
    # Converting x and y to NumPy arrays
    x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
    if x.shape[0] != y.shape[0]:
        raise ValueError("'x' and 'y' lengths do not match")
    
    # Initializing the values of the variables
    vector = np.array(start, dtype=dtype_)
    
    # Setting up and checking the learning rate
    learn_rate = np.array(learn_rate, dtype=dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn_rate' must be greater than zero")
    
    # Setting up and checking the maximal number of iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than zero")
    
    # Setting up and checking the tolerance
    tolerance = np.array(tolerance, dtype=dtype_)
    if np.any(tolerance <=0):
        raise ValueError("'tolerance' must be greater than zero" )
    
    # Performing the gradient descent loop
    for _ in range(n_iter):
        diff = -learn_rate * np.array(gradient(x,y, vector), dtype_)
        
        if np.all(np.abs(diff) <= tolerance):
            break
        
        vector += diff
    
    return vector if vector.shape else vector.item()


def sgd(gradient, x, y,n_vars=None, start=None, learn_rate=0.1, decay_rate=0.0, batch_size=1, n_iter=50, tolerance=1e-06, dtype="float64", random_state=None):
    
    # checking if the gradient is callable
    if not callable(gradient):
        raise TypeError("'gradient' must be callable")
    
    # Setting up the data type for Numpy arrays
    dtype_ = np.dtype(dtype)
    
    # Converting x and y to NumPy arrays
    x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
    n_obs = x.shape[0]
    if n_obs != y.shape[0]:
        raise ValueError("'x' and 'y' lengths do not match")
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]
    
    # Initializing the random number generator
    seed = None if random_state is None else int(random_state)
    rng = np.random.default_rng(seed=seed)
    
    # Initializing the values of the variables
    vector = (
        rng.normal(size=int(n_vars)).astype(dtype_)
        if start is None else
        np.array(start, dtype=dtype_)
    )
    
    # Setting up and checking the learning rate
    learn_rate = np.array(learn_rate, dtype=dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn_rate' must be greater than zero")
    
    # Setting up the decay rate
    decay_rate = np.array(decay_rate, dtype=dtype_)
    if np.any(decay_rate < 0) or np.any(decay_rate > 1):
        raise ValueError("'decay_rate' must be between zero and one")
    
    # Setting up and checking the maximal number of iterations
    batch_size = int(batch_size)
    if not 0 < batch_size <= n_obs:
        raise ValueError(
            "'batch_size' must be greater than zero and less than "
            "or equal to the number of observations"
        )
    
    # Setting up and checking the maximal number of iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than zero")
    
    # Setting up and checking the tolerance
    tolerance = np.array(tolerance, dtype=dtype_)
    if np.any(tolerance <=0):
        raise ValueError("'tolerance' must be greater than zero" )
    
    # Setting the difference to zero for the first iteration
    diff = 0
    
    # Performing the gradient descent loop
    for _ in range(n_iter):
        # Shuffle x and y
        rng.shuffle(xy)
        
        # Performing minibatch moves
        for start in range(0, n_obs, batch_size):
            stop = batch_size + start
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]
            
            # recalculating the difference
            grad = np.array(gradient(x_batch, y_batch, vector), dtype_)
            diff = decay_rate * diff - learn_rate * grad
            
            # checking the absolute value difference is small enough
            if np.all(np.abs(diff) <= tolerance):
                break
            
            # Updating the values of the variables
            vector += diff
    
    return vector if vector.shape else vector.item()





def ssr_gradient(x, y, b):
    res = b[0] + b[1] * x - y
    return res.mean(), (res * x).mean()


# Function that works for every dimension, but in theory is slower than using numpy vector calculus
def ssr(x, y, b):
    ssr = 0
    for row in range(len(x)):
        ssr += b[0]
        for col in range(len(x[row])):        
            ssr += b[col + 1] * x[row][col]
        ssr += -y[row]

    return ssr

if __name__ == "__main__":
    x = np.array([5, 15, 25, 35, 45, 55])
    y = np.array([5, 20, 14, 32, 22, 38])
#    print(sgd(ssr_gradient, x, y, start=[0.5, 0.5], learn_rate=0.0008,batch_size=5, n_iter=100_000))
#    print(gradient_descent(ssr_gradient, x, y, start=[0.5, 0.5], learn_rate=0.0008, n_iter=100_000))
    print(sgd(ssr_gradient, x, y, n_vars=2, learn_rate=0.0001, decay_rate=0.8, batch_size=3, n_iter=100_000, random_state=0))




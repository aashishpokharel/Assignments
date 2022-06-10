import numpy as np
import pandas as pd


class LinearRegression:

    """ 
    LinearRegression fits a linear model with coefficients w = (w1, …, wp) 
    to minimize the residual sum of squares between the observed targets in the dataset,
    and the targets predicted by the linear approximation.
    """    
    def __init__(self, alpha=0.0001 , max_iters=10000, precision = 1e-3):
        """Initialize the LinearRegression 

        Args:
            alpha (float, optional): Learning Rate. Defaults to 0.0001.
            max_iters (int, optional): Maximum number of iterations. Defaults to 10000.
            precision (_type_, optional): The tolerable error. Defaults to 1e-3.
        """        
        self.alpha = alpha
        self.max_iters = max_iters
        self.precision = precision

    def add_ones(self, X):
        ''' Pad ones in of the X 
        Parameters
        --------------------
        X:{array-like, sparse matrix} of shape (n_samples, n_features)
        Array/matrix to be padded.
        returns:
        ------------------
        X:{array-like, sparse matrix} of shape (n_samples, n_features)
        Padded array/matrix
        '''
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return X


    def fit(self, X, y):
        """
        Fit linear model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.
        Returns
        -------
        self : object
            Fitted Estimator.
        """
        np.random.seed(0)
        X = self.add_ones(X)
        n = X.shape[0]
        d = X.shape[1]

        iteration = 0 
        difference = 1 
        costs = [1e12] 

        betas = None
        gradients = None
        cost = None

        # Initialize betas
        
        betas = np.random.randn(d,1)
    
        while difference > self.precision and iteration <= self.max_iters :
            gradients   = np.dot(X.T, (np.dot(X, betas)- y))
            betas      -= self.alpha * gradients
            cost        = (1/2) * np.sum((np.dot(X,betas)- y) ** 2)


            difference = np.abs(costs[iteration] - cost) 
            costs.append(cost)

            iteration += 1

            if(cost == np.infty):
                print("Maximum cost value exceeded!")
                break

        # return betas, iteration, costs
        self.betas = betas
        self.costs = costs
        self.iteration = iteration

    def predict(self,X):
        """Predict Values using LinearRegression

        Args:
            X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns:
            y : array, shape (n_samples,)
            Returns predicted values.
        """               
        X = self.add_ones(X)
        y = np.dot(X, self.betas)
        return y


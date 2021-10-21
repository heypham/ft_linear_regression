"""
LinearRegression class
class used to train regression model
"""

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression(object):
    """
    Linear Regression using Gradient Descent.
    Resulting thetas for model to be used in predict.py
    """
    def __init__(self):
        """
        Class initializer
        """
        self.mean = 0
        self.stdev = 0
        self.iterations = 0
        self.cost_history = []
        self.theta_history = []
        self.theta = np.zeros(2)

    def feature_scale_normalise(self, x):
        """
        Normalises & Standardise feature vector X so that
        mean    Xnorm = 0
        stdev   Xnorm = 1
        """
        self.mean = x.mean()
        self.stdev = x.std()
        xnorm = (x - self.mean) / self.stdev
        return xnorm

    def cost(self, X, y, theta):
        """
        Calculates cost for given X and y
        the higher the cost, the more inaccurate the theta values are
        """
        m = X.shape[0]
        theta_append = copy.deepcopy(theta)
        self.theta_history.append(theta_append)
        prediction = self.hypothesis(X, theta)
        cost = (1/(2*m) * np.sum(np.square(prediction - y)))
        return cost

    def fit(self, X, y, alpha, iterations):
        """
        Gradient descent algorithm to update theta values
        """
        self.training_dataset = pd.DataFrame({'X': X, 'y': y})
        X = self.feature_scale_normalise(X)
        self.training_dataset['X_norm'] = X
        m = X.shape[0]
        for _ in range(iterations):
            loss = self.hypothesis(X, self.theta) - y
            self.training_dataset['loss'] = loss
            self.theta[0] -= (alpha / m) * np.sum(loss)
            self.theta[1] -= (alpha / m) * np.sum(loss * X)
            self.cost_history.append(self.cost(X, y, self.theta))
        # print(self.training_dataset)
        return self.theta

    def hypothesis(self, X, theta):
        """
        This is valid because X is a single feature vector.
        Otherwise we need to do dot product of X by theta, as well as
        adding a column of 1s in X matrix (to multiply by theta[0])
        """
        ret = X * theta[1] + theta[0]
        return ret

    def train(self, X, y, alpha, iterations):
        """
        Train model with training set
        """
        self.iterations = iterations
        self.fit(X, y, alpha, iterations)

    def show_data(self, X, y):
        """
        Plot data (need to adjust once theta calculation is good)
        """
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6))
        # Plot of cost history
        ax1.plot(range(self.iterations), self.cost_history)
        ax1.set_title("Cost history plot")

        # Plot of real prices vs hypothetical prices
        Xnorm = self.feature_scale_normalise(X)
        ax2.set_title("Evolution of model fitting")
        ax2.plot(X, y, 'b.')
        # Plot of evolution of thetas
        for i in range(len(self.theta_history)):
            if i % 50 == 0:
                ax2.plot(X, self.theta_history[i][0] + Xnorm * self.theta_history[i][1], 'c-', label='_nolegend_')
        ax2.plot(X, self.theta[0] + Xnorm * self.theta[1], 'r-')
        ax2.plot()
        ax2.set(xlabel="$km$")
        ax2.set(ylabel="$price$")
        ax2.legend(['real prices', 'hypothetical prices'], loc='upper right')
        plt.show()

    def predict(self, Xval):
        """
        Predict dollar value of car depending on the km given (Xval)
        """
        Xnorm = (Xval - self.mean) / self.stdev
        prediction = self.hypothesis(Xnorm, self.theta)
        return prediction

    def score(self, X, y):
        """
        Calculate R squared value (accuracy in %) of our model
        """
        y_pred = self.predict(X)
        y_mean = np.full(X.shape[0],[y.mean()])
        diff_y = y_pred - y
        variance = 0
        diff_mean_y = y_mean - y
        total_variance = 0
        for i in diff_y:
            variance += i ** 2
        for i in diff_mean_y:
            total_variance += i ** 2
        return (total_variance - variance)/total_variance

    def validate(self, x, y):
        for i in range(len(x)):
            print("Predicted value for %7d km : $%.2f vs real value : $%d" % (x[i], self.predict(x[i]), y[i]))

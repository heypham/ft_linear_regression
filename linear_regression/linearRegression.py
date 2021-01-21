#/usr/bin/python3

import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

class linearRegression(object):
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
        self.theta = np.zeros(2)
    
    def feature_scale_normalise(self, X):
        """
        Normalises & Standardise feature vector X so that
        mean    Xnorm = 0
        stdev   Xnorm = 1
        """
        self.mean = X.mean()
        self.stdev = X.std()
        Xnorm = (X - self.mean) / self.stdev
        print("mean : ", self.mean, "std : ", self.stdev)
        return Xnorm

    def cost(self, X, y, theta):
        """
        Calculates cost for given X and y
        the higher the cost, the more inaccurate the theta values are
        """
        m = X.shape[0]
        prediction = self.hypothesis(X, theta)
        cost = (1/(2*m) * np.sum(np.square(prediction - y)))
        return cost

    def fit(self, X, y, alpha, iter):
        """
        Gradient descent algorithm to update theta values
        """
        X = self.feature_scale_normalise(X)
        m = X.shape[0]
        cost = []
        for i in range(iter):
            loss = self.hypothesis(X, self.theta) - y
            self.theta[0] -= (alpha / m) * np.sum(loss)
            self.theta[1] -= (alpha / m) * np.sum(loss * X)
            cost.append(self.cost(X, y, self.theta))
        print("theta : ", self.theta)
        return self.theta

    def hypothesis(self, X, theta):
        """
        This is valid because X is a single feature vector.
        Otherwise we need to do dot product of X by theta, as well as
        adding a column of 1s in X matrix (to multiply by theta[0])
        """
        ret = X * theta[1] + theta[0]
        # print("hypothesis ret : ", ret)
        return ret

    def show_data(self, X, y):
        """
        Plot data (need to adjust once theta calculation is good)
        """
        Xnorm = self.feature_scale_normalise(X)
        plt.plot(X, y, 'b.')
        plt.plot(X, self.theta[0] + Xnorm * self.theta[1], 'r-')
        # plt.plot()
        plt.xlabel("$km$", fontsize=18)
        plt.ylabel("$price$", rotation=0, fontsize=18)
        plt.legend(['real prices', 'hypothetical prices'], loc='upper right')
        plt.show()

    def predict(self, Xval):
        """
        Predict dollar value of car depending on the km given (Xval)
        """
        Xnorm = (Xval - self.mean) / self.stdev
        prediction = self.hypothesis(Xnorm, self.theta)
        return prediction

# === MAIN ====
# To move to train.py
data_file = os.path.join("data", "data.csv")
f = pd.read_csv(data_file)
X = np.array(f.get('km'))
y = np.array(f.get('price'))

alpha = 0.01
it = 1000

model = linearRegression()
model.fit(X, y, alpha, it)
model.show_data(X, y)
pickle.dump(model, open("linear_regression_model.42", 'wb'))


# To move to predict.py
# load model
print(model.predict(400000))
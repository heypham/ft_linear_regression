
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression():
    """
    Model class that applies linear regression
    """
    def __init__(self):
        self.theta0 = 0
        self.theta1 = 0
        self.training_dataframe = None

    def feature_scale_normalise(self, X):
        """
        Normalises & Standardise feature vector X so that
        mean    Xnorm = 0
        stdev   Xnorm = 1
        """
        self.mean = X.mean()
        self.stdev = X.std()
        xnorm = (X - self.mean) / self.stdev
        return xnorm

    def hypothesis(self, X_train, theta0, theta1):
        """
        Linear hypothesis : theta0 * X + theta1
        where X = mileage of car array
        """
        y_pred = []
        for count, X in enumerate(X_train):
            y_pred.append(theta1 * X + theta0)
        y_pred = np.array(y_pred)
        return y_pred

    def gradient_descent(self):
        """
        Computation of the gradient descent for predicted price iteration
        X = mileage
        y = actual price
        y_pred = predicted price with hypothesis and current theta0, theta1
        n = sample size of data
        """
        X = self.training_dataframe['normalized_kms']
        y = self.training_dataframe['price']
        y_pred = self.training_dataframe['predicted_price']
        n = len(self.training_dataframe)
        self.training_dataframe['loss_price'] = y_pred - y
        # self.training_dataframe['loss_price*mileage'] = self.training_dataframe['loss_price'] 
        delta_theta0 = (self.lr/n) * sum(self.training_dataframe['loss_price'])
        delta_theta1 = (self.lr/n) * sum(self.training_dataframe['loss_price'] * X)
        return delta_theta0, delta_theta1

    def fit(self, X_train, y_train, lr, it):
        """
        training method.
        Arguments are
        - X_train (data kms)
        - y_train (corresponding prices)
        - learning rate (default 0.01)
        - iterations (default 1000)
        """
        self.lr = lr
        X_train_normalized = self.feature_scale_normalise(X_train)
        self.training_dataframe = pd.DataFrame({'kms': X_train, 'normalized_kms': X_train_normalized, 'price': y_train})
        for i in range(it):
            self.training_dataframe['predicted_price'] = self.hypothesis(X_train, self.theta0, self.theta1)
            delta_theta0, delta_theta1 = self.gradient_descent()
            self.theta0 -= delta_theta0
            self.theta1 -= delta_theta1
        print(self.training_dataframe)
        print(self.theta1, self.theta0)

    def predict(self, thetafile):
        """
        predict method
        Argument is a file containing the weights coming from the training
        """
        return

    def plot(self, X, y):
        """
        Plotting method to see result of linear regression
        """
        plt.scatter(X, y)
        plt.show()

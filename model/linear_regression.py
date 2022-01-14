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
        self.theta_history = []
        self.error_history = []

    def _verify_user_input(self, user_input):
        """
        Make sure the user input is valid (int)
        """
        try:
            km = int(user_input)
            if km < 0:
                raise NameError("You entered a negative value for the amount of kms. If you want me to predict the price correctly you need to make sense.")
            return km
        except ValueError:
            raise NameError("Invalid format for kms input. Make sure you enter an int number.")

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
        y_pred = X_train * theta1 + theta0
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
        delta_theta0 = (self.lr/n) * self.training_dataframe['loss_price'].sum()
        delta_theta1 = (self.lr/n) * (self.training_dataframe['loss_price'] * X).sum()
        self.error_history.append(abs(self.training_dataframe['loss_price'].mean())) # mean absolute error MAE
        # self.error_history.append((np.square(self.training_dataframe['loss_price'])).mean()) # mean squared error
        return delta_theta0, delta_theta1

    def fit(self, X_train, y_train, lr=0.01, it=1000, verbose=False, plot=False):
        """
        training method.
        Arguments are
        - X_train (data kms)
        - y_train (corresponding prices)
        - learning rate (default 0.01)
        - iterations (default 1000)
        """
        self.lr = lr
        self.it = it
        X_train_normalized = self.feature_scale_normalise(X_train)
        self.training_dataframe = pd.DataFrame({'kms': X_train, 'normalized_kms': X_train_normalized, 'price': y_train})
        self.training_dataframe.reset_index(inplace=True, drop=True)
        for i in range(it):
            self.training_dataframe['predicted_price'] = self.hypothesis(X_train_normalized, self.theta0, self.theta1)
            delta_theta0, delta_theta1 = self.gradient_descent()
            if verbose == True:
                if i % (it/10) == 0:
                    print("Iteration {0: <5} : Updating thetas from θ₀ = {1: <19} -> {2: <19}, θ₁ = {3: <19} -> {4: <19}".format(i, self.theta0, (self.theta0 - delta_theta0), self.theta1, (self.theta1 - delta_theta1)))
            self.theta0 -= delta_theta0
            self.theta1 -= delta_theta1
            self.theta_history.append([self.theta0, self.theta1])

    def save_model(self):
        """
        Saving model in pickle and theta values in csv file
        """
        print("Saving final θ values : θ₀ = {}, θ₁ = {}, mean = {}, stdev = {} in weights.csv file.".format(self.theta0, self.theta1, self.mean, self.stdev))
        weights = pd.DataFrame({'theta0': [self.theta0], 'theta1': [self.theta1], 'mean':[self.mean], 'stdev':[self.stdev]})
        weights.to_csv('weights.csv')

    def predict(self, user_input, theta0, theta1, mean, stdev):
        """
        predict method
        Argument is user input for kms to predict price
        """
        kms = self._verify_user_input(user_input)
        kms_norm = (kms - mean) / stdev
        predicted_price = self.hypothesis(kms_norm, theta0, theta1)
        if predicted_price < 0:
            print("{}kms?? Oh well, I'm afraid you won't be able to get much from this car... You might as well give it away.".format(kms))
        else:
            print("The predicted price for a car with {}kms is ${:.2f}".format(kms, predicted_price))

    def plot(self, X, y):
        """
        Plotting method to see result of linear regression
        """
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6))
        # fig, ax2 = plt.subplots(1, figsize=(12, 6))
        # Plot of cost history
        ax1.plot(range(self.it), self.error_history)
        ax1.set_title("Error history plot")
        print(self.error_history)
        print(len(self.error_history))

        # Plot of real prices vs hypothetical prices
        Xnorm = self.feature_scale_normalise(X)
        ax2.set_title("Evolution of model fitting")
        ax2.plot(X, y, 'b.')
        # Plot of evolution of thetas
        for i in range(len(self.theta_history)):
            if i % 100 == 0:
                ax2.plot(X, self.theta_history[i][0] + Xnorm * self.theta_history[i][1], 'c-', label='_nolegend_')
        ax2.plot(X, self.theta0 + Xnorm * self.theta1, 'r-')
        ax2.plot()
        ax2.set(xlabel="$km$")
        ax2.set(ylabel="$price$")
        ax2.legend(['real prices', 'predicted prices'], loc='upper right')
        plt.show()

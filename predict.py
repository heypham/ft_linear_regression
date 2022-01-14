#/usr/bin/python3

import argparse
import pandas as pd
import os.path

from model.linear_regression import LinearRegression

def parse_arguments():
    """
    Parse arguments from command line and stores equation/verbose arguments
    """
    try:
        parser = argparse.ArgumentParser(prog='predict', usage='python3 %(prog)s.py [-h] weights_file', description='Linear regression model predict program')
        parser.add_argument('-w', '--weights', help='weights file', type=str)
        args = parser.parse_args()
        return args
    except Exception as e:
        print(e)

def main():
    try:
        args = parse_arguments()

        weights_file = args.weights
        if weights_file and os.path.exists(weights_file):
            saved_weights = pd.read_csv(weights_file)
        else:
            # raise NameError("File path does not exist. Verify the file exists or run train.py with the dataset.")
            print("No weights.csv file found. Running model training")
            model = LinearRegression()
            data = pd.read_csv('data/data.csv')
            X, y = data.km, data.price
            model.fit(X, y)
            model.save_model()
            saved_weights = pd.read_csv('weights.csv')
        print("How many kilometers has your car?")
        user_input = input()
        model = LinearRegression()
        model.predict(user_input, saved_weights['theta0'][0], saved_weights['theta1'][0], saved_weights['mean'][0], saved_weights['stdev'][0])
    except Exception as e:
        print(e)

if __name__=='__main__':
    main()
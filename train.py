#/usr/bin/python3

import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from sklearn.model_selection import train_test_split
from classes.linearRegression import linearRegression

def parse_arguments():
    try:
        parser = argparse.ArgumentParser(prog='train.py', usage='%(prog)s [-h][-v][-cst][-data datafile.csv]', description='Train the model to predict the price of a car based on the number of kilometers.')
        parser.add_argument('-d', '--datafile', help='.csv file containing the data to train the model', default="data/data.csv")
        parser.add_argument('-v', '--verbose', help='increase output verbosity', type=int, default=0)
        parser.add_argument('-lr', '--learning_rate', help='[default = 0.01]', type=float, default=0.01)
        parser.add_argument('-it', '--iterations', help='[default = 100]', type=int, default=1000)
        args = parser.parse_args()
        return args
    except ValueError as e:
        raise NameError('\n[Input error]\nThere has been an error while parsing the arguments.\n')

def get_data(data_file):
    f = pd.read_csv(data_file)
    X = np.array(f.get('km'))
    y = np.array(f.get('price'))
    return X, y

def main():
    args = parse_arguments()
    X, y = get_data(args.datafile)

    alpha = args.learning_rate
    it = args.iterations

    # Split data into testing and training sets
    x_train, x_test,y_train,y_test = train_test_split(X,y,test_size =0.2)

    model = linearRegression()
    if args.verbose == 1:
        print("Training the model with lr %f and %d iterations" % (alpha, it))
    model.train(x_train, y_train, alpha, it)
    # r2 = model.score(x_test, y_test)
    # print("accuracy : ", r2)
    if args.verbose > 0:
        model.show_data(X, y)
        # model.validate(x_test, y_test)
    pickle.dump(model, open("linear_regression_model.42", 'wb'))

if __name__ == '__main__':
    main()
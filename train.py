#/usr/bin/python3

import argparse
import pandas as pd
from model.linear_regression import LinearRegression
from sklearn.model_selection import train_test_split

def parse_arguments():
    """
    Parse arguments from command line and stores equation/verbose arguments
    """
    try:
        parser = argparse.ArgumentParser(prog='train', usage='python3 %(prog)s.py [-h] csv_datafile', description='Linear regression model training program')
        parser.add_argument('datafile', help='csv file containing mileage and prices', type=str)
        parser.add_argument('-lr', '--learning_rate', help='learning rate (default = 0.01)', type=float, default=0.01)
        parser.add_argument('-it', '--iterations', help='[default = 100]', type=int, default=1000)
        parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
        parser.add_argument('-p', '--plot', help='display function graph', action='store_true')
        args = parser.parse_args()
        return args
    except Exception as e:
        print(e)

def read_csv_file(file):
        """
        Read csv file given as argument and store resulting dataframe
        """
        try:
            data = pd.read_csv(file)
            return data
        except:
            raise NameError("Error LinearRegression.train : Invalid file. Unable to read the provided csv file.")

def main():
    try:
        args = parse_arguments()

        learning_rate = args.learning_rate
        iterations = args.iterations

        data = read_csv_file(args.datafile)
        X = data.km
        y = data.price

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2)

        model = LinearRegression()
        model.fit(X_train, y_train, lr=learning_rate, it=iterations)
    except Exception as e:
        print(e)

if __name__=='__main__':
    main()

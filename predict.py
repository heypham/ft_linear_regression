#/usr/bin/python3

import numpy
# from classes.linear_regression import LinearRegression
import pickle
import re
import sys

def get_saved_model():
    try:
        saved_model = open("linear_regression_model.42", "rb")
        model = pickle.load(saved_model)
        saved_model.close()
        return model
    except:
        raise NameError("Error : No model found. Please train the model first [python train.py]")

def check_user_input(input):
    km = re.sub(r"\s+", "", input)
    try:
        km = int(km)
        return km
    except:
        raise NameError("The value you entered is not a number. Please only use digits.")

def main():
    try:
        model = get_saved_model()
        print("How many kilometers has your car?")
        user_input = input()
        km = check_user_input(user_input)
        if km == 0:
            print("It's brand new. Selling it already!? Well, anyway")
        prediction = model.predict(km)
        if prediction < 0:
            print("Wow, your car is so bad it has a negative value. You might as well throw it away.")
        else:
            print("Your car is worth $%.2f" % prediction)
    except KeyboardInterrupt:
        print("Exiting gracefully.")
        sys.exit(0)
    except NameError as e:
        print(e)


if __name__ == '__main__':
    main()
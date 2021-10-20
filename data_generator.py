#/usr/bin/python3

import argparse
import numpy as np
import pandas as pd
from numpy import random
import csv
import matplotlib.pyplot as plt

def parse_arg():
    try:
        parser = argparse.ArgumentParser(prog='generate_datasets.py', usage='%(prog)s [-h][-dl][-min][-max][-err][-t0][-t1][-file][-plt][-v]', description='Program to generate a dataset to train the linear regression model.')
        parser.add_argument('-min', help='min mileage [default = 0 km]', type=int, default=0)
        parser.add_argument('-max', help='max mileage [default = 200000 km]', type=float, default=200000)
        parser.add_argument('-err', help='desired error in the price output [default = 15%]', type=float, default=15)
        parser.add_argument('-t0', '--tetha0', help='price of a new vehicle (0km) [default = 15000 €]', type=float, default=15000)
        parser.add_argument('-t1', '--tetha1', help='price lost per km [default = -8000 €/km]', type=float, default=-8000)
        parser.add_argument('-file', help='file name to save the generated dataset', type=str, default='generated_data')
        parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
        args = parser.parse_args()
        return args
    except:
        raise NameError('\n[Input error]\nThere has been an error while parsing the arguments.\n')

def generate_km_for_dataset(dataset_len, max, min):
    try:
        km = (random.rand(dataset_len) * (max - min)) + min
        floored_km = []
        for i in km:
            floored_km.append(int(i))
        floored_km = np.array(floored_km)
        KM = (km - km.mean()) / km.std()
        return KM, floored_km
    except NameError as e:
        print(e)
        raise NameError('\n[Process error]\nThere has been an error while generating km for the dataset.\n')

def generate_price_for_dataset(dataset_len, KM, err, tetha0, tetha1):
    try:
        err /= 100
        price = tetha0 + (KM * tetha1)
        price_range = price.max() - price.min()
        noise = random.rand(dataset_len)
        price_noise = (noise - noise.mean()) * (err * price_range)
        for i in range(len(price)):
            if i % 2 == 0:
                price[i] += price_noise[i]
            else:
                price[i] -= price_noise[i]
        return price
    except:
        raise NameError('\n[Process error]\nThere has been an error while generating prices for the dataset.\n')

def generate_dataset(dataset_len, KM, price, filename):
    try:
        with open(filename, 'w') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(['km', 'price'])
            for i in range(dataset_len):
                spamwriter.writerow([KM[i], price[i]])
                i += 1
    except:
        raise NameError('\n[Process error]\nThere has been an error while saving the dataset.\n')

def display_information(dataset_len, min, max, err, tetha0, tetha1, km, filename):
    try:
        print('\n[ Dataset generated ]\nSaved into {}\n'.format(filename))
        print('Number of data   : {}'.format(dataset_len))
        print('Min mileage      : {}'.format(min))
        print('Max mileage      : {}'.format(max))
        print('Price variation  : {}%'.format(err))
        print('Tetha 0          : {:.5f}'.format(tetha0).strip('0').strip('.'))
        print('Tetha 1          : {:.5f}'.format(tetha1).strip('0').strip('.'))
        print('KM mean          : {:.5f}'.format(km.mean()).strip('0').strip())
        print('KM std           : {:.5f}'.format(km.std()).strip('0').strip('.'))
    except NameError as e:
        print(e)
        raise NameError('\n[Process error]\nThere has been an error while displaying the information.\n')

def main():
    try:
        args = parse_arg()
        KM, km = generate_km_for_dataset(100, args.min, args.max)
        price = generate_price_for_dataset(100, KM, args.err, args.tetha0, args.tetha1)
        filename = 'data/{}.csv'.format(args.file)
        generate_dataset(100, km, price, filename)
        if args.verbose:
            display_information(100, args.min, args.max, args.err, args.tetha0, args.tetha1, km, filename)
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()
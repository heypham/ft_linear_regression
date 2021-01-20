#/usr/bin/python3

import pandas as pd
import numpy as np

data = pd.read_csv("data/data.csv")
# print(data)

# Feature matrix (here vectore because only 1 feature)
X = np.array([data.get('km')]).T
y = np.array(data.get('price')).reshape(-1,1)

# Theta
theta = np.zeros(2).reshape(-1,1)

# Print dimensions of X and theta (24, 1) (2, 1)
print("X dimension : ", X.shape)
print("Theta dimension : ", theta.shape)

# To allow matrix multiplication between X and theta we need X to have dim (24, 2)
# Add an extra column at index 0 full of 1s.
X = np.insert(X, 0, 1, axis=1)
# print(X)
print(X.shape)
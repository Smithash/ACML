import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

# # URL of the Iris dataset on UCI Machine Learning Repository
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# # Column names for the dataset
# column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# # Load the dataset into a pandas DataFrame
# iris_data = pd.read_csv(url, names=column_names)

# #Save to CSV file
# iris_data.to_csv("iris_dataset.csv", index=False)


#Load dataset into a input array X and target array T
data = pd.read_csv('./lab 2/iris_dataset.csv')

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
T = data[['class']]

# print(T)
ohe = OneHotEncoder()
transformed = ohe.fit_transform(data[['class']])
T = transformed.toarray()

print(X)

#change target dataset to a one-hot encoding 

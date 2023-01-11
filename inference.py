import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
# from sklearn.linear_model import LogisticRegression

from Dataset import DatasetManager
from Model import build_and_compile_model
from plot import *


# Data to fetch from the workbook:
# [x]:
# - ShapeSize
# - #input
# - #output
# - #op
# y:
# - RealNdpxCost

# if using xlsx data
# dataset = DatasetManager.fetch_xlsx_data()
# dataset.columns = dataset.iloc[0]
# dataset = dataset[1:]

# else
dataset = DatasetManager.fetch_data("Dataset/new_ndpx_dataset.csv")
dataset = dataset.drop(dataset[dataset.RealNdpxCost > 1000000].index)

print(dataset)

# for shuffling dataset
dataset = shuffle(dataset)

# generate training & testing dataset & labels
test_dataset = dataset
test_dataset = test_dataset.astype(float)

test_features = test_dataset.copy()

test_labels = test_features.pop('RealNdpxCost')

# generate model
dnn_model = tf.keras.models.load_model("dnn_model")

# test
test_results = {}
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
print(f"test_results: {test_results}")

pd.DataFrame(test_results, index=['Mean absolute error [RealNdpxCost]']).T

# predictions
test_predictions = dnn_model.predict(test_features).flatten()
plot_prediction(test_predictions, test_labels)

# get R2 score
print(type(test_labels))
print(type(test_predictions))
metric = tfa.metrics.r_square.RSquare()
metric.update_state(test_labels, test_predictions)
result = metric.result()
print(f"R2 Score: {result.numpy()}")
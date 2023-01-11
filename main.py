import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from sklearn.utils import shuffle
# from sklearn.linear_model import LogisticRegression

from Dataset import DatasetManager
from Model import *
from plot import *

np.set_printoptions(precision=3, suppress=True) # for easier read

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
# dataset = DatasetManager.fetch_csv_data()
dataset = DatasetManager.fetch_data("Dataset/new_ndpx_dataset.csv")

print(dataset)

# for shuffling dataset
dataset = shuffle(dataset)

# generate training & testing dataset & labels
train_dataset = dataset.sample(frac=0.8, random_state=1)
train_dataset = train_dataset.astype(float)
test_dataset = dataset.drop(train_dataset.index)
test_dataset = test_dataset.astype(float)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('RealNdpxCost')
test_labels = test_features.pop('RealNdpxCost')

normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

# # generate model
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

# train model
history = dnn_model.fit(
  train_features,
  train_labels,
  validation_split=0.25,
  verbose=1,
  epochs=1000
)

# draw loss
plot_loss(history)

# test
test_results = {}
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
# predictions
pd.DataFrame(test_results, index=['Mean absolute error [RealNdpxCost]']).T
test_predictions = dnn_model.predict(test_features).flatten()
# get R2 score
metric = tfa.metrics.r_square.RSquare()
metric.update_state(test_labels, test_predictions)
result = metric.result()
print(f"R2 Score: {result.numpy()}")
# plot
plot_prediction(test_predictions, test_labels)

# overall plotting
prediction_results = {}
dataset_labels = dataset.pop('RealNdpxCost')
prediction_results['dnn_model'] = dnn_model.evaluate(dataset, dataset_labels, verbose=0)
print(prediction_results)
pd.DataFrame(prediction_results, index=['Mean absolute error [RealNdpxCost]']).T
dataset_predictions = dnn_model.predict(dataset).flatten()
plot_all(history, dataset_predictions, dataset_labels)

# save model
dnn_model.save('dnn_model')

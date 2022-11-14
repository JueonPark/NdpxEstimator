import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
# from sklearn.linear_model import LogisticRegression

from Dataset import DatasetManager
from Model import build_and_compile_model

def plot_prediction(test_predictions, test_labels):
  plt.clf()
  a = plt.axes(aspect='equal')
  plt.scatter(test_predictions, test_labels)
  plt.xlabel('Predictions [RealNdpxCost]')
  plt.ylabel('True Values [RealNdpxCost]')
  lims = [0, 1000000]
  plt.xlim(lims)
  plt.ylim(lims)
  plt.plot(lims, lims)
  plt.tight_layout()
  plt.show()
  plt.savefig("result_graph.png")


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
dataset = DatasetManager.fetch_csv_data()

print(dataset)

# for removing N/A data
dataset = dataset.dropna()

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
print(test_results)

pd.DataFrame(test_results, index=['Mean absolute error [RealNdpxCost]']).T

# predictions
test_predictions = dnn_model.predict(test_features).flatten()
plot_prediction(test_predictions, test_labels)

# save model
dnn_model.save('dnn_model')

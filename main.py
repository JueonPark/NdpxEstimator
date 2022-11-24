import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
# from sklearn.linear_model import LogisticRegression

from Dataset import DatasetManager
from Model import *

np.set_printoptions(precision=3, suppress=True) # for easier read

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss', color='#225ea8')
  plt.plot(history.history['val_loss'], label='val_loss', color="#fe9929")
  plt.ylim([0, 200000])
  plt.xlabel('Epoch')
  plt.ylabel('Error [Mean Absolute Error]')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()
  plt.savefig("train_loss.pdf")

def plot_prediction(test_predictions, test_labels):
  plt.clf()
  plt.axes(aspect='equal')
  plt.scatter(test_predictions, test_labels, c='#225ea8')
  plt.xlabel('Predictions [Predicted NDPX Cost]')
  plt.ylabel('True Values [Real NDPX Cost]')
  lims = [0, 1000000]
  plt.xlim(lims)
  plt.ylim(lims)
  plt.plot(lims, lims)
  plt.tight_layout()
  plt.show()
  plt.savefig("train_prediction.pdf")

def plot_all(history, test_predictions, test_labels):
  plt.clf()
  plt.figure(figsize=(18,8), tight_layout=True)
  loss_ax = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
  loss_ax.set_title("Training Loss", fontsize=32)
  pred_ax = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=2)
  pred_ax.set_title("Prediction", fontsize=32)
  
  # plot loss first
  # loss_ax.set_title("Training Loss")
  loss_ax.plot(history.history['loss'], label='loss', color='#225ea8')
  loss_ax.plot(history.history['val_loss'], label='val_loss', color="#fe9929")
  loss_ax.set_ylim([0, 200000])
  loss_ax.set_xlabel('Epoch', fontsize=20)
  loss_ax.set_ylabel('Error [Real NDPX Cost]', fontsize=20)
  loss_ax.tick_params(labelsize=14)
  loss_ax.legend()
  loss_ax.grid(True)
  # loss_ax.tight_layout()
  # then, plot predictions
  # plt.subplot(1, 2, 2)
  # pred_ax.set_title('Prediction')
  # pred_ax.axes(aspect='equal')
  pred_ax.scatter(test_predictions, test_labels, c='#225ea8')
  pred_ax.set_xlabel('Predictions [Predicted NDPX Cost]', fontsize=20)
  pred_ax.set_ylabel('True Values [Real NDPX Cost]', fontsize=20)
  lims = [0, 1000000]
  pred_ax.set_xlim(lims)
  pred_ax.set_ylim(lims)
  pred_ax.tick_params(labelsize=14)
  pred_ax.plot(lims, lims)
  # pred_ax.tight_layout()
  # plot all
  # fig.tight_layout()
  plt.show()
  plt.savefig("overall_results.pdf")

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
  validation_split=0.2,
  verbose=1,
  epochs=400
)

# draw loss
plot_loss(history)

# test
test_results = {}
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
# predictions
pd.DataFrame(test_results, index=['Mean absolute error [RealNdpxCost]']).T
test_predictions = dnn_model.predict(test_features).flatten()
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

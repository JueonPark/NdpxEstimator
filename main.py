import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Dataset import DatasetManager

np.set_printoptions(precision=3, suppress=True) # for easier read

def build_and_compile_model(norm):
  model = tf.keras.Sequential([
    norm,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 100000])
  plt.xlabel('Epoch')
  plt.ylabel('Error [RealNdpxCost]')
  plt.legend()
  plt.grid(True)
  plt.show()
  plt.savefig("train_loss.png")

# Data to fetch from the workbook:
# [x]:
# - ShapeSize
# - #input
# - #output
# - #op
# y:
# - RealNdpxCost
dataset = DatasetManager.fetch_data()
dataset.columns = dataset.iloc[0]
dataset = dataset[1:]
print(dataset)

# for removing N/A data
dataset = dataset.dropna()

# generate training & testing dataset & labels
train_dataset = dataset.sample(frac=0.8, random_state=0)
train_dataset = train_dataset.astype(float)
test_dataset = dataset.drop(train_dataset.index)
test_dataset = train_dataset.astype(float)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('RealNdpxCost')
test_labels = test_features.pop('RealNdpxCost')

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

# generate model
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

# train model
history = dnn_model.fit(
  train_features,
  train_labels,
  validation_split=0.2,
  verbose=0, epochs=1000)

plot_loss(history)

# test
test_results = {}
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
print(test_results)

pd.DataFrame(test_results, index=['Mean absolute error [RealNdpxCost]']).T

# predictions
test_predictions = dnn_model.predict(test_features).flatten()

plt.clf()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [RealNdpxCost]')
plt.ylabel('Predictions [RealNdpxCost]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()
plt.savefig("result_graph.png")

dnn_model.save('dnn_model')

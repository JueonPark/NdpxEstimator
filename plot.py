import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# baseline dataset: 60 for MSLE, 200000 for MAE
# new dataset: 100 for MSLE, 700000 for MAE
LOSS_LIMIT = 100
PREDICTION_LIMIT = 1000000

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss', color='#225ea8')
  plt.plot(history.history['val_loss'], label='val_loss', color="#fe9929")
  plt.ylim([0, LOSS_LIMIT])
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
  lims = [0, PREDICTION_LIMIT]
  plt.xlim(lims)
  plt.ylim(lims)
  plt.plot(lims, lims)
  plt.tight_layout()
  plt.show()
  plt.savefig("train_prediction.pdf")

def plot_all(history, test_predictions, test_labels):
  plt.clf()
  # cm = 1/2.54 
  plt.figure(figsize=(18,8), tight_layout=True)
  loss_ax = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
  # loss_ax.set_title("Training Loss", fontsize=36)
  pred_ax = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=2)
  # pred_ax.set_title("Prediction", fontsize=36)
  
  # plot loss first
  loss_ax.plot(history.history['loss'], label='loss', color='#225ea8')
  loss_ax.plot(history.history['val_loss'], label='val_loss', color="#fe9929")
  def killo(x, pos):
    'The two args are the value and tick position'
    return '%dK' % (x * 1e-3)
  formatter = FuncFormatter(killo)
  loss_ax.yaxis.set_major_formatter(formatter)
  loss_ax.set_ylim([0, LOSS_LIMIT])
  loss_ax.set_xlabel('Epoch', fontsize=28)
  loss_ax.set_ylabel('Error', fontsize=28)
  loss_ax.tick_params(labelsize=28)
  loss_ax.legend(fontsize=28)
  loss_ax.grid(True)
  
  # then, plot predictions
  pred_ax.scatter(test_predictions, test_labels, c='#225ea8')
  pred_ax.set_xlabel('Predicted NDPX Cycle', fontsize=28)
  pred_ax.set_ylabel('Real NDPX Cycle', fontsize=28)
  lims = [0, PREDICTION_LIMIT]
  pred_ax.set_xlim(lims)
  pred_ax.set_ylim(lims)
  def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fM' % (x * 1e-6)

  formatter = FuncFormatter(millions)
  pred_ax.xaxis.set_major_formatter(formatter)
  pred_ax.yaxis.set_major_formatter(formatter)
  # remove the first and the last labels
  xticks = pred_ax.xaxis.get_major_ticks()
  xticks[0].set_visible(False)
  yticks = pred_ax.yaxis.get_major_ticks()
  yticks[0].set_visible(False)

  pred_ax.tick_params(labelsize=28)
  pred_ax.plot(lims, lims)
  plt.show()
  plt.savefig("overall_results.pdf")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.utils.class_weight import compute_class_weight

# Read the data set containing credit card history. There are 1M data entries with 8 columns regarding:
# 1. Distance from home when transaction happened
# 2. Distance from last transaction
# 3. Ratio of transaction purchase amount to median purchase amount
# 4. Is the transaction from the same retailer?
# 5. Is the transaction through the chip?
# 6. Did the transaction use a PIN number?
# 7. Is the transaction an online order?

# 8. Is the transaction fraudulent?

# Dataset obtained from https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud

def createDataFrame(fileName):
  creditCardData = pd.read_csv(fileName)

createDataFrame('card_transaction_history.csv')

# Splits the data into the training and test sets. a 80/20 split is used here

# Model will use columns 1-7 in the dataset to predict the value of the 8th column - is it fraudulent?
x = creditCardData.drop('fraud', axis=1)
y = creditCardData['fraud']

x_train = tf.constant(x[:round(len(creditCardData)*0.8)])
y_train = tf.constant(y[:round(len(creditCardData)*0.8)])

x_test = tf.constant(x[round(len(creditCardData)*0.8):])
y_test = tf.constant(y[round(len(creditCardData)*0.8):])

# We have an imbalanced dataset, so we will assign class weights to the minority group (fraudulent purchases)
y_train_np = y_train.numpy()

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train_np),
                                     y=y_train_np)
class_weights = dict(enumerate(class_weights))

# Builds a Sequential model for binary classification

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),

              # We use F1 score instead of accuracy since we want a good balance of precision and recall when it comes to credit card fraud
              metrics=[tf.keras.metrics.F1Score(),
                       'accuracy',
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall()])

# Fit the model for 10 evolutions

model.fit(x_train, y_train, epochs=10, class_weight=class_weights)

# Evaluate model performance

accuracy = model.evaluate(x_test, y_test)

# Creates a tensor of model predictions given the test set

y_preds = model.predict(x_test)

# Function to create a confusion matrix given a test set and predicted values

def createCFMatrix(y_test, y_preds):
  figsize = (10, 10)

  # Create the confusion matrix
  cm = confusion_matrix(y_test, tf.round(y_preds))
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0]

  # Let's prettify it
  fig, ax = plt.subplots(figsize=figsize)
  # Create a matrix plot
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.axes.Axes.matshow.html
  fig.colorbar(cax)

  # Create classes
  classes = False

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label the axes
  ax.set(title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels)

  # Set x-axis labels to bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Adjust label size
  ax.xaxis.label.set_size(20)
  ax.yaxis.label.set_size(20)
  ax.title.set_size(20)

  # Set threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            size=15)

createCFMatrix(y_test, y_preds)

predictions = tf.squeeze(tf.round(model.predict(x_test)))

def predTable(predictions):
  # Create a table to represent your results
  table = pd.DataFrame(predictions)
  table.replace({0.0: 'Legitimate', 1.0: 'Fraud'}, inplace=True)
  table.columns = ['Status']
  print(table)

def findFraudPurchaseNumbers(predictions):
  fraudulent_purchases = []

  # Iterates through model predictions and isolates the indices where fraudulent purchases are made
  # WARNING: Can be time consuming on large datasets
  for i in range(len(predictions)):
    if predictions[i] == 1.0:
      fraudulent_purchases.append(i+1)

  # The numbers in these array correspond to the purchase number in your purchase history, where 1 is the first purchase from the top, 2 is the second, etc
  print(fraudulent_purchases)

predTable(predictions)
findFraudPurchaseNumbers(predictions)

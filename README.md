# Credit Card Fraud Detection Using A Deep Neural Network (DNN)

This is a deep learning model designed to monitor credit card transaction history and activity in order to flag fraudulent purchases

## Performance
The model performs with an accuracy of **99.6%** over a test set of 200,000 purchases.

**F1 Score:** 0.51

**Precision:** 0.98

**Recall:** 0.98

```bash
loss: 0.0103 - f1_score: 0.5068 - accuracy: 0.9964 - precision: 0.9757 - recall: 0.9830
```

## Trying the model
Download FraudDetectionModel and add to your local directory. Then load the model using tensorflow

```python
import tensorflow as tf
loaded_model = tf.keras.models.load_model("FraudDetectionModel")
```

## Confusion Matrix

![Confusion Matrix](confusion_matrix_ccfraud.png)

## License

[MIT](https://choosealicense.com/licenses/mit/)

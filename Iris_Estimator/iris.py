import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float32)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3)

classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

new_samples = np.array([[6.4,3.2,4.5,1.5], [6.3,3.3,6.0,2.5]], dtype=np.float32)
y =list(classifier.predict(new_samples))
print('Predictions: {}' .format(y))



import load_datacode
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import tensorflow as tf

#learning rate for the model
LEARNING_RATE=0.1

def model_fn(features, targets, mode, params):
    l1 = tf.contrib.layers.relu(features, 10)
    l2 = tf.contrib.layers.relu(l1, 10)
    output_layer = tf.contrib.layers.fully_connected(inputs=l2, num_outputs=1, activation_fn=None)
    predictions = tf.reshape(output_layer, [-1])
    predictions_dict = {"ages": predictions}
    loss = tf.losses.mean_squared_error(targets, predictions)
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer="Adam")
    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op)

def get_train_inputs():
    x = tf.constant(train_set.data)
    y = tf.constant(train_set.target)
    return x, y

def get_test_inputs():
    x = tf.constant(val_set.data)
    y = tf.constant(val_set.target)
    return x, y

train_set, val_set, test_set = load_datacode.load_data()
model_params = {"learning_rate": LEARNING_RATE}
nn = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params)

nn.fit(input_fn=get_train_inputs, steps=5000)

ev = nn.evaluate(input_fn=get_test_inputs, steps=1)
print("Loss: %s" % ev["loss"])

predictions = nn.predict(x=test_set.data, as_iterable=True)
print("Target values: %s" %test_set.target)

for i, p in enumerate(predictions):
    print("Prediction %s: %s" % (i + 1, p["ages"]))

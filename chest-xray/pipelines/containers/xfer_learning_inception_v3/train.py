import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
import os


tf.logging.set_verbosity(tf.logging.INFO)

def inceptionv3_model_fn(features, labels, mode):
    # Load Inception-v3 model.
    module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
    input_layer = features["image_data"]
    outputs = module(input_layer)

    logits = tf.layers.dense(inputs=outputs, units=15)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def dataset_input_fn(filenames, num_epochs=None):
  dataset = tf.data.TFRecordDataset(filenames)

  # Use `tf.parse_single_example()` to extract data from a `tf.Example`
  # protocol buffer, and perform any additional per-record preprocessing.
  def parser(record):
    keys_to_features = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
        "image/class/label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    features = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    image = tf.cast(image, tf.float32)

    #image_shape = tf.stack([299, 299, 1])
    image = tf.reshape(image, [height, width, 1])
    image = tf.image.grayscale_to_rgb(image)

    #image = tf.image.encode_jpeg(image, format='grayscale', quality=100)

    #image = tf.cast(image, tf.float32)
    label = tf.cast(features["image/class/label"], tf.int32)

    return {"image_data": image }, label

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(32)
  dataset = dataset.repeat(num_epochs)

  # Each element of `dataset` is tuple containing a dictionary of features
  # (in which each value is a batch of values for that feature), and a batch of
  # labels.
  return dataset

def amain(unused_argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--eval", type=str, required=True)
    app_args = parser.parse_args()

    train_file = app_args.train
    eval_file = app_args.test

    with tf.Graph().as_default() as g:
        # Create an estimator
        classifier = tf.estimator.Estimator(
            model_fn=inceptionv3_model_fn, model_dir="/tmp/convnet_model")

        # Set up logging for predictions
        #tensors_to_log = {"probabilities": "softmax_tensor"}
        #logging_hook = tf.train.LoggingTensorHook(
            #tensors=tensors_to_log, every_n_iter=10)

        # Train network.
        classifier.train(
            input_fn=lambda: dataset_input_fn([train_file],100),
            steps=500,
            #hooks=[logging_hook]
        )

        # Evaluate the model and print results.
        eval_results = classifier.evaluate(input_fn=lambda: dataset_input_fn([eval_file]))
        print(eval_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    app_args = parser.parse_args()

    train_file = app_args.train_file
    eval_file = app_args.eval_file
    dirs = os.listdir( "." )
    for file in dirs:
        print(file)


    with tf.Graph().as_default() as g:
        # Create an estimator
        classifier = tf.estimator.Estimator(
            model_fn=inceptionv3_model_fn, model_dir="/tmp/convnet_model")

        # Set up logging for predictions
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)

        # Train network.
        classifier.train(
            input_fn=lambda: dataset_input_fn([train_file],100),
            steps=500,
            hooks=[logging_hook])

        # Evaluate the model and print results.
        eval_results = classifier.evaluate(input_fn=lambda: dataset_input_fn([eval_file]))
        print(eval_results)
    #tf.app.run()


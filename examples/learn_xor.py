"""Build a binary classifier that learns an XOR using tensorflow."""

import argparse

try:
    import matplotlib.pyplot as plot
except ImportError:
    pass

import numpy as np
import tensorflow as tf

sm = tf.saved_model

def get_training_data():
    """Generate training data."""
    n_examples = 200
    inputs = 2 * np.random.random(size=(n_examples, 2)) - 1
    outputs = np.where(inputs[:, 0] * inputs[:, 1] < 0, 1.0, 0)
    return np.concatenate([inputs, outputs.reshape(n_examples, 1)], 1)

def add_layer(inp, in_rows, in_cols, out_cols, name, act=tf.sigmoid):
    """Add a NN layer with specified activation function."""
    weights = tf.Variable(np.random.normal(size=(in_cols, out_cols)),
                          name=name + "_w")
    bias = tf.Variable(np.random.normal(size=(in_rows, out_cols)),
                       name=name + "_b")
    return act(tf.matmul(inp, weights) + bias, name=name)

def safe_log(inp):
    """Log that replaces 0 input by a small value."""
    return tf.log(tf.maximum(inp, 1e-15))

def crossentropy_loss(prediction, actual):
    """Binary crossentropy loss."""
    return (-actual * safe_log(prediction) -
            (1 - actual) * safe_log(1 - prediction))

def get_graph(layers):
    """Create a single layer nn."""
    input_placeholder = inp = tf.placeholder(tf.float64, (1, 2), "input")
    target_placeholder = tf.placeholder(tf.float64, (1, 1), "target")
    cols = 2

    for i, layer_size in enumerate(layers):
        inp = add_layer(inp, 1, cols, layer_size, "layer_%d" % i)
        cols = layer_size

    output = add_layer(inp, 1, cols, 1, "out")
    loss = crossentropy_loss(output, target_placeholder)
    return input_placeholder, target_placeholder, output, loss

def prediction_heat_map(sess, input_ph, output_op, grid_resolution):
    """Plot predictions as heat map."""
    predictions = np.zeros((grid_resolution, grid_resolution))
    for grid_x in xrange(grid_resolution):
        for grid_y in xrange(grid_resolution):
            x = (grid_x + 0.5) * 2.0 / grid_resolution - 1
            y = (grid_y + 0.5) * 2.0 / grid_resolution - 1
            val = sess.run(
                output_op,
                feed_dict={input_ph: [[x, y]]})
            predictions[grid_x, grid_y] = val
    plot.imshow(predictions.transpose(), extent=[-1, 1, -1, 1],
                interpolation="bicubic")

def get_parser():
    """Return argument parser."""
    parser = argparse.ArgumentParser(
        description="Train a simple XOR classifier.")
    parser.add_argument("--job-dir", type=str,
                        help="Google cloud job dir")
    return parser

def main():
    """Train a simple deep NN to recognize XORs."""
    args = get_parser().parse_args()

    training_data = get_training_data()
    inp, target, output, loss = get_graph([4, 2])
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    opt_op = opt.minimize(loss)

    def to_feed_dict(datum):
        """Transform input format to feed dict."""
        return {
            inp: datum[0:2].reshape(1, 2),
            target: datum[2].reshape(1, 1)}

    def avg_loss():
        """Compute average loss."""
        total_loss = 0.0
        for datum in training_data:
            total_loss += sess.run(loss, feed_dict=to_feed_dict(datum))[0, 0]
        return total_loss / len(training_data)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epochs = 100
        for i in xrange(epochs):
            print("Epoch %d, average loss %f" % (i, avg_loss()))
            for datum in training_data:
                opt_op.run(to_feed_dict(datum))

        try:
            prediction_heat_map(sess, inp, output, 32)
            plot.scatter(
                [datum[0] for datum in training_data],
                [datum[1] for datum in training_data], c=[datum[2] for datum in training_data])
            plot.show()
        except NameError:
            print("No matplotlib. Skipping plot.")

        if args.job_dir:
            print("Writing model to path %s" % args.job_dir)
            builder = sm.builder.SavedModelBuilder(args.job_dir)
            sig_def_map = {
                "serving_default": sm.signature_def_utils.build_signature_def(
                    inputs={
                        "input": sm.utils.build_tensor_info(inp)},
                    outputs={
                        "output": sm.utils.build_tensor_info(output)})}
            builder.add_meta_graph_and_variables(
                sess,
                [sm.tag_constants.TRAINING],
                signature_def_map=sig_def_map)
            builder.save()

if __name__ == "__main__":
    main()

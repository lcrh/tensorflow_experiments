"""Build a (bad) numeric sequence predictor using a simple RNN."""

import random

import numpy as np
import tensorflow as tf

def get_training_data():
    """Count up to nine and down again forever."""
    data_length = 36
    num_examples = 18
    training_data = np.zeros((num_examples, data_length))

    sequence = np.concatenate(
        [np.arange(10), np.arange(0, 9)[::-1],
         np.arange(1, 10), np.arange(1, 9)[::-1]])

    for index in xrange(num_examples):
        training_data[index] = np.roll(sequence, -index, 0)
    return training_data

def rnn_layer(input_frame, rnn_state, rnn_weights, rnn_bias, name=None):
    """Build a fully connected RNN layer."""
    # Shape (1, input_width + state_width)
    all_layer_inputs = tf.reshape(
        tf.concat([input_frame, rnn_state], 0),
        [1, input_frame.get_shape()[0].value + rnn_state.get_shape()[0].value])
    matmul = tf.reshape(
        tf.matmul(all_layer_inputs, rnn_weights),
        (rnn_state.get_shape()[0].value,))
    return tf.sigmoid(matmul + rnn_bias, name=name)

def loss(prediction, target):
    """Squared difference loss."""
    return tf.reduce_sum(
        tf.squared_difference(prediction, target))

def rnn(unroll_depth, width, input_width):
    """Create a fully connected RNN."""
    input_sequence = tf.placeholder(
        tf.float64, (unroll_depth, input_width), name="input")
    rnn_weights = tf.Variable(
        np.random.normal(size=(width + input_width, width)), name="weights")
    output_weights = tf.Variable(
        np.random.normal(size=(width,)), name="out_weights")
    rnn_bias = tf.Variable(
        np.random.normal(size=(width,)), name="bias")
    rnn_init_state = tf.Variable(
        np.random.normal(size=(width,)), name="init")
    targets = tf.placeholder(
        tf.float64, (unroll_depth, input_width), name="targets")

    rnn_states = list()
    losses = list()
    predictions = list()
    rnn_state = rnn_init_state
    for i in xrange(unroll_depth):
        rnn_state = rnn_layer(input_sequence[i], rnn_state,
                              rnn_weights, rnn_bias)
        rnn_states.append(rnn_state)
        prediction = tf.matmul(tf.reshape(rnn_state, (1, width)),
                               tf.reshape(output_weights, (width, 1)))[0, 0]
        predictions.append(prediction)
        losses.append(loss(prediction, targets[i]))

    tf.stack(losses, name="losses")
    tf.stack(predictions, name="predictions")

    # Add L1 regularization term:
    reg_term = tf.add_n(
        [tf.reduce_sum(tf.abs(rnn_bias)),
         tf.reduce_sum(tf.abs(rnn_weights)),
         tf.reduce_sum(tf.abs(output_weights))],
        name="regularization_term") * 0.01

    tf.add_n(losses + [reg_term], name="total_loss")

    # Create 1-frame model for generating sequences:
    predict_input = tf.placeholder(tf.float64, (1,), name="predict_input")
    predict_state = tf.placeholder(tf.float64, (width,), name="predict_state")
    predict_layer = rnn_layer(predict_input, predict_state,
                              rnn_weights, rnn_bias, name="predict_layer")
    predict_out = tf.matmul(tf.reshape(predict_layer, (1, width)),
                            tf.reshape(output_weights, (width, 1)))
    tf.reshape(predict_out, (1,), name="predict_out")

def predict_sequence(sess, init_elem, length):
    """Create a sequence prediction."""
    result = [init_elem]
    state = sess.run("init:0")
    prediction = [init_elem]
    for _ in xrange(length):
        prediction, state = sess.run(
            ["predict_out:0", "predict_layer:0"],
            {"predict_input:0": prediction,
             "predict_state:0": state})
        assert prediction
        result.append(prediction[0])
    return result

def pretty_float_vec(vec):
    """Pretty print a float vector."""
    return ", ".join("%4.2f" % num for num in vec)

def print_unrolled_example(sess, sequence):
    """Print input sequence / predictions / loss on the unrolled model."""
    unroll_depth = tf.get_default_graph().get_tensor_by_name(
        "input:0").get_shape()[0].value
    feed_dict = {
        "input:0": sequence[0:unroll_depth].reshape(unroll_depth, 1),
        "targets:0": sequence[1:unroll_depth + 1].reshape(unroll_depth, 1)
    }
    inputs, losses, results, targets = sess.run(
        ["input:0", "losses:0", "predictions:0", "targets:0"], feed_dict)
    print("I: %s" % pretty_float_vec(inputs))
    print("T: %s" % pretty_float_vec(targets))
    print("R: %s" % pretty_float_vec(results))
    print("L: %s" % pretty_float_vec(losses))

def shuffled(inp):
    """Yield input list as shuffled."""
    shuffled_inp = list(inp)
    random.shuffle(shuffled_inp)
    for elem in shuffled_inp:
        yield elem

def main():
    """Train a woefully underequipped sequence predictor."""
    data = get_training_data()
    unroll_depth = 10
    rnn(unroll_depth, 5, 1)
    epoch_ph = tf.placeholder(tf.float64, (1,), "epoch")
    learning_rate = 0.05 / (1 + 0.05 * epoch_ph[0])

    total_loss = tf.get_default_graph().get_tensor_by_name("total_loss:0")

    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    opt_op = opt.minimize(total_loss)

    def to_feed(sequence, epoch=0):
        """Translate sequence to feed_dict."""
        return {
            "input:0": sequence[0:unroll_depth].reshape(
                unroll_depth, 1),
            "targets:0": sequence[1:unroll_depth + 1].reshape(
                unroll_depth, 1),
            "epoch:0": [epoch]}

    def avg_loss(sess):
        """Compute average loss."""
        sum_loss = 0.0
        for sequence in data:
            sum_loss += sess.run(
                total_loss, to_feed(sequence))
        return sum_loss / len(data)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epochs = 10000
        for i in xrange(epochs):
            print("Epoch %d, average loss %g" % (i, avg_loss(sess)))
            for sequence in shuffled(data):
                opt_op.run(to_feed(sequence, i))

        for i, seq in enumerate(data):
            print("==== Input: %d" % i)
            print_unrolled_example(sess, seq)

        print("Generated:")
        for i in xrange(10):
            # Pick random starting element.
            start = int(np.random.random() * 10)
            seq = predict_sequence(sess, start, 20)
            print(pretty_float_vec(seq))

if __name__ == "__main__":
    main()

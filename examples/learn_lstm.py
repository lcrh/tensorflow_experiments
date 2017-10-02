"""Build a (better) numeric sequence predictor using an LSTM."""

import random

import numpy as np
import tensorflow as tf

_MAX_INPUT = 9

def get_training_data():
    """Count up modulo ten with a 0 every third step."""
    data_length = 50
    num_examples = 18
    training_data = [
        np.concatenate([
            [0], np.arange(i, i + data_length - 1) % 9 + 1
        ])
        for i in xrange(num_examples)
    ]

    for seq in training_data:
        # Every 3rd element is 0.
        seq[::3] = 0

    return training_data

def nn_layer(input_tensor,
             output_columns,
             name,
             params=None,
             act_fun=tf.sigmoid):
    """Create a fully connected nn-layer."""
    input_row_dim, input_col_dim = input_tensor.get_shape()
    input_rows, input_cols = input_row_dim.value, input_col_dim.value
    params = params or (
        tf.Variable(
            np.random.normal(size=(input_cols, output_columns)),
            name=name + "_w"),
        tf.Variable(
            np.random.normal(size=(input_rows, output_columns)),
            name=name + "_b"))

    weights, bias = params
    return params, act_fun(tf.matmul(input_tensor, weights) + bias)

def nn_layers(input_tensor,
              widths,
              name,
              params=None,
              act_fun=tf.sigmoid):
    """Create multi-layered net."""
    params = params or [None] * len(widths)
    for i, width in enumerate(widths):
        params[i], input_tensor = nn_layer(
            input_tensor, width, "%s_l%d" % (name, i), params[i], act_fun)
    return params, input_tensor

def lstm_layer(state_tensor, lstm_output, input_tensor, name, params=None):
    """Create an LSTM layer. Return (params, state, lstm_output)."""
    all_inputs = tf.concat([input_tensor, lstm_output], 1)
    _, state_col_dim = state_tensor.get_shape()
    params = params or [None] * 3
    params[0], forget = nn_layer(
        all_inputs, state_col_dim.value, name + "_forget", params[0])
    params[1], store = nn_layer(
        all_inputs, state_col_dim.value, name + "_store", params[1])
    store_vals = tf.tanh(store)
    next_state = store * store_vals + forget * state_tensor
    params[2], output_select = nn_layer(
        all_inputs, state_col_dim.value, name + "_select", params[2])
    output = tf.tanh(next_state) * output_select
    return (params, next_state, output)

def graph_layer(input_tensor, prev_lstm_state, prev_lstm_output,
                name, params=None):
    """Create full net. Return (params, lstm_state, lstm_output, output)."""
    params = params or [None, None, None, None]

    input_cols = input_tensor.get_shape()[1].value

    params[0], preproc_net = nn_layers(
        input_tensor, [10],
        name + "_preproc", params[0], act_fun=tf.sigmoid)

    params[1], lstm_state, lstm_output = lstm_layer(
        prev_lstm_state, prev_lstm_output, preproc_net,
        name + "_lstm", params[1])

    params[2], post_proc = nn_layers(
        lstm_output, [10],
        name + "_preproc", params[2], act_fun=tf.sigmoid,)

    params[3], output = nn_layers(
        post_proc, [input_cols],
        name + "_output", params[3], act_fun=tf.nn.softmax)

    return (params, lstm_state, lstm_output, output)

def get_prediction_graph(n_states, params=None):
    """Creates a graph for one-step sequence predictions.
    Returns (params, input_placeholder,
             lstm_state_placeholder,
             lstm_output_placeholder,
             lstm_state,
             lstm_output,
             output_pred).
    """
    input_ = tf.placeholder(tf.int32, tuple())
    input_onehot = tf.reshape(
        tf.one_hot(input_, depth=_MAX_INPUT + 1, dtype=tf.float64),
        (1, _MAX_INPUT + 1))
    lstm_state = tf.placeholder(tf.float64, (1, n_states),
                                name="pred_lstm_state")
    lstm_output = tf.placeholder(tf.float64, (1, n_states),
                                 name="pred_lstm_output")

    params, next_state, next_output, output = graph_layer(
        input_onehot, lstm_state, lstm_output, "net", params)
    return (params, input_, lstm_state, lstm_output,
            next_state, next_output, output)

def predict_sequence(sess, input_tensor, prediction_tensor,
                     lstm_state_tensor, lstm_output_tensor,
                     lstm_next_state_tensor, lstm_next_output_tensor,
                     init_lstm_state, init_lstm_output,
                     length):
    """Predict most likely sequence."""
    result = [0]
    lstm_state = sess.run(init_lstm_state)
    lstm_output = sess.run(init_lstm_output)
    for _ in xrange(length):
        feed_dict = {
            input_tensor: result[-1],
            lstm_state_tensor: lstm_state,
            lstm_output_tensor: lstm_output}
        predictions, lstm_state, lstm_output = sess.run(
            [prediction_tensor,
             lstm_next_state_tensor,
             lstm_next_output_tensor],
            feed_dict)
        most_likely = np.argmax(predictions)
        result.append(most_likely)
    return result

def pretty_float_vec(vec):
    """Pretty print a float vector."""
    return ", ".join("%4.2f" % num for num in vec)

def safe_log(inp):
    """Log that replaces 0 input by a small value."""
    return tf.log(tf.maximum(inp, 1e-15))

def crossentropy_loss(prediction, actual):
    """Binary crossentropy loss."""
    loss = (-actual * safe_log(prediction) -
            (1 - actual) * safe_log(1 - prediction))
    return loss

def unroll_lstm(depth, init_lstm_state, init_lstm_output, params):
    """Returns (input_frames, target_frames, loss)."""
    lstm_state, lstm_output = init_lstm_state, init_lstm_output
    input_frames = tf.placeholder(tf.int32, (depth,), name="input_frames")
    target_frames = tf.placeholder(tf.int32, (depth,), name="target_frames")

    losses = []
    for i in xrange(depth):
        params, lstm_state, lstm_output, output = graph_layer(
            tf.reshape(
                tf.one_hot(input_frames[i], _MAX_INPUT + 1, dtype=tf.float64),
                (1, _MAX_INPUT + 1)),
            lstm_state, lstm_output, "layer_%d" % i, params)
        losses.append(
            tf.reduce_sum(
                crossentropy_loss(
                    tf.reshape(output, (output.get_shape()[1].value,)),
                    tf.one_hot(target_frames[i],
                               _MAX_INPUT + 1, dtype=tf.float64))))
    total_loss = tf.add_n(losses)
    return input_frames, target_frames, total_loss

def main():
    """Train LSTM to predict numeric sequence."""
    (params, input_tensor,
     lstm_state_tensor, lstm_output_tensor,
     lstm_next_state_tensor, lstm_next_output_tensor,
     prediction) = get_prediction_graph(10)

    init_lstm_state = tf.Variable(np.random.normal(size=(1, 10)))
    init_lstm_output = tf.Variable(np.random.normal(size=(1, 10)))

    unroll_depth = 30
    input_frames, target_frames, loss = unroll_lstm(
        unroll_depth, init_lstm_state, init_lstm_output, params)

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    opt_op = opt.minimize(loss)

    training_data = get_training_data()
    def feed(sequence):
        """Turn sequence into (input, target) feed."""
        return {
            input_frames: sequence[0:unroll_depth],
            target_frames: sequence[1:unroll_depth+1]}

    def avg_loss(sess):
        """Compute average loss."""
        loss_sum = 0.0
        for seq in training_data:
            loss_sum += sess.run(loss, feed(seq))
        return loss_sum / len(training_data)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Train
        epochs = 100
        for i in xrange(epochs):
            print "Epoch %d, average loss %.2f" % (i, avg_loss(sess))
            random.shuffle(training_data)
            for seq in training_data:
                opt_op.run(feed(seq))
            # Generate
            seq = predict_sequence(
                sess, input_tensor, prediction,
                lstm_state_tensor, lstm_output_tensor,
                lstm_next_state_tensor, lstm_next_output_tensor,
                init_lstm_state, init_lstm_output,
                20)
            print "Example: " + pretty_float_vec(seq)

if __name__ == "__main__":
    main()

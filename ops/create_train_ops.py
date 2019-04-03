import tensorflow as tf

def create_train_ops_st(reduced_loss, args):
    variables_to_train = tf.trainable_variables()
    lower_conv_to_train = [v for v in variables_to_train if 'fc8' not in v.name]

    fc8_w_to_train = [v for v in variables_to_train if 'fc8' in v.name and 'weights' in v.name]  # lr * 10
    fc8_b_to_train = [v for v in variables_to_train if 'fc8' in v.name and 'biases' in v.name]  # lr * 20

    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))

    opt_lower_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    opt_fc8_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    opt_fc8_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

    grads = tf.gradients(reduced_loss, lower_conv_to_train + fc8_w_to_train + fc8_b_to_train)
    grads_lower_conv = grads[:len(lower_conv_to_train)]
    grads_fc8_w = grads[len(lower_conv_to_train): len(lower_conv_to_train) + len(fc8_w_to_train)]
    grads_fc8_b = grads[len(lower_conv_to_train) + len(fc8_w_to_train):]

    train_op_lower_conv = opt_lower_conv.apply_gradients(zip(grads_lower_conv, lower_conv_to_train))
    train_op_fc8_w = opt_fc8_w.apply_gradients(zip(grads_fc8_w, fc8_w_to_train))
    train_op_fc8_b = opt_fc8_b.apply_gradients(zip(grads_fc8_b, fc8_b_to_train))

    train_op = tf.group(train_op_lower_conv, train_op_fc8_w, train_op_fc8_b)

    return train_op, step_ph


def create_train_ops_mt(reduced_loss, args):
    variables_to_train = tf.trainable_variables()
    if 'nddr' in args.network:
        nddr_to_train = [v for v in variables_to_train if 'nddr' in v.name]

    variables_to_train = [v for v in variables_to_train if 'nddr' not in v.name]
    lower_conv_to_train = [v for v in variables_to_train if 'fc8' not in v.name]

    fc8_w_to_train = [v for v in variables_to_train if 'fc8' in v.name and 'weights' in v.name]  # lr * 10
    fc8_b_to_train = [v for v in variables_to_train if 'fc8' in v.name and 'biases' in v.name]  # lr * 20

    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))

    if 'nddr' in args.network:
        opt_nddr = tf.train.MomentumOptimizer(learning_rate * args.nddr_mult, args.momentum)

    opt_lower_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    opt_fc8_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    opt_fc8_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

    if 'nddr' in args.network:
        grads = tf.gradients(reduced_loss, lower_conv_to_train + fc8_w_to_train + fc8_b_to_train + nddr_to_train)
    else:
        grads = tf.gradients(reduced_loss, lower_conv_to_train + fc8_w_to_train + fc8_b_to_train)

    grads_lower_conv = grads[:len(lower_conv_to_train)]
    grads_fc8_w = grads[len(lower_conv_to_train): len(lower_conv_to_train) + len(fc8_w_to_train)]
    grads_fc8_b = grads[
                  len(lower_conv_to_train) + len(fc8_w_to_train): len(lower_conv_to_train) + len(fc8_w_to_train) + len(
                      fc8_b_to_train)]

    if 'nddr' in args.network:
        grads_nddr = grads[len(lower_conv_to_train) + len(fc8_w_to_train) + len(fc8_b_to_train):]

    train_op_lower_conv = opt_lower_conv.apply_gradients(zip(grads_lower_conv, lower_conv_to_train))

    train_op_fc8_w = opt_fc8_w.apply_gradients(zip(grads_fc8_w, fc8_w_to_train))
    train_op_fc8_b = opt_fc8_b.apply_gradients(zip(grads_fc8_b, fc8_b_to_train))

    if 'nddr' in args.network:
        train_op_nddr = opt_nddr.apply_gradients(zip(grads_nddr, nddr_to_train))
        train_op = tf.group(train_op_lower_conv, train_op_fc8_w, train_op_fc8_b, train_op_nddr)
    else:
        train_op = tf.group(train_op_lower_conv, train_op_fc8_w, train_op_fc8_b)

    return train_op, step_ph, learning_rate
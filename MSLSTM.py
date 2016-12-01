import tensorflow as tf
def sess_init():
    init = tf.initialize_all_tables()
    config = tf.ConfigProto()
    config.gpu_option.allow_groth = True
    sess = tf.Session(config=config)
    sess.run(init)
    return sess

def train():
    with tf.Graph().as_default():
        glo
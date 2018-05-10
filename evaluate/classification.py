# classification.py

import tensorflow as tf

class Classification:
    """docstring for Classification"""
    def __init__(self, **kwargs):
        super(Classification, self).__init__()

    def __call__(self, logits, targets):
        predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
        targets = tf.cast(targets, tf.int64)
        batch_size = int(logits.shape[0])
        return tf.reduce_sum(tf.cast(tf.equal(predictions, targets), dtype=tf.float32)) / batch_size

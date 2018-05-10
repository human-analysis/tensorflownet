# classfication.py

import tensorflow as tf

class Classification:
    def __init__(self):
        self.loss = tf.losses.sparse_softmax_cross_entropy

    def __call__(self, outputs, targets):
        loss = self.loss(labels=targets, logits=outputs)
        return loss

# model.py

import math
import models
import losses
import evaluate
import tensorflow as tf

class Model:
    def __init__(self, args):
        self.args = args
        self.model_type = args.model_type
        self.model_options = args.model_options
        self.loss_type = args.loss_type
        self.loss_options = args.loss_options
        self.evaluation_type = args.evaluation_type
        self.evaluation_options = args.evaluation_options

    def setup(self, checkpoints):
        model = getattr(models, self.model_type)(self.args, **self.model_options)
        criterion = getattr(losses, self.loss_type)(**self.loss_options)
        evaluation = getattr(evaluate, self.evaluation_type)(**self.evaluation_options)

        ckpt = checkpoints.latest()
        if ckpt == None:
            pass
        else:
            model = checkpoints.load(model, ckpt)

        # initialize a sample input to build the model for the first time
        batch = tf.zeros((1, self.args.nchannels, self.args.resolution_high, self.args.resolution_wide))
        model(batch)

        return model, criterion, evaluation

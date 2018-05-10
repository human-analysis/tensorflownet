# checkpoints.py

import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe

class Checkpoints:
    def __init__(self, args):
        self.save_path = args.save
        self.resume_path = args.resume_path
        self.save_results = args.save_results

        if self.save_results and not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def latest(self):
        return tf.train.latest_checkpoint(self.resume_path)

    def save(self, epoch, model, best):
        model_objects = {'model': model}
        if best is True:
            ckpt = tfe.Checkpoint(**model_objects)
            ckpt.save('%s/model_epoch_%d' % (self.save_path, epoch))

    def load(self, model, filename):
        model_objects = {'model': model}
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = tfe.Checkpoint(**model_objects)
        ckpt.restore(filename)

        return model_objects['model']

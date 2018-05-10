from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config
import os
import sys
import time
import datetime
import tensorflow as tf
import utils
import random
from model import Model
from train import Trainer
from test import Tester
from dataloader import Dataloader
from checkpoints import Checkpoints

def main():
    # parse the arguments
    args = config.parse_args()
    random.seed(args.manual_seed)
    tf.set_random_seed(args.manual_seed)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S/')
    args.save = os.path.join(args.result_path, now, 'save')
    args.logs = os.path.join(args.result_path, now, 'logs')
    if args.save_results:
        utils.saveargs(args)

    # initialize the checkpoint class
    checkpoints = Checkpoints(args)

    # Create Model
    models = Model(args)
    model, criterion, evaluation = models.setup(checkpoints)

    # initialize a sample input to build the model for the first time and print its summary
    batch = tf.zeros((1, args.nchannels, args.resolution_high, args.resolution_wide))
    model(batch)
    print('Model summary: {}'.format(model.name))
    print(model.summary())

    # Data Loading
    dataloader_obj = Dataloader(args)
    dataloader = dataloader_obj.create()

    # initialize trainer and tester
    trainer = Trainer(args, model, criterion, evaluation)
    tester = Tester(args, model, criterion, evaluation)

    # start training !!!
    loss_best = 1e10
    for epoch in range(args.nepochs):
        print('\nEpoch %d/%d\n' % (epoch + 1, args.nepochs))

        # train for a single epoch
        print("Training...")
        loss_train = trainer.train(epoch, dataloader["train"])
        print("Testing...")
        loss_test = tester.test(epoch, dataloader["test"])

        if loss_best > loss_test:
            model_best = True
            loss_best = loss_test
            if args.save_results:
                checkpoints.save(epoch, model, model_best)

if __name__ == "__main__":
    utils.setup_graceful_exit()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        utils.cleanup()

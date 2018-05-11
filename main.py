from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config
import os
import sys
import traceback
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
    # Parse the Arguments
    args = config.parse_args()
    random.seed(args.manual_seed)
    tf.set_random_seed(args.manual_seed)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S/')
    args.save = os.path.join(args.result_path, now, 'save')
    args.logs = os.path.join(args.result_path, now, 'logs')
    if args.save_results:
        utils.saveargs(args)

    # Initialize the Checkpoints Class
    checkpoints = Checkpoints(args)

    # Create Model
    models = Model(args)
    model, criterion, evaluation = models.setup(checkpoints)

    # Print Model Summary
    print('Model summary: {}'.format(model.name))
    print(model.summary())

    # Data Loading
    dataloader_obj = Dataloader(args)
    dataloader = dataloader_obj.create()

    # Initialize Trainer and Tester
    trainer = Trainer(args, model, criterion, evaluation)
    tester = Tester(args, model, criterion, evaluation)

    # Start Training !!!
    loss_best = 1e10
    for epoch in range(args.nepochs):
        print('\nEpoch %d/%d' % (epoch + 1, args.nepochs))

        # Train and Test for a Single Epoch
        loss_train = trainer.train(epoch, dataloader["train"])
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

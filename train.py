# train.py

import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import plugins

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

class Trainer:
    """docstring for Trainer"""
    def __init__(self, args, model, criterion, evaluation):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation
        self.save_results = args.save_results
        self.save_path = args.save
        self.log_type = args.log_type

        self.cuda = args.cuda
        if self.cuda:
            self.device = "/gpu:0"
        else:
            self.device = "/cpu:0"
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size
        self.step_counter = tf.train.get_or_create_global_step()

        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide
        self.nc = args.nchannels

        self.lr = args.learning_rate
        self.optim_method = args.optim_method
        self.optim_options = args.optim_options
        self.scheduler_method = args.scheduler_method
        self.scheduler_options = args.scheduler_options

        if self.optim_method == "SGD":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        # summary writing
        self.summary_writer = tf.contrib.summary.create_summary_file_writer(self.save_path, flush_millis=1000)

        self.params = {}
        self.training_params = ['Loss', 'Accuracy']
        if self.log_type == 'traditional':
            # display training progress
            self.print_formatter = 'Train [%d/%d][%d/%d] '
            for item in self.training_params:
                self.print_formatter += item + " %.4f "
        elif self.log_type == 'progressbar':
            # progress bar message formatter
            self.print_formatter = '({}/{})' \
                                   ' Load: {:.6f}s' \
                                   ' | Process: {:.3f}s' \
                                   ' | Total: {:}' \
                                   ' | ETA: {:}'
            for item in self.training_params:
                self.print_formatter += ' | ' + item + ' {:.4f}'
            self.print_formatter += ' | lr: {:.2e}'

    def train(self, epoch, dataloader):
        dataset, data_len = dataloader[0], dataloader[1]
        if self.log_type == 'progressbar':
            # Progress bar
            processed_data_len = 0
            bar = plugins.Bar('{:<10}'.format('Train'), max=data_len)
        end = time.time()

        with self.summary_writer.as_default():
            with tf.device(self.device):
                for i, (inputs, labels) in enumerate(tfe.Iterator(dataset)):
                    # keeps track of data loading time
                    data_time = time.time() - end

                    ############################
                    # Update network
                    ############################
                    inputs = tf.reshape(inputs, shape=(-1, self.nc, self.resolution_high, self.resolution_wide))
                    with tf.device('/cpu:0'):
                        tf.assign_add(self.step_counter, 1)

                    with tf.contrib.summary.always_record_summaries():
                        with tf.GradientTape() as tape:
                            # get outputs
                            outputs = self.model(inputs)

                            # compute loss
                            loss = self.criterion(outputs, labels)

                            # perform evaluation
                            accuracy = self.evaluation(outputs, labels)

                            # logging and visualization
                            tf.contrib.summary.scalar('loss', loss)
                            tf.contrib.summary.scalar('accuracy', accuracy)
                            self.params['Loss'] = loss.cpu()._numpy()
                            self.params['Accuracy'] = accuracy.cpu()._numpy()

                        # compute gradients
                        grads = tape.gradient(loss, self.model.variables)

                        # optimize the network
                        self.optimizer.apply_gradients(zip(grads, self.model.variables), global_step=self.step_counter)

                        # print the progress
                        if self.log_type == 'traditional':
                            # print batch progress
                            print(self.print_formatter % tuple(
                                [epoch + 1, self.nepochs, i, data_len] +
                                [self.params[key] for key in self.training_params]))
                        elif self.log_type == 'progressbar':
                            # update progress bar
                            batch_time = time.time() - end
                            processed_data_len += inputs._shape_as_list()[0]

                            bar.suffix = self.print_formatter.format(
                                *[processed_data_len, data_len, data_time,
                                  batch_time, bar.elapsed_td, bar.eta_td] +
                                 [self.params[key] for key in self.training_params] +
                                 [self.optimizer._learning_rate]
                            )
                            bar.next()
                            end = time.time()

                if self.log_type == 'progressbar':
                    bar.finish()

        return loss.cpu()._numpy()

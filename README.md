
# Welcome to *TensorFlowNet*!

****TensorFlowNet**** is a Machine Learning framework that is built on top of [TensorFlow](https://github.com/tensorflow/tensorflow) and it uses TensorFlow's [Eager](https://www.tensorflow.org/programmers_guide/eager) framework for fast research and experimentation. Visualization is done using [TensorBoard](https://github.com/tensorflow/tensorboard).

TensorFlowNet is easy to be customized by creating the necessary classes:
 1. **Data Loading**: a dataset class is required to load the data.
 2. **Model Design**: a tf.keras.Model class that represents the network model.
 3. **Loss Method**: an appropriate class for the loss, for example CrossEntropyLoss or MSELoss.
 4. **Evaluation Metric**: a class to measure the accuracy of the results.

# Structure
TensorFlowNet consists of the following packages:
## Datasets
This is for loading and transforming datasets.
## Models
Network models are kept in this package. It already includes [ResNet](https://arxiv.org/abs/1512.03385), [PreActResNet](https://arxiv.org/abs/1603.05027), [Stacked Hourglass](https://arxiv.org/abs/1603.06937) and [SphereFace](https://arxiv.org/abs/1704.08063).
## Losses
There are number of different choices available for Classification or Regression. New loss methods can be put here.
## Evaluates
There are number of different choices available for Classification or Regression. New accuracy metrics can be put here.
## Plugins
As of now, the following plugins are available:
1. **ProgressBar**:
## Root
 - main
 - dataloader
 - checkpoints
 - model
 - train
 - test

# Setup
First, you need to download TensorFlowNet by calling the following command:
> git clone --recursive https://github.com/human-analysis/tensorflownet.git

Since TensorFlowNet relies on several Python packages, you need to make sure that the requirements exist by executing the following command in the *tensorflownet* directory:
> pip install -r requirements.txt

**Notice**
* If you do not have TensorFlow or it does not meet the requirements, please follow the instruction on [the TensorFlow website](https://www.tensorflow.org/install/).

Congratulations!!! You are now ready to use TensorFlowNet!

# Usage
TensorFlowNet comes with a classification example in which a [ResNet](https://arxiv.org/abs/1512.03385) model is trained for the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
> python [main.py](https://github.com/human-analysis/tensorflownet/blob/dev/main.py)
# Configuration
TensorFlowNet loads its parameters at the beginning via a config file and/or the command line.
## Config file
When TensorFlowNet is being run, it will automatically load all parameters from [args.txt](https://github.com/human-analysis/tensorflownet/blob/master/args.txt) by default, if it exists. In order to load a custom config file, the following parameter can be used:
> python main.py --config custom_args.txt
### args.txt
> [Arguments]
>
> log_type = traditional\
> save_results = No\
> \
> \# dataset options\
> dataroot = ./data\
> dataset_train = CIFAR10\
> dataset_test = CIFAR10\
> batch_size = 64


## Command line
Parameters can also be set in the command line when invoking [main.py](https://github.com/human-analysis/tensorflownet/blob/master/main.py). These parameters will precede the existing parameters in the configuration file.
> python [main.py](https://github.com/human-analysis/tensorflownet/blob/master/main.py) --log-type progressbar

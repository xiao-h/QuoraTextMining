import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import yaml

# data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "percentage of the training data to use for validation")

# model hyperparameters
tf.flags.DEFINE_boolean("enable_word_embeddings", True, "Enable/disable the word embeeding (default: True)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_size", "3,4,5", "Comma-seperated filter size (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128)

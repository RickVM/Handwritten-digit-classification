import logging

import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist

from assignments.deeplearning.reading.tensorflow_batch_data import TensorflowBatchData
from assignments.digits.digit_network import DigitNetwork


class NeuralExample:
  def __init__(self):
    tf.logging.set_verbosity(tf.logging.WARN)
    self.save_dir = './data/digit-model'
    self.save_filename = 'digits.ckpt'

  def run(self):
    handwritten_digits = mnist.input_data.read_data_sets("./data/", one_hot=True, reshape=False, validation_size=0)
    training_data_batches = TensorflowBatchData(handwritten_digits.train, batch_size=100)

    neural_network = DigitNetwork().neural_network
    saver = tf.train.Saver()

    with tf.Session() as session:
      logging.info('Training...')
      neural_network.train(session, training_data_batches, num_epochs=10)
      logging.info('Testing...')
      accuracy = neural_network.score(session, handwritten_digits.test.images, handwritten_digits.test.labels)
      logging.info('Accuracy: {}'.format(accuracy))
      logging.info('Saving...')
      saver.save(session, self.save_dir + '/' + self.save_filename)

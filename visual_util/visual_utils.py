import tensorflow as tf
import numpy as np

def variable_summarises(var):
    with tf.name.scope('summaries'):
        mean = tf.reduce_mean(var)
        with tf.name.scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var -mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

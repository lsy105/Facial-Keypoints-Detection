import tensorflow as tf
import numpy as np
from util import *
from tensorflow.tensorboard.tensorboard import main
from resnet import *
from proc_data import *

img_W = 96         #image weight
img_H = 96         #image height
img_C = 1
F_dim = 3          #3x3
pool_dim = 2       #pool layer dimension


reg_rate = 0.6e-3
LR = 0.6e-3
num_class = 30
batch_size = 32
num_epoch = 20
model_path = "face_model_2_6"
test_data_path = "test.csv"

#placeholder layer
X   = tf.placeholder(tf.float32, shape=[None, img_H, img_W, img_C])
y_t = tf.placeholder(tf.float32, shape=[None, num_class])
training = tf.placeholder(tf.bool)

#obtain the score matrix, softmax is not applied yet
output = Resnet(X, num_class, training)

saver = tf.train.Saver()
init_op = tf.group(tf.initialize_all_variables())

with tf.Session() as test:

      test.run(init_op)

      #load process training images
      test_imgs = ReadTestData(test_data_path)

      saver.restore(test, model_path)

      result = list()
      idx = 0
      while idx < test_imgs.shape[0]:
            end = idx + 30
            if end >= test_imgs.shape[0]: end = test_imgs.shape[0]
            img = test_imgs[idx:end]
            img = DatasetMeanNorm(img)
            sample = test.run(output, feed_dict={X: img,
                                                 training: 0})
            for line in sample:
                  result.append(line)
            idx += 30
      PrintTestResult(result, 'test_result.txt')

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
num_epoch = 20
batch_size = 1
model_path = "face_model_tiny_5_3"
test_data_path = "test.csv"

#placeholder layer
X   = tf.placeholder(tf.float32, shape=[None, img_H, img_W, img_C])
y_t = tf.placeholder(tf.float32, shape=[None, num_class])
training = tf.placeholder(tf.bool)

#obtain the score matrix, softmax is not applied yet
output_1 = Resnet(X, num_class, training)
output_2 = Level2(X, batch_size, [16, 16], output_1, 10, training)
output_3 = Level2(X, batch_size, [16, 16], output_1, 7, training)
output_5 = Level2(X, batch_size, [16, 16], output_1, 12, training)
output = output_1 + (output_2 + output_3) * 1.5 / 2
saver = tf.train.Saver()
init_op = tf.group(tf.initialize_all_variables())

with tf.Session() as test:

      test.run(init_op)

      #load process training images
      test_imgs = ReadTestData(test_data_path)
      test_imgs = DatasetMeanNorm(test_imgs)
      saver.restore(test, model_path)

      result = list()
      idx = 0
      while idx < test_imgs.shape[0]:
            end = idx + 1
            img = test_imgs[idx:end]
            sample = test.run(output, feed_dict={X: img,
                                                 training: 0})
            for line in sample:
                  result.append(line)
            idx += 1
      PrintTestResult(result, 'test_result.txt')

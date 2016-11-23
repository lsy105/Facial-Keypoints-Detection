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


reg_rate = 0.0
LR = 0.3e-3
num_class = 30
batch_size = 32
num_epoch = 20

#placeholder layer
X   = tf.placeholder(tf.float32, shape=[None, img_H, img_W, img_C])
y_t = tf.placeholder(tf.float32, shape=[None, num_class])
y_sup = tf.placeholder(tf.float32, shape=[None, num_class])
training = tf.placeholder(tf.bool)

#obtain the score matrix, softmax is not applied yet
output_1 = Resnet(X, num_class, training)
output_2 = Level2(X, batch_size, [32, 64, 128], output_1, 7, training)
output_3 = Level2(X, batch_size, [32, 64, 128], output_2, 5, training)
output = output_1 + ((output_2 - output_1) + (output_3 - output_1)) / 2
loss = MeanSquareLoss(output, y_t, y_sup, reg_rate)
acc = Accuracy(output, y_t, y_sup)
train_step = Train(LR, loss)

#add summary
loss_summary = tf.scalar_summary('loss', loss)
train_acc_summary  = tf.scalar_summary('training accuracy', acc)
val_acc_summary    = tf.scalar_summary('val accuracy', acc)

saver = tf.train.Saver()
init_op = tf.group(tf.initialize_all_variables())

with tf.Session() as test:
      test.run(init_op)
      train_writer = tf.train.SummaryWriter('./resnet_train_tiny', test.graph)
      train_imgs, train_labels, val_imgs, val_labels = ReadData("training.csv")     
      train_imgs = DatasetMeanNorm(train_imgs)
      val_imgs = DatasetMeanNorm(val_imgs) 
      data_size = train_labels.shape[0]
      print_iter = 10
      num_iter = num_epoch * data_size // batch_size 
      for i in range(num_iter):
            img_batch, label_batch, sup_batch = GenBatch(train_imgs, 
                                                         train_labels,
                                                         0,
                                                         batch_size, 
                                                         num_class)
            run_loss, summary = test.run([loss, loss_summary], 
                                         feed_dict={X: img_batch, 
                                                    y_t: label_batch, 
                                                    y_sup: sup_batch,
                                                    training: 1})
            #loss summary
            train_writer.add_summary(summary, i)
            train_step.run(feed_dict={X: img_batch, 
                                     y_t: label_batch, 
                                     y_sup: sup_batch,
                                     training: 1})

            if i % print_iter == 0:
                  if i == 0: print (run_loss)
                  
                  #training acc summary
                  train_summary = test.run(train_acc_summary, 
                                           feed_dict={X: img_batch, 
                                           y_t: label_batch,
                                           y_sup: sup_batch, 
                                           training: 0})
                  train_writer.add_summary(train_summary, i)

                  #val acc summary
                  img_batch, label_batch, sup_batch = GenBatch(val_imgs, 
                                                               val_labels, 
                                                               0,
                                                               batch_size, 
                                                               num_class)       

                  val_summary = test.run(val_acc_summary, 
                                         feed_dict={X: img_batch, 
                                         y_t: label_batch, 
                                         y_sup: sup_batch,
                                         training: 0})
                  train_writer.add_summary(val_summary, i)
            i += 1
            if i % 1000 == 0 and i > 0: saver.save(test, './face_model_tiny')

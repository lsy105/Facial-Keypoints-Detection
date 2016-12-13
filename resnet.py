from util import *
from layers import *
import tensorflow as tf


def Res3x3(x, out_C, is_pooling, training_mode):
      """
      create a residual net with 2 layers
  
      Args:
            x: input of residual net
            out_C: number of output channels for this single residual net
            is_pooling: if do pooling in the begining of this net 
                        (2x2 pool with stride 2)

      Returns:
            conv_out: a tensor with shape [N, x_W, x_H, out_C] x_W is 
            width of x, x_H is height of x
      """

      if is_pooling:
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                               strides=[1, 2, 2, 1], 
                               padding='SAME')
  
      x_size, x_H, x_W, x_C = x.get_shape()
      x_H, x_W, x_C = int(x_H), int(x_W), int(x_C)

      #first conv layer
      conv_out = ConvNormReluForward(x, out_C, training_mode)

      """
      #add dropout
      if training_mode is not None:
            conv_out = tf.nn.dropout(conv_out, 0.8)
      """

      #second conv layer without relu
      conv_out = ConvNormForward(conv_out, out_C, training_mode)

      #shortcut path
      if x_C != out_C:
            x = Conv1x1(x, out_C, training_mode)
      conv_out += x
      conv_out = tf.nn.relu(conv_out)
      return conv_out                        
                          
                                                      

def TinyRegionResnet(img, batch_size, x, y, out_C, local_region_size, training_mode):
      """
      Run resnet neural network on small area in image
      centered at x and y with width 10

      Args:
           img: original img
           x: x coordinate
           y: y coordinate
           out_C: output_channel
           local_region_size: size of local region in image
           training_mode: it it is in training

      Return:
           new_x: adjusted x
           new_y: adjusted y
      """
      local_size = local_region_size
      int_x, int_y = tf.cast(x, tf.int32), tf.cast(y, tf.int32)
      tiny_img = [list() for i in range(batch_size)]
      for idx in range(batch_size):
            low_x = tf.maximum(0, int_x[idx] - local_size//2)
            low_y = tf.maximum(0, int_y[idx] - local_size//2)
            low_x = tf.minimum(low_x, 96 - local_size)
            low_y = tf.minimum(low_y, 96 - local_size)
            tiny_img[idx] = tf.slice(img[idx], [low_x, low_y, 0], 
                                     [local_size, local_size, 1]) 
      tiny_img = tf.pack(tiny_img)
      conv_out = Res3x3(tiny_img, out_C[0], 0, training_mode)
      conv_out = Res3x3(conv_out, out_C[1], 0, training_mode)
      last_layer_size = out_C[-1] * local_size * local_size
      conv_out = tf.reshape(conv_out, [-1, last_layer_size])
      output = FCLayer(conv_out, 2, 1.0, 1, training_mode)
      return output
     

 
def Resnet(x, num_class, training_mode):
      """
      build 16 layers residual neural network
      Args:
            x: input with shape [batch_size, width, height, channel]
            training_mode: if in training mode

      Returns:
            return the score matrix with shape [batch_size, num_class(200)] 
      """
      #define number of output channel of filters in each resdidual unit
      filter_list  = [64, 64, 128, 128, 256, 256, 256]                         
      #define if do pooling at current residual unit(1: do pooling)
      if_pool_list = [ 1,  0,  1,  0,  1,  0,  0]                
      num_pool = sum(if_pool_list)
     
      #get shape of x
      x_size, x_W, x_H, x_C = x.get_shape()     
      x_H, x_W, x_C = int(x_H), int(x_W), int(x_C)

      #verify if there are too many pooling layer
      assert x_W % (2**num_pool) == 0, "too many pooling layer"
 
      #first conv layer 3x3 filter
      with tf.name_scope('first_layer'):
            conv_out = ConvNormReluForward(x, filter_list[0], training_mode)
            conv_out = ConvNormReluForward(conv_out, filter_list[0], training_mode)
      
      #build all residual units
      with tf.name_scope('residual_network'):  
            for idx in range(len(filter_list)):
                  with tf.name_scope('unit_%d_%d' % (idx, filter_list[idx])):
                        conv_out = Res3x3(conv_out, filter_list[idx], 
                                          if_pool_list[idx], training_mode)             
      
      #last layer
      with tf.name_scope('output_layer'):
            last_layer_size = filter_list[-1] * (x_W // (2**num_pool))**2 
            conv_out = tf.reshape(conv_out, [-1, last_layer_size])
            output = FCLayer(conv_out, num_class, 1.0, 1, training_mode)

      return output



def Level2(x, batch_size, out_C, x_y_value, local_region_size, training_mode):
      #tiny region layer
      with tf.name_scope('tiny_region_layer'):
            output_list = [list() for i in range(batch_size)]
            idx = 0
            while idx < 30 :
                  temp_out = TinyRegionResnet(x, batch_size, 
                                              x_y_value[:, idx],
                                              x_y_value[:, idx + 1], 
                                              out_C,
                                              local_region_size, 
                                              training_mode)
                  for i in range(batch_size):
                        output_list[i].append(temp_out[i, 0])
                        output_list[i].append(temp_out[i, 1])
                  idx += 2
      output = tf.pack(output_list)
      return output

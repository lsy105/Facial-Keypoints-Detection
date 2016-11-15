import tensorflow as tf
import numpy as np

def InitWeights(shape, name=''):
      """  
      initilize weights with zero mean and std provided by user
      calibrate the variance with sqrt(2/n) so that output has the same
      variance as input. n is the fan-in of the weight.
            
      Args:
            shape: shape of the weight matrix
                   weight of convolution layer [F_dim, F_dim, 
                                                num_input_feature_map, 
                                                num_output_feature_map]
            name:  name of this variable

      Returns:
            weight_matrix: tensorflow variables with the same shape as user specified
                           and initial values with zero mean and std specified by user 
      """
      #calculate fan-in and fan_out of weight matrix
      shape = list(map(int, shape))
      fan_in = 0
      fan_out = 0
      if(len(shape) == 2):
            fan_in = shape[0]
            fan_out = shape[1]
      else:
            fan_in  = shape[0] * shape[1] * shape[2]
            fan_out = shape[0] * shape[1] * shape[3]
      weight_matrix = tf.truncated_normal(shape, mean=0.0, stddev=tf.sqrt(2.0/(fan_in + fan_out)))
      return tf.Variable(weight_matrix, name=name)



def InitBias(shape, name=''):
      """  
      initilize bias with 0.0
      
      Args:
            shape: shape of the bias matrix
            name:  name of this bias 
 
      Returns:
            bias_matrix: tensorflow variables with the same shape as user specified
                         and initial values with zero mean and std specified by user 
      """

      bias_matrix = tf.constant(0.0, shape = shape)
      return tf.Variable(bias_matrix, name=name)



def Loss(score_matrix, bool_matrix, reg_rate):
      """
      calculate loss
      
      Args:
            score_matrix: score_matrix with shape[batch_size, num_class]
            bool_matrix:  matrix with shape[batch_size, num_class]
                          most elements are zeros except the correct class element 
            reg_rate:     regulation rate for L2 loss 
      Returns:
            loss: the total loss 
      """
      output = tf.nn.softmax(score_matrix)
      #calculate loss with L2
      loss = tf.reduce_mean(-tf.reduce_sum(bool_matrix * tf.log(output), reduction_indices=[1]))

      #calculate weight L2 loss
      W_loss = 0.0
      for W in tf.trainable_variables():
            if W.op.name.find(r'AW') > 0:
                  print (W.op.name)
                  W_loss += tf.nn.l2_loss(W)      
      loss += reg_rate * W_loss
      print (reg_rate * W_loss)
      return loss



def MeanSquareLoss(score_matrix, truth_matrix, sup_matrix, reg_rate):
      """
      calculate loss
      
      Args:
            score_matrix: score_matrix with shape[batch_size, num_class]
            bool_matrix:  matrix with shape[batch_size, num_class]
                          most elements are zeros except the correct class element 
            reg_rate:     regulation rate for L2 loss 
      Returns:
            loss: the total loss 
      """
      
      #set empty target equal to the corresponding element in score matrix
      #so the loss of this element is 0
      sub_score_matrix = tf.mul(sup_matrix, score_matrix)
      truth_matrix += sub_score_matrix
      #calculate loss with L2
      temp_loss = tf.reduce_sum(0.5 * tf.square(score_matrix - truth_matrix), 1)
      loss = tf.reduce_mean(temp_loss, 0)
      
      #calculate weight L2 loss
      W_loss = 0.0
      for W in tf.trainable_variables():
            if W.op.name.find(r'AW') > 0:
                  print (W.op.name)
                  W_loss += tf.nn.l2_loss(W)      
      loss += reg_rate * W_loss
      print (reg_rate * W_loss)
      return tf.sqrt(loss)



def Accuracy(score_matrix, truth_matrix, sup_matrix):
      """
      calculate accuracy for batch
      
      Args:
            score_matrix: score_matrix
            y_t   : truth matrix

      Returns:
            accuracy: accuracy rate
      """
      sub_score_matrix = tf.mul(sup_matrix, score_matrix)
      truth_matrix += sub_score_matrix
      error = tf.reduce_sum(tf.sqrt(truth_matrix - score_matrix), 1)
      accuracy = tf.sqrt(tf.reduce_mean(error, 0))
      return accuracy



def FlipImageandLabel(image, label):
      """
      Flip image in left/right direction
      Args:
            image: input image with type np.array
            label: input label with type np.array
      Returns:
            new_image: flipped input image
            label: flipped input label
      """
      new_image = np.fliplr(image)
      new_label = np.zeros((label.shape), np.float32)
      idx = 0
      while idx < 30:
            if label[idx] == 0:
                  new_label[idx + 1] = label[idx + 1]
                  idx += 2
                  continue
            new_label[idx] = 96 - label[idx]
            new_label[idx + 1] = label[idx + 1]
            idx += 2
      return new_image, new_label


      
def RotateLabel90(label, k=1):
      """
      Rotate labels by k * 90 degree in the counter-clockwise direction

      Args:
            label: label array
            k: number of 90 degree rotation

      Returns:
            rot_label: rotated label array
      """
      num_90 = k % 4
      idx = 0
      size = label.shape
      new_label = np.empty((size), np.float32)
      len_row = 96
      len_col = 96
      if num_90 == 1:
            while idx < 30:
                  row = label[idx + 1]
                  col = label[idx]
                  new_label[idx] = row
                  new_label[idx + 1] = len_col - col - 1
                  if col == 0.0: 
                        new_label[idx + 1] = 0.0
                  idx += 2

      if num_90 == 2:
            while idx < 30:
                  row = label[idx + 1]
                  col = label[idx]
                  new_label[idx] = len_col - col - 1
                  new_label[idx + 1] = len_row - row - 1
                  if col == 0.0: 
                        new_label[idx] = 0.0
                  if row == 0.0:
                        new_label[idx + 1] = 0.0
                  idx += 2
      
      if num_90 == 3:
            while idx < 30:
                  row = label[idx + 1]
                  col = label[idx]
                  new_label[idx] = len_row - row - 1
                  new_label[idx + 1] = col
                  if row == 0.0: 
                        new_label[idx] = 0.0
                  idx += 2
      return new_label



def GenBatch(img_in, labels, if_rotate, batch_size=32, num_class=200):
      """
      Generate a batch
      Args:
            img_in: image dataset
            labels: labels for images with shape [N, 1]
            batch_size: size of batch
      Returns:
            img_out: a np.array of image data with shape[batch_size, img_width, 
                                                         img_height, img_channel]
            label_matrix: label np.array with shape[batch_size, num_class]
      """
      if not if_rotate:
            idx = np.random.choice(labels.shape[0], batch_size)
            img_out = img_in[idx]
            label_matrix = labels[idx]
            sup_matrix = np.copy(label_matrix)
            sup_matrix[sup_matrix > 0] = 0.0
            sup_matrix[sup_matrix < 0] = 1.0
            label_matrix += sup_matrix
            return img_out, label_matrix, sup_matrix

      #create a label matrix with shape [batch_size, num_class]
      idx = np.random.choice(labels.shape[0], batch_size)
      img_out = img_in[idx]
      label_matrix = labels[idx]
      sup_matrix = np.copy(label_matrix)
      sup_matrix[sup_matrix > 0] = 0.0
      sup_matrix[sup_matrix < 0] = 1.0
      label_matrix += sup_matrix

      """
      #Rotate images and labels to create more data
      new_img_out = np.empty((batch_size, 96, 96, 1))
      new_label_matrix = np.empty((batch_size, 30))
      for idx in range(batch_size):      
            #k = np.random.randint(3) + 1
            new_img_out[idx], new_label_matrix[idx] = FlipImageandLabel(img_out[idx],
                                                                        label_matrix[idx])
            #new_label_matrix[idx] = RotateLabel90(label_matrix[idx], k)
            #new_img_out[idx] = np.rot90(img_out[idx], k)
      label_matrix = np.concatenate((label_matrix, new_label_matrix), axis=0)  
      img_out = np.concatenate((img_out, new_img_out), axis=0) 
      sup_matrix = np.concatenate((sup_matrix, sup_matrix), axis=0) 
      """
      #return result
      return img_out, label_matrix, sup_matrix



def Train(LR, loss):
      """
      training the model(use adam algorithm)

      Args:
            LR: learning rate
            loss: scalar loss of this model
      Returns:
            train_step: training operation
      """
      train_step = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)
      return train_step



def TestDataLookUp(path):
      """
      Create list of list locations for test images
  
      Args:
            path: path to the LookUpTable File which generated by 
                  proc.sh

      Returns:
            result: list of list of locations for each test image 
      """
      with open(path, 'r') as fin:
            result = dict()
            for line in fin:
                  idx = line.split(' ')
                  idx[1], idx[2] = int(idx[1]), int(idx[2])
                  if int(idx[1]) in result:
                        result[idx[1]].append(idx[2] - 1) 
                  else:
                        result[idx[1]] = list()
                        result[idx[1]].append(idx[2] - 1)
      fin.close()
      return result     



def PrintTestResult(result_list, file_name):
      """
      write the output to text file

      Args:
            result_list: list of result
            file_name: output file name
      """
      with open (file_name, 'w') as fo:
            size = len(result_list[0])
            p = 1
            table = TestDataLookUp('./table.txt')
            fo.write('RowId Location\n')
            for element in result_list:
                  for idx in table[p]:
                        line = str(p) + ' ' + str(element[idx]) + '\n'
                        fo.write(line)                         
                        p += 1
      fo.close()

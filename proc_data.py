import numpy as np
import csv as csv
import random
from PIL import Image

def DatasetMeanNorm(dataset):
      """
      subtract mean and do normalization for each image
      Args:
            dataset: dataset with shape [N, H, W, C]
                     N data points, Height H, Width W, Channel C

      Return:
            dataset: processed dataset
      """
      N, H, W, C = dataset.shape
      X = dataset.reshape((N, -1))
      X -= np.mean(X, axis = 0)
      X /= np.std(X, axis = 0)
      X = X.reshape((N, H, W, C))
      return X

def ReadData(path):
      read_data = csv.reader(open(path))
      i = 0
      train_img_list = list()
      train_label_list = list()
      val_img_list = list()
      val_label_list = list()
      size = int(sum(1 for line in open(path)) * 0.8)
      read_data = list(read_data)[1:]
      random.shuffle(read_data)
      for row in read_data:
            if i == 0: 
                  i += 1
                  continue
            array = np.fromstring(row[-1], dtype=np.int8, sep=' ')
            array = np.reshape(array, (96, 96, 1))
            if i <= size:
                  train_img_list.append(array)
                  labels =[float(x = '-1.0') if not x else float(x) for x in row[0:30]]
                  train_label_list.append(labels)
            else:
                  val_img_list.append(array)
                  labels =[float(x = '-1.0') if not x else float(x) for x in row[0:30]]
                  val_label_list.append(labels)
            i += 1

      return (np.array(train_img_list, dtype=np.float32),
              np.array(train_label_list, dtype=np.float32),
              np.array(val_img_list, dtype=np.float32),
              np.array(val_label_list, dtype=np.float32))


def ReadTestData(path):
      read_data = csv.reader(open(path))
      i = 0
      test_img_list = list()
      test_label_list = list()
      for row in read_data:
            if i == 0: 
                  i += 1
                  continue
            array = np.fromstring(row[-1], dtype=np.int8, sep=' ')
            array = np.reshape(array, (96, 96, 1))
            test_img_list.append(array)
            i += 1

      return np.array(test_img_list, dtype=np.float32)

